/**
 * viewer.js — Three.js point cloud viewer with CLIP text queries
 *
 * Data files expected in ./data/:
 *   scene.bin            Single binary (see export_web_data.py for layout)
 *   cluster_centers.bin  Float32, K×D, L2-normalized
 *   metadata.json        n_points, n_clusters, feature_dim, bbox_min, bbox_max
 *   models/              CLIP ONNX files (from download_clip_model.py)
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { AutoTokenizer, CLIPTextModelWithProjection, env } from
  'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

// Load CLIP from local files (committed to repo) — no HuggingFace download
env.localModelPath   = './data/models/';
env.allowLocalModels = true;
env.allowRemoteModels = true;  // flip to true as fallback if local files missing

// ── State ─────────────────────────────────────────────────────────────────────
let renderer, scene, camera, controls, pointCloud, geometry;
let nPoints = 0, nClusters = 0, featureDim = 0;
let rgbColorsF32  = null;   // Float32Array N*3, [0,1]
let clusterIds    = null;   // Int32Array N  (-1 = unlabeled)
let clusterCenters = null;  // Float32Array K*D, L2-normalized
let currentSims   = null;   // Float32Array N
let metadata      = null;
let currentView   = 'rgb';

// CLIP
let clipTokenizer = null;
let clipModel     = null;
let modelReady    = false;
let dataReady     = false;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const overlay        = document.getElementById('loading-overlay');
const loadingMsg     = document.getElementById('loading-msg');
const loadingSub     = document.getElementById('loading-sub');
const modelDot       = document.getElementById('model-dot');
const modelText      = document.getElementById('model-text');
const dataDot        = document.getElementById('data-dot');
const dataText       = document.getElementById('data-text');
const queryInput     = document.getElementById('query-input');
const searchBtn      = document.getElementById('search-btn');
const resultsSection = document.getElementById('results-section');

// ── Entry point ───────────────────────────────────────────────────────────────
async function init() {
  setupThreeJS();
  animate();
  await Promise.all([loadData(), loadCLIP()]);
  overlay.classList.add('hidden');
  enableSearch();
}

// ── Three.js setup ─────────────────────────────────────────────────────────────
function setupThreeJS() {
  const wrap = document.getElementById('canvas-wrap');

  renderer = new THREE.WebGLRenderer({ antialias: false });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(wrap.clientWidth, wrap.clientHeight);
  renderer.setClearColor(0x0d0d0d);
  wrap.appendChild(renderer.domElement);

  scene  = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(60, wrap.clientWidth / wrap.clientHeight, 0.1, 100000);
  camera.position.set(0, -400, 600);
  camera.up.set(0, 0, 1);
  camera.lookAt(0, 0, 0);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.update();

  window.addEventListener('resize', () => {
    camera.aspect = wrap.clientWidth / wrap.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(wrap.clientWidth, wrap.clientHeight);
  });
}

// ── Load scene.bin + cluster_centers.bin ──────────────────────────────────────
async function loadData() {
  setStatus(dataDot, dataText, 'yellow', 'Loading metadata…');
  metadata  = await fetchJSON('./data/metadata.json');
  nPoints   = metadata.n_points;
  nClusters = metadata.n_clusters;
  featureDim = metadata.feature_dim;

  const bboxMin = metadata.bbox_min;   // [x, y, z]
  const bboxMax = metadata.bbox_max;

  setStatus(dataDot, dataText, 'yellow', `Loading scene (${(nPoints/1e6).toFixed(1)}M pts)…`);
  setOverlay(`Loading scene (${(nPoints/1e6).toFixed(1)}M pts)…`);

  const [sceneBuf, centersBuf] = await Promise.all([
    fetchBin('./data/scene.bin'),
    fetchBin('./data/cluster_centers.bin'),
  ]);

  // ── Parse scene.bin ────────────────────────────────────────────────────────
  // Layout: [uint32 n][uint16 xyz N×3][uint8 rgb N×3][pad?][uint16 ids N]
  const N       = new Uint32Array(sceneBuf, 0, 1)[0];
  const xyzU16  = new Uint16Array(sceneBuf, 4, N * 3);

  const rgbByteOffset = 4 + N * 6;
  const rgbU8   = new Uint8Array(sceneBuf, rgbByteOffset, N * 3);

  const idsRaw  = rgbByteOffset + N * 3;
  const idsAligned = (idsRaw % 2 === 0) ? idsRaw : idsRaw + 1;
  const idsU16  = new Uint16Array(sceneBuf, idsAligned, N);

  // ── Dequantize uint16 XYZ → float32 ───────────────────────────────────────
  const positions = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    for (let d = 0; d < 3; d++) {
      positions[i*3+d] = (xyzU16[i*3+d] / 65535) * (bboxMax[d] - bboxMin[d]) + bboxMin[d];
    }
  }

  // ── RGB uint8 → float32 [0,1] ─────────────────────────────────────────────
  rgbColorsF32 = new Float32Array(N * 3);
  for (let i = 0; i < N * 3; i++) rgbColorsF32[i] = rgbU8[i] / 255;

  // ── Cluster IDs (65535 sentinel → -1) ─────────────────────────────────────
  clusterIds = new Int32Array(N);
  for (let i = 0; i < N; i++) clusterIds[i] = idsU16[i] === 65535 ? -1 : idsU16[i];

  clusterCenters = new Float32Array(centersBuf);

  // ── Build Three.js geometry ────────────────────────────────────────────────
  geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color',    new THREE.BufferAttribute(rgbColorsF32.slice(), 3));

  const extent = Math.max(bboxMax[0] - bboxMin[0], bboxMax[1] - bboxMin[1]);
  const material = new THREE.PointsMaterial({
    size:            Math.max(0.5, extent / 800),
    vertexColors:    true,
    sizeAttenuation: true,
  });

  pointCloud = new THREE.Points(geometry, material);
  scene.add(pointCloud);

  // Camera: bird's-eye above center
  const cx = (bboxMin[0] + bboxMax[0]) / 2;
  const cy = (bboxMin[1] + bboxMax[1]) / 2;
  camera.position.set(cx, cy - extent * 0.3, bboxMax[2] + extent * 0.6);
  camera.lookAt(cx, cy, 0);
  controls.target.set(cx, cy, 0);
  controls.update();

  setStatus(dataDot, dataText, 'green', `${(N/1e6).toFixed(1)}M points loaded`);
  dataReady = true;
}

// ── Load CLIP model (from local files) ────────────────────────────────────────
async function loadCLIP() {
  const modelId = 'Xenova/clip-vit-large-patch14';

  setStatus(modelDot, modelText, 'yellow', 'Loading CLIP tokenizer…');
  try {
    clipTokenizer = await AutoTokenizer.from_pretrained(modelId);
  } catch (e) {
    // Local files missing — fall back to remote
    console.warn('Local CLIP files not found, falling back to HuggingFace…', e);
    env.allowRemoteModels = true;
    clipTokenizer = await AutoTokenizer.from_pretrained(modelId);
  }

  setStatus(modelDot, modelText, 'yellow', 'Loading CLIP text encoder…');

  clipModel = await CLIPTextModelWithProjection.from_pretrained(modelId, {
    quantized: true,
    progress_callback: (p) => {
      if (p.status === 'downloading' && p.file) {
        const pct = p.progress != null ? ` ${Math.round(p.progress)}%` : '';
        const mb  = p.total ? ` (${(p.total/1e6).toFixed(0)} MB)` : '';
        setStatus(modelDot, modelText, 'yellow', `Downloading CLIP${mb}…${pct}`);
        setOverlay(`Downloading CLIP${mb}…${pct}`, 'One-time download — cached afterwards');
      } else if (p.status === 'loading') {
        setStatus(modelDot, modelText, 'yellow', 'Loading CLIP…');
        setOverlay('Loading CLIP into memory…');
      }
    },
  });

  setStatus(modelDot, modelText, 'green', 'CLIP ready');
  modelReady = true;
}

// ── Handle search ─────────────────────────────────────────────────────────────
async function handleSearch(e) {
  e.preventDefault();
  const query = queryInput.value.trim();
  if (!query || !modelReady || !dataReady) return;

  queryInput.disabled   = true;
  searchBtn.disabled    = true;
  searchBtn.textContent = '…';

  try {
    // 1. Encode text
    const inputs = clipTokenizer([query], { padding: true, truncation: true });
    const { text_embeds } = await clipModel(inputs);
    const rawEmbed = text_embeds.data;  // Float32Array D

    // 2. L2-normalize
    let norm = 0;
    for (let d = 0; d < featureDim; d++) norm += rawEmbed[d] * rawEmbed[d];
    norm = Math.sqrt(norm);
    const queryVec = new Float32Array(featureDim);
    for (let d = 0; d < featureDim; d++) queryVec[d] = rawEmbed[d] / (norm + 1e-8);

    // 3. Cosine similarity vs cluster centers (already L2-normalized)
    const clusterSims = dotProductBatch(queryVec, clusterCenters, nClusters, featureDim);

    // 4. Map cluster → point
    const simMin = clusterSims.reduce((a, b) => Math.min(a, b), Infinity);
    const simMax = clusterSims.reduce((a, b) => Math.max(a, b), -Infinity);
    currentSims = new Float32Array(nPoints);
    for (let i = 0; i < nPoints; i++) {
      const cid = clusterIds[i];
      currentSims[i] = cid >= 0 ? clusterSims[cid] : simMin;
    }

    // 5. UI
    document.getElementById('stat-query').textContent = `"${query}"`;
    document.getElementById('cb-min').textContent     = simMin.toFixed(3);
    document.getElementById('cb-max').textContent     = simMax.toFixed(3);
    resultsSection.style.display = '';
    setView('heatmap');

  } catch (err) {
    console.error('Query failed:', err);
    alert(`Query failed: ${err.message}`);
  }

  queryInput.disabled   = false;
  searchBtn.disabled    = false;
  searchBtn.textContent = 'Go';
}

// ── View toggle ───────────────────────────────────────────────────────────────
window.setView = function(mode) {
  currentView = mode;
  document.getElementById('btn-heatmap').classList.toggle('active', mode === 'heatmap');
  document.getElementById('btn-rgb').classList.toggle('active',     mode === 'rgb');
  applyColors();
};

function applyColors() {
  if (!geometry) return;
  const arr = geometry.attributes.color.array;

  if (currentView === 'rgb' || !currentSims) {
    for (let i = 0; i < nPoints * 3; i++) arr[i] = rgbColorsF32[i];
  } else {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < nPoints; i++) {
      if (currentSims[i] < lo) lo = currentSims[i];
      if (currentSims[i] > hi) hi = currentSims[i];
    }
    const range = hi - lo || 1;
    for (let i = 0; i < nPoints; i++) {
      const t = (currentSims[i] - lo) / range;
      const [r, g, b] = coolwarm(t);
      arr[i*3] = r; arr[i*3+1] = g; arr[i*3+2] = b;
    }
  }
  geometry.attributes.color.needsUpdate = true;
}

// ── Colormap ──────────────────────────────────────────────────────────────────
function coolwarm(t) {
  if (t < 0.5) { const s = t*2; return [s, s, 1.0]; }
  else          { const s = (t-0.5)*2; return [1.0, 1-s, 1-s]; }
}

// ── Math ──────────────────────────────────────────────────────────────────────
function dotProductBatch(queryVec, centers, K, D) {
  const sims = new Float32Array(K);
  for (let k = 0; k < K; k++) {
    let dot = 0;
    const base = k * D;
    for (let d = 0; d < D; d++) dot += centers[base+d] * queryVec[d];
    sims[k] = dot;
  }
  return sims;
}

// ── Fetch helpers ─────────────────────────────────────────────────────────────
async function fetchBin(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  return res.arrayBuffer();
}
async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`);
  return res.json();
}

// ── UI helpers ────────────────────────────────────────────────────────────────
function setStatus(dot, textEl, color, msg) {
  dot.className = `dot ${color}`;
  textEl.textContent = msg;
}
function setOverlay(msg, sub = '') {
  loadingMsg.textContent = msg;
  loadingSub.textContent = sub;
}
function enableSearch() {
  queryInput.disabled = false;
  searchBtn.disabled  = false;
  queryInput.focus();
  overlay.classList.add('hidden');
}

// ── Render loop ───────────────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

document.getElementById('search-form').addEventListener('submit', handleSearch);

init().catch(err => {
  loadingMsg.textContent = 'Error loading data';
  loadingSub.textContent = err.message;
  console.error(err);
});
