/**
 * viewer.js — Three.js point cloud viewer with CLIP text queries
 *
 * Data files expected in ./data/:
 *   xyz.bin              Float32, N×3, centered coordinates
 *   rgb.bin              Uint8,   N×3
 *   cluster_ids.bin      Int32,   N
 *   cluster_centers.bin  Float32, K×D, L2-normalized
 *   metadata.json
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { AutoTokenizer, CLIPTextModelWithProjection, env } from
  'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

// Force remote model loading (don't look for local /models/ folder)
env.allowLocalModels = false;

// ── State ───────────────────────────────────────────────────────────────────
let renderer, scene, camera, controls, pointCloud, geometry;
let nPoints = 0, nClusters = 0, featureDim = 0;
let rgbColorsF32   = null;   // Float32Array N*3, values 0-1 (for Three.js)
let clusterIds     = null;   // Int32Array N
let clusterCenters = null;   // Float32Array K*D, L2-normalized
let currentSims    = null;   // Float32Array N, current per-point similarities
let metadata       = null;
let currentView    = 'rgb';  // 'rgb' | 'heatmap'

// CLIP
let clipTokenizer = null;
let clipModel     = null;
let modelReady    = false;
let dataReady     = false;

// ── DOM refs ─────────────────────────────────────────────────────────────────
const overlay      = document.getElementById('loading-overlay');
const loadingMsg   = document.getElementById('loading-msg');
const loadingSub   = document.getElementById('loading-sub');
const modelDot     = document.getElementById('model-dot');
const modelText    = document.getElementById('model-text');
const dataDot      = document.getElementById('data-dot');
const dataText     = document.getElementById('data-text');
const queryInput   = document.getElementById('query-input');
const searchBtn    = document.getElementById('search-btn');
const resultsSection = document.getElementById('results-section');

// ── Entry point ──────────────────────────────────────────────────────────────
async function init() {
  setupThreeJS();
  animate();

  // Load data + model in parallel
  await Promise.all([
    loadData(),
    loadCLIP(),
  ]);

  overlay.classList.add('hidden');
  enableSearch();
}

// ── Three.js setup ────────────────────────────────────────────────────────────
function setupThreeJS() {
  const wrap = document.getElementById('canvas-wrap');

  renderer = new THREE.WebGLRenderer({ antialias: false });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(wrap.clientWidth, wrap.clientHeight);
  renderer.setClearColor(0x0d0d0d);
  wrap.appendChild(renderer.domElement);

  scene  = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(60, wrap.clientWidth / wrap.clientHeight, 0.1, 100000);

  // Start above the scene looking down; updated once data loads
  camera.position.set(0, -400, 600);
  camera.up.set(0, 0, 1);   // Z is up for outdoor LiDAR
  camera.lookAt(0, 0, 0);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 0, 0);
  controls.update();

  window.addEventListener('resize', () => {
    camera.aspect = wrap.clientWidth / wrap.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(wrap.clientWidth, wrap.clientHeight);
  });
}

// ── Load binary data ──────────────────────────────────────────────────────────
async function loadData() {
  setStatus(dataDot, dataText, 'yellow', 'Loading metadata…');

  metadata   = await fetchJSON('./data/metadata.json');
  nPoints    = metadata.n_points;
  nClusters  = metadata.n_clusters;
  featureDim = metadata.feature_dim;

  setStatus(dataDot, dataText, 'yellow', `Loading point cloud (${(nPoints/1e6).toFixed(2)}M pts)…`);
  setOverlay(`Loading point cloud (${(nPoints/1e6).toFixed(2)}M pts)…`);

  const [xyzBuf, rgbBuf, idsBuf, centersBuf] = await Promise.all([
    fetchBin('./data/xyz.bin'),
    fetchBin('./data/rgb.bin'),
    fetchBin('./data/cluster_ids.bin'),
    fetchBin('./data/cluster_centers.bin'),
  ]);

  const xyz     = new Float32Array(xyzBuf);       // N*3
  const rgbU8   = new Uint8Array(rgbBuf);          // N*3
  clusterIds    = new Int32Array(idsBuf);          // N
  clusterCenters = new Float32Array(centersBuf);  // K*D

  // Convert uint8 RGB → float32 [0,1] for Three.js vertex colors
  rgbColorsF32 = new Float32Array(nPoints * 3);
  for (let i = 0; i < nPoints * 3; i++) {
    rgbColorsF32[i] = rgbU8[i] / 255.0;
  }

  // Build Three.js BufferGeometry
  geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(xyz.slice(), 3));
  geometry.setAttribute('color',    new THREE.BufferAttribute(rgbColorsF32.slice(), 3));

  const material = new THREE.PointsMaterial({
    size:         computePointSize(metadata.bbox),
    vertexColors: true,
    sizeAttenuation: true,
  });

  pointCloud = new THREE.Points(geometry, material);
  scene.add(pointCloud);

  // Set camera to look at bbox center from above
  const bbox = metadata.bbox;
  const cx = (bbox.min[0] + bbox.max[0]) / 2;
  const cy = (bbox.min[1] + bbox.max[1]) / 2;
  const extent = Math.max(bbox.max[0] - bbox.min[0], bbox.max[1] - bbox.min[1]);
  camera.position.set(cx, cy - extent * 0.3, bbox.max[2] + extent * 0.6);
  camera.lookAt(cx, cy, 0);
  controls.target.set(cx, cy, 0);
  controls.update();

  setStatus(dataDot, dataText, 'green', `${(nPoints/1e6).toFixed(2)}M points loaded`);
  dataReady = true;
}

// ── Load CLIP model ───────────────────────────────────────────────────────────
async function loadCLIP() {
  const modelId = 'Xenova/clip-vit-large-patch14';

  setStatus(modelDot, modelText, 'yellow', 'Loading CLIP tokenizer…');
  clipTokenizer = await AutoTokenizer.from_pretrained(modelId);

  setStatus(modelDot, modelText, 'yellow', 'Downloading CLIP text encoder…');

  clipModel = await CLIPTextModelWithProjection.from_pretrained(modelId, {
    quantized: true,
    progress_callback: (p) => {
      if (p.status === 'downloading' && p.file) {
        const pct = p.progress != null ? ` ${Math.round(p.progress)}%` : '';
        const mb  = p.total  ? ` (${(p.total / 1e6).toFixed(0)} MB)` : '';
        setStatus(modelDot, modelText, 'yellow', `Downloading CLIP${mb}…${pct}`);
        setOverlay(`Downloading CLIP text encoder${mb}…${pct}`,
                   'First visit only — cached in browser afterwards');
      } else if (p.status === 'loading') {
        setStatus(modelDot, modelText, 'yellow', 'Loading CLIP into memory…');
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

  queryInput.disabled = true;
  searchBtn.disabled  = true;
  searchBtn.textContent = '…';

  try {
    // 1. Encode text with CLIP
    const inputs = clipTokenizer([query], { padding: true, truncation: true });
    const { text_embeds } = await clipModel(inputs);
    const rawEmbed = text_embeds.data;  // Float32Array, length = feature_dim

    // 2. L2-normalize query embedding
    let norm = 0;
    for (let d = 0; d < featureDim; d++) norm += rawEmbed[d] * rawEmbed[d];
    norm = Math.sqrt(norm);
    const queryVec = new Float32Array(featureDim);
    for (let d = 0; d < featureDim; d++) queryVec[d] = rawEmbed[d] / (norm + 1e-8);

    // 3. Cosine similarity vs cluster centers (already L2-normalized)
    const clusterSims = dotProductBatch(queryVec, clusterCenters, nClusters, featureDim);

    // 4. Map cluster similarities → per-point similarities
    const simMin = clusterSims.reduce((a, b) => Math.min(a, b), Infinity);
    currentSims = new Float32Array(nPoints);
    for (let i = 0; i < nPoints; i++) {
      const cid = clusterIds[i];
      currentSims[i] = cid >= 0 ? clusterSims[cid] : simMin;
    }

    // 5. Update stats UI
    const simMax = currentSims.reduce((a, b) => Math.max(a, b), -Infinity);

    document.getElementById('stat-query').textContent = `"${query}"`;
    document.getElementById('cb-min').textContent     = simMin.toFixed(3);
    document.getElementById('cb-max').textContent     = simMax.toFixed(3);

    resultsSection.style.display = '';
    setView('heatmap');

  } catch (err) {
    console.error('Query failed:', err);
    alert(`Query failed: ${err.message}`);
  }

  queryInput.disabled = false;
  searchBtn.disabled  = false;
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
  const colorAttr = geometry.attributes.color;
  const arr = colorAttr.array;

  if (currentView === 'rgb' || !currentSims) {
    for (let i = 0; i < nPoints * 3; i++) arr[i] = rgbColorsF32[i];
  } else {
    // Heatmap — normalize to [0,1] then apply coolwarm colormap
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < nPoints; i++) {
      if (currentSims[i] < lo) lo = currentSims[i];
      if (currentSims[i] > hi) hi = currentSims[i];
    }
    const range = hi - lo || 1;

    for (let i = 0; i < nPoints; i++) {
      const t = (currentSims[i] - lo) / range;
      const [r, g, b] = coolwarm(t);
      arr[i * 3]     = r;
      arr[i * 3 + 1] = g;
      arr[i * 3 + 2] = b;
    }
  }
  colorAttr.needsUpdate = true;
}

// ── Colormaps ─────────────────────────────────────────────────────────────────
function coolwarm(t) {
  // Blue → white → red
  if (t < 0.5) {
    const s = t * 2;            // 0→1
    return [s, s, 1.0];         // (0,0,1) → (1,1,1)
  } else {
    const s = (t - 0.5) * 2;   // 0→1
    return [1.0, 1 - s, 1 - s]; // (1,1,1) → (1,0,0)
  }
}

// ── Math helpers ──────────────────────────────────────────────────────────────
function dotProductBatch(queryVec, centers, K, D) {
  // centers: Float32Array K*D (L2-normalized)
  // returns Float32Array K
  const sims = new Float32Array(K);
  for (let k = 0; k < K; k++) {
    let dot = 0;
    const base = k * D;
    for (let d = 0; d < D; d++) {
      dot += centers[base + d] * queryVec[d];
    }
    sims[k] = dot;
  }
  return sims;
}


function computePointSize(bbox) {
  const extent = Math.max(
    bbox.max[0] - bbox.min[0],
    bbox.max[1] - bbox.min[1],
  );
  return Math.max(0.5, extent / 800);
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

// ── Wire up form ──────────────────────────────────────────────────────────────
document.getElementById('search-form').addEventListener('submit', handleSearch);

// ── Go! ───────────────────────────────────────────────────────────────────────
init().catch(err => {
  loadingMsg.textContent = 'Error loading data';
  loadingSub.textContent = err.message;
  console.error(err);
});
