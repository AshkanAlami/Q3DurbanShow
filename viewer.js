/**
 * viewer.js — Three.js point cloud viewer with CLIP text queries
 *
 * Multi-scene layout expected in ./data/:
 *   scenes.json          Array of folder names, e.g. ["scene1","scene2"]
 *   <name>/
 *     scene.bin            Single binary (see export_web_data.py for layout)
 *     cluster_centers.bin  Float32, K×D, L2-normalized
 *     metadata.json        n_points, n_clusters, feature_dim, bbox_min, bbox_max
 *
 * Single-scene fallback: if scenes.json is absent, loads directly from ./data/
 *
 * Models:
 *   data/models/         CLIP ONNX files (from download_clip_model.py)
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { AutoTokenizer, CLIPTextModelWithProjection, env } from
  'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

env.localModelPath    = './data/models/';
env.allowLocalModels  = false;
env.allowRemoteModels = false;
env.backends.onnx.wasm.numThreads = 1;

// ── Global Three.js objects ───────────────────────────────────────────────────
let renderer, threeScene, camera, controls;
let pointCloud   = null;
let geometry     = null;
let pointMaterial = null;

// ── Active-scene state (updated on every scene switch) ────────────────────────
let nPoints = 0, nClusters = 0, featureDim = 0;
let rgbColorsF32   = null;   // Float32Array N*3, [0,1]
let clusterIds     = null;   // Int32Array N  (-1 = unlabeled)
let clusterCenters = null;   // Float32Array K*D, L2-normalized
let alphaArr       = null;   // Float32Array N, per-point alpha
let currentSims    = null;   // Float32Array N — null until a query runs
let currentView    = 'rgb';
let alphaThreshold = 0.5;

// ── Multi-scene state ─────────────────────────────────────────────────────────
let sceneList      = [];     // [{id, label}]
let sceneCache     = {};     // id → parsed scene data
let currentSceneId = null;

// ── CLIP ──────────────────────────────────────────────────────────────────────
let clipTokenizer = null;
let clipModel     = null;
let modelReady    = false;

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
const sceneSec       = document.getElementById('scene-section');
const sceneListEl    = document.getElementById('scene-list');

// ── Entry point ───────────────────────────────────────────────────────────────
async function init() {
  setupThreeJS();
  animate();

  // Discover scenes + start CLIP load in parallel
  const [scenes] = await Promise.all([
    discoverScenes(),
    loadCLIP(),
  ]);

  // Build sidebar scene list (hidden if only 1 scene)
  if (scenes.length > 1) {
    renderSceneList(scenes);
    sceneSec.style.display = '';
  }

  // Load + activate first scene
  setStatus(dataDot, dataText, 'yellow', `Loading "${scenes[0].label}"…`);
  await loadSceneData(scenes[0].id);
  activateScene(scenes[0].id);

  overlay.classList.add('hidden');
  enableSearch();

  // Background-preload remaining scenes (fire-and-forget)
  preloadRemaining();
}

// ── Scene discovery ───────────────────────────────────────────────────────────
async function discoverScenes() {
  try {
    const list = await fetchJSON('./data/scenes.json');
    // Accept either ["name1","name2"] or [{id,label},...]
    sceneList = list.map(entry =>
      typeof entry === 'string'
        ? { id: entry, label: entry }
        : { id: entry.id, label: entry.label ?? entry.id }
    );
  } catch {
    // Single-scene fallback — data lives directly in ./data/
    sceneList = [{ id: '__single__', label: 'Scene' }];
  }
  return sceneList;
}

function sceneBase(id) {
  return id === '__single__' ? './data' : `./data/${id}`;
}

// ── Load + parse one scene (cached) ──────────────────────────────────────────
async function loadSceneData(id) {
  if (sceneCache[id]) return sceneCache[id];

  const base = sceneBase(id);
  const meta = await fetchJSON(`${base}/metadata.json`);

  const [sceneBuf, centersBuf] = await Promise.all([
    fetchBin(`${base}/scene.bin`),
    fetchBin(`${base}/cluster_centers.bin`),
  ]);

  const parsed = parseBin(sceneBuf, centersBuf, meta);
  sceneCache[id] = parsed;
  return parsed;
}

function parseBin(sceneBuf, centersBuf, meta) {
  const N  = new Uint32Array(sceneBuf, 0, 1)[0];
  const bboxMin = meta.bbox_min;
  const bboxMax = meta.bbox_max;

  const xyzU16 = new Uint16Array(sceneBuf, 4, N * 3);

  const rgbByteOffset = 4 + N * 6;
  const rgbU8 = new Uint8Array(sceneBuf, rgbByteOffset, N * 3);

  const idsRaw = rgbByteOffset + N * 3;
  const idsAligned = (idsRaw % 2 === 0) ? idsRaw : idsRaw + 1;
  const idsU16 = new Uint16Array(sceneBuf, idsAligned, N);

  // XYZ: uint16 → float32
  const positions = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    for (let d = 0; d < 3; d++) {
      positions[i*3+d] = (xyzU16[i*3+d] / 65535) * (bboxMax[d] - bboxMin[d]) + bboxMin[d];
    }
  }

  // RGB: uint8 → float32 [0,1]
  const rgb = new Float32Array(N * 3);
  for (let i = 0; i < N * 3; i++) rgb[i] = rgbU8[i] / 255;

  // Cluster IDs: 65535 → -1
  const ids = new Int32Array(N);
  for (let i = 0; i < N; i++) ids[i] = idsU16[i] === 65535 ? -1 : idsU16[i];

  return {
    positions,
    rgbColorsF32:   rgb,
    clusterIds:     ids,
    clusterCenters: new Float32Array(centersBuf),
    metadata:       meta,
    nPoints:        N,
    nClusters:      meta.n_clusters,
    featureDim:     meta.feature_dim,
    bboxMin,
    bboxMax,
  };
}

// ── Activate a cached scene ───────────────────────────────────────────────────
function activateScene(id) {
  const data = sceneCache[id];
  if (!data) return;

  // Tear down old point cloud
  if (pointCloud)  { threeScene.remove(pointCloud); }
  if (geometry)    { geometry.dispose(); }
  if (pointMaterial) { pointMaterial.dispose(); }

  // Update active-scene module state
  ({ nPoints, nClusters, featureDim,
     rgbColorsF32, clusterIds, clusterCenters } = data);
  const { bboxMin, bboxMax } = data;

  // Build geometry
  geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(data.positions, 3));
  geometry.setAttribute('color',    new THREE.BufferAttribute(rgbColorsF32.slice(), 3));

  alphaArr = new Float32Array(nPoints).fill(1.0);
  geometry.setAttribute('aAlpha',   new THREE.BufferAttribute(alphaArr, 1));

  // Custom shader: per-vertex alpha + circular points
  const extent = Math.max(bboxMax[0] - bboxMin[0], bboxMax[1] - bboxMin[1]);
  const VERT = `
    attribute vec3 color;
    attribute float aAlpha;
    varying vec3 vColor;
    varying float vAlpha;
    uniform float pointSize;
    uniform float renderScale;
    void main() {
      vColor = color;
      vAlpha = aAlpha;
      vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = pointSize * (renderScale / -mvPos.z);
      gl_Position  = projectionMatrix * mvPos;
    }
  `;
  const FRAG = `
    varying vec3 vColor;
    varying float vAlpha;
    void main() {
      if (vAlpha < 0.01) discard;
      vec2 uv = gl_PointCoord - 0.5;
      if (dot(uv, uv) > 0.25) discard;
      gl_FragColor = vec4(vColor, vAlpha);
    }
  `;
  pointMaterial = new THREE.ShaderMaterial({
    vertexShader:   VERT,
    fragmentShader: FRAG,
    transparent:    true,
    uniforms: {
      pointSize:   { value: Math.max(0.3, extent / 800) },
      renderScale: { value: renderer.getSize(new THREE.Vector2()).y / 2 },
    },
  });

  pointCloud = new THREE.Points(geometry, pointMaterial);
  threeScene.add(pointCloud);

  // Camera: bird's-eye above scene center
  const cx = (bboxMin[0] + bboxMax[0]) / 2;
  const cy = (bboxMin[1] + bboxMax[1]) / 2;
  camera.position.set(cx, cy - extent * 0.3, bboxMax[2] + extent * 0.6);
  camera.lookAt(cx, cy, 0);
  controls.target.set(cx, cy, 0);
  controls.update();

  // Reset query state
  currentSims  = null;
  currentView  = 'rgb';
  document.getElementById('btn-rgb').classList.add('active');
  document.getElementById('btn-heatmap').classList.remove('active');
  resultsSection.style.display = 'none';
  document.getElementById('alpha-slider').value = 0.5;
  alphaThreshold = 0.5;
  document.getElementById('alpha-val').textContent = '50%';

  // Highlight active scene button
  currentSceneId = id;
  document.querySelectorAll('.scene-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.id === id);
    if (b.dataset.id === id) b.querySelector('.scene-dot')?.remove();
  });

  setStatus(dataDot, dataText, 'green', `${(nPoints/1e6).toFixed(1)}M pts — ${data.metadata.n_clusters} clusters`);
}

// ── Switch scene (user-triggered) ─────────────────────────────────────────────
window.switchScene = async function(id) {
  if (id === currentSceneId) return;

  // Disable search while switching
  queryInput.disabled = true;
  searchBtn.disabled  = true;

  if (!sceneCache[id]) {
    setStatus(dataDot, dataText, 'yellow', `Loading "${id}"…`);
    await loadSceneData(id);
  }
  activateScene(id);

  queryInput.disabled = false;
  searchBtn.disabled  = false;
};

// ── Background preload ────────────────────────────────────────────────────────
async function preloadRemaining() {
  for (const s of sceneList) {
    if (sceneCache[s.id]) continue;
    // Mark as loading in sidebar
    const dot = document.querySelector(`.scene-btn[data-id="${s.id}"] .scene-dot`);
    if (dot) dot.className = 'scene-dot dot yellow';

    await loadSceneData(s.id);

    if (dot) dot.className = 'scene-dot dot green';
  }
}

// ── Render scene list in sidebar ──────────────────────────────────────────────
function renderSceneList(scenes) {
  sceneListEl.innerHTML = '';
  for (const s of scenes) {
    const btn = document.createElement('button');
    btn.className = 'scene-btn' + (s.id === sceneList[0].id ? ' active' : '');
    btn.dataset.id = s.id;
    btn.onclick = () => switchScene(s.id);

    const dot = document.createElement('span');
    dot.className = 'scene-dot dot ' + (s.id === sceneList[0].id ? 'green' : 'yellow');
    btn.appendChild(dot);

    const label = document.createElement('span');
    label.textContent = s.label;
    btn.appendChild(label);

    sceneListEl.appendChild(btn);
  }
}

// ── Three.js setup ─────────────────────────────────────────────────────────────
function setupThreeJS() {
  const wrap = document.getElementById('canvas-wrap');

  renderer = new THREE.WebGLRenderer({ antialias: false });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(wrap.clientWidth, wrap.clientHeight);
  renderer.setClearColor(0x0d0d0d);
  wrap.appendChild(renderer.domElement);

  threeScene = new THREE.Scene();
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
    if (pointMaterial) {
      pointMaterial.uniforms.renderScale.value = renderer.getSize(new THREE.Vector2()).y / 2;
    }
  });
}

// ── CLIP ──────────────────────────────────────────────────────────────────────
async function loadCLIP() {
  const modelId = 'Xenova/clip-vit-large-patch14';
  setStatus(modelDot, modelText, 'yellow', 'Loading CLIP tokenizer…');
  try {
    clipTokenizer = await AutoTokenizer.from_pretrained(modelId);
  } catch {
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

// ── Search ────────────────────────────────────────────────────────────────────
async function handleSearch(e) {
  e.preventDefault();
  const query = queryInput.value.trim();
  if (!query || !modelReady || !currentSceneId) return;

  queryInput.disabled   = true;
  searchBtn.disabled    = true;
  searchBtn.textContent = '…';

  try {
    const inputs = clipTokenizer([query], { padding: true, truncation: true });
    const { text_embeds } = await clipModel(inputs);
    const rawEmbed = text_embeds.data;

    let norm = 0;
    for (let d = 0; d < featureDim; d++) norm += rawEmbed[d] * rawEmbed[d];
    norm = Math.sqrt(norm);
    const queryVec = new Float32Array(featureDim);
    for (let d = 0; d < featureDim; d++) queryVec[d] = rawEmbed[d] / (norm + 1e-8);

    const clusterSims = dotProductBatch(queryVec, clusterCenters, nClusters, featureDim);
    const simMin = clusterSims.reduce((a, b) => Math.min(a, b), Infinity);
    const simMax = clusterSims.reduce((a, b) => Math.max(a, b), -Infinity);

    currentSims = new Float32Array(nPoints);
    for (let i = 0; i < nPoints; i++) {
      const cid = clusterIds[i];
      currentSims[i] = cid >= 0 ? clusterSims[cid] : simMin;
    }

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
  const colorArr  = geometry.attributes.color.array;
  const alphaAttr = geometry.attributes.aAlpha;

  if (currentView === 'rgb' || !currentSims) {
    for (let i = 0; i < nPoints * 3; i++) colorArr[i] = rgbColorsF32[i];
    alphaAttr.array.fill(1.0);
    alphaAttr.needsUpdate = true;
  } else {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < nPoints; i++) {
      if (currentSims[i] < lo) lo = currentSims[i];
      if (currentSims[i] > hi) hi = currentSims[i];
    }
    const range = hi - lo || 1;

    for (let i = 0; i < nPoints; i++) {
      const t = (currentSims[i] - lo) / range;

      // Blend ramp: 0 below threshold, rises to 1 at t=1
      const span  = Math.max(0.001, 1 - alphaThreshold);
      const blend = Math.pow(Math.max(0, (t - alphaThreshold) / span), 0.7);

      const [hr, hg, hb] = coolwarm(t);
      colorArr[i*3]   = rgbColorsF32[i*3]   * (1 - blend) + hr * blend;
      colorArr[i*3+1] = rgbColorsF32[i*3+1] * (1 - blend) + hg * blend;
      colorArr[i*3+2] = rgbColorsF32[i*3+2] * (1 - blend) + hb * blend;

      // All points always visible
      alphaAttr.array[i] = 1.0;
    }
    alphaAttr.needsUpdate = true;
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
  renderer.render(threeScene, camera);
}

// ── Event listeners ───────────────────────────────────────────────────────────
document.getElementById('search-form').addEventListener('submit', handleSearch);

document.getElementById('alpha-slider').addEventListener('input', function() {
  alphaThreshold = parseFloat(this.value);
  document.getElementById('alpha-val').textContent = Math.round(alphaThreshold * 100) + '%';
  if (currentView === 'heatmap') applyColors();
});

init().catch(err => {
  loadingMsg.textContent = 'Error loading data';
  loadingSub.textContent = err.message;
  console.error(err);
});
