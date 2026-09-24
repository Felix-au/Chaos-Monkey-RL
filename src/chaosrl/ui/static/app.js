/**
 * ChaosRL Interactive Web Portal & Real-Time Visualizer
 */

// Application State
const state = {
  currentTab: 'live-lab',
  simulation: null,
  autoPlaying: false,
  autoPlayTimer: null,
  activeFaultTarget: null,
  chartHistory: { steps: [], latencies: [], errors: [] },
  benchmarkData: [],
  selectedDoc: 'REPORT.md',
};

// DOM Elements
const elements = {
  navTabs: document.querySelectorAll('.nav-tab'),
  tabPanes: document.querySelectorAll('.tab-pane'),
  agentSelect: document.getElementById('agent-select'),
  scaleSelect: document.getElementById('scale-select'),
  budgetSlider: document.getElementById('budget-slider'),
  budgetVal: document.getElementById('budget-val'),
  rewardSelect: document.getElementById('reward-select'),
  btnInit: document.getElementById('btn-init'),
  btnStep: document.getElementById('btn-step'),
  btnAutoplay: document.getElementById('btn-autoplay'),
  btnPause: document.getElementById('btn-pause'),
  kpiLatency: document.getElementById('kpi-latency'),
  kpiError: document.getElementById('kpi-error'),
  kpiDiscovered: document.getElementById('kpi-discovered'),
  kpiDamage: document.getElementById('kpi-damage'),
  kpiExpCount: document.getElementById('kpi-exp-count'),
  kpiExpTotal: document.getElementById('kpi-exp-total'),
  kpiSimpleTag: document.getElementById('kpi-simple-tag'),
  kpiCompoundTag: document.getElementById('kpi-compound-tag'),
  badgeSloLat: document.getElementById('badge-slo-lat'),
  badgeSloErr: document.getElementById('badge-slo-err'),
  barLatency: document.getElementById('bar-latency'),
  barError: document.getElementById('bar-error'),
  barDiscovery: document.getElementById('bar-discovery'),
  barBudget: document.getElementById('bar-budget'),
  topologySvg: document.getElementById('topology-svg'),
  weaknessList: document.getElementById('weakness-list'),
  feedContainer: document.getElementById('feed-container'),
  btnClearFeed: document.getElementById('btn-clear-feed'),
  liveChart: document.getElementById('live-chart'),
  benchmarkTbody: document.getElementById('benchmark-tbody'),
  statsCardsGrid: document.getElementById('stats-cards-grid'),
  docLinks: document.querySelectorAll('.doc-link'),
  docContainer: document.getElementById('doc-markdown-container'),
  scaleFilters: document.querySelectorAll('.filter-btn'),
};

// Initialization
document.addEventListener('DOMContentLoaded', () => {
  setupNavigation();
  setupEventListeners();
  initSimulation();
  loadBenchmarks();
  loadDocument('REPORT.md');
});

// Navigation Tabs
function setupNavigation() {
  elements.navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.tab;
      elements.navTabs.forEach(t => t.classList.remove('active'));
      elements.tabPanes.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const pane = document.getElementById(`pane-${target}`);
      if (pane) pane.classList.add('active');
      state.currentTab = target;

      if (target === 'benchmarks') loadBenchmarks();
    });
  });
}

function setupEventListeners() {
  elements.budgetSlider.addEventListener('input', (e) => {
    elements.budgetVal.textContent = e.target.value;
  });

  elements.btnInit.addEventListener('click', () => initSimulation());
  elements.btnStep.addEventListener('click', () => stepSimulation());
  elements.btnAutoplay.addEventListener('click', () => startAutoPlay());
  elements.btnPause.addEventListener('click', () => stopAutoPlay());
  elements.btnClearFeed.addEventListener('click', () => {
    elements.feedContainer.innerHTML = '<div class="feed-item item-init"><span class="feed-tick">[System]</span><span class="feed-msg">Feed cleared.</span></div>';
  });

  // Scale filter in benchmarks
  elements.scaleFilters.forEach(btn => {
    btn.addEventListener('click', () => {
      elements.scaleFilters.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderBenchmarkTable(btn.dataset.scale);
    });
  });

  // Doc switcher
  elements.docLinks.forEach(link => {
    link.addEventListener('click', () => {
      elements.docLinks.forEach(l => l.classList.remove('active'));
      link.classList.add('active');
      loadDocument(link.dataset.doc);
    });
  });
}

// ----------------------------------------------------
// TAB 1: LIVE SIMULATION ENGINE & VISUALIZER
// ----------------------------------------------------

async function initSimulation() {
  stopAutoPlay();
  const reqData = {
    num_services: parseInt(elements.scaleSelect.value),
    seed: 42,
    budget: parseInt(elements.budgetSlider.value),
    agent_type: elements.agentSelect.value,
    reward_variant: elements.rewardSelect.value,
  };

  try {
    const res = await fetch('/api/simulation/init', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(reqData),
    });
    const data = await res.json();
    state.simulation = data;

    // Reset UI Counters
    elements.kpiExpCount.textContent = '0';
    elements.kpiExpTotal.textContent = data.budget;
    elements.kpiDiscovered.textContent = '0';
    elements.kpiDamage.textContent = '0.00';
    elements.kpiLatency.textContent = '118 ms';
    elements.kpiError.textContent = '0.0%';
    elements.barLatency.style.width = '35%';
    elements.barError.style.width = '0%';
    elements.barDiscovery.style.width = '0%';
    elements.barBudget.style.width = '0%';
    elements.badgeSloLat.className = 'kpi-badge badge-healthy';
    elements.badgeSloLat.textContent = 'Healthy';
    elements.badgeSloErr.className = 'kpi-badge badge-healthy';
    elements.badgeSloErr.textContent = 'Healthy';

    // Clear chart
    state.chartHistory = { steps: [0], latencies: [118], errors: [0] };
    drawLiveChart();

    // Render Weakness Scorecard
    renderWeaknessScorecard(data.planted_weaknesses);

    // Render Topology Graph
    renderTopology(data.services, data.nodes);

    addFeedItem('init', `System generated with ${Object.keys(data.services).length} services & ${Object.keys(data.nodes).length} nodes. Agent: ${data.agent}`);
  } catch (err) {
    console.error('Failed to init simulation:', err);
  }
}

async function stepSimulation() {
  if (!state.simulation) return;

  try {
    const res = await fetch('/api/simulation/step', { method: 'POST' });
    if (!res.ok) {
      stopAutoPlay();
      return;
    }
    const stepData = await res.json();
    updateStepUI(stepData);

    if (stepData.terminated) {
      stopAutoPlay();
      addFeedItem('discovery', `[Episode Finished] Budget exhausted (${stepData.step}/${state.simulation.budget}). Total weaknesses found: ${stepData.total_found}/${stepData.total_planted}. Total damage: ${stepData.total_damage.toFixed(2)}.`);
    }
  } catch (err) {
    console.error('Error during step:', err);
    stopAutoPlay();
  }
}

function startAutoPlay() {
  if (state.autoPlaying) return;
  state.autoPlaying = true;
  elements.btnAutoplay.style.display = 'none';
  elements.btnPause.style.display = 'inline-flex';
  state.autoPlayTimer = setInterval(stepSimulation, 400);
}

function stopAutoPlay() {
  state.autoPlaying = false;
  clearInterval(state.autoPlayTimer);
  elements.btnAutoplay.style.display = 'inline-flex';
  elements.btnPause.style.display = 'none';
}

function updateStepUI(step) {
  // Update KPIs
  elements.kpiLatency.textContent = `${step.system_p99.toFixed(1)} ms`;
  elements.kpiError.textContent = `${(step.system_error * 100).toFixed(1)}%`;
  elements.kpiDiscovered.textContent = step.total_found;
  elements.kpiDamage.textContent = step.total_damage.toFixed(2);
  elements.kpiExpCount.textContent = step.step;

  // Progress Bars
  const budgetRatio = (step.step / state.simulation.budget) * 100;
  elements.barBudget.style.width = `${Math.min(100, budgetRatio)}%`;

  const discRatio = (step.total_found / Math.max(1, step.total_planted)) * 100;
  elements.barDiscovery.style.width = `${Math.min(100, discRatio)}%`;

  const latRatio = Math.min(100, (step.system_p99 / 600) * 100);
  elements.barLatency.style.width = `${latRatio}%`;

  const errRatio = Math.min(100, step.system_error * 100);
  elements.barError.style.width = `${errRatio}%`;

  // SLO Badges
  if (step.system_p99 > state.simulation.slo_latency_threshold) {
    elements.badgeSloLat.className = 'kpi-badge badge-breach';
    elements.badgeSloLat.textContent = 'Breach';
  } else {
    elements.badgeSloLat.className = 'kpi-badge badge-healthy';
    elements.badgeSloLat.textContent = 'Healthy';
  }

  if (step.system_error > state.simulation.slo_error_threshold) {
    elements.badgeSloErr.className = 'kpi-badge badge-breach';
    elements.badgeSloErr.textContent = 'Breach';
  } else {
    elements.badgeSloErr.className = 'kpi-badge badge-healthy';
    elements.badgeSloErr.textContent = 'Healthy';
  }

  // Update Scorecard
  if (step.newly_found && step.newly_found.length > 0) {
    step.newly_found.forEach(wid => {
      const item = document.getElementById(`w-item-${wid}`);
      if (item) {
        item.classList.add('discovered');
        const pill = item.querySelector('.w-status-pill');
        if (pill) {
          pill.className = 'w-status-pill pill-unlocked';
          pill.textContent = 'FOUND';
        }
      }
    });
    addFeedItem('discovery', `Weakness Discovered! [${step.newly_found.join(', ')}]`);
  }

  // Update Event Feed
  const faultStr = step.fault_descriptions.join(' + ');
  const violationStr = step.slo_violated ? 'SLO VIOLATED' : 'SLO Safe';
  const type = step.slo_violated ? 'violation' : 'step';
  addFeedItem(type, `Exp #${step.step}: Injected [${faultStr}] &rarr; ${violationStr} (Damage: +${step.damage.toFixed(2)})`);

  // Update Telemetry Chart
  state.chartHistory.steps.push(step.step);
  state.chartHistory.latencies.push(step.system_p99);
  state.chartHistory.errors.push(step.system_error * 100);
  if (state.chartHistory.steps.length > 30) {
    state.chartHistory.steps.shift();
    state.chartHistory.latencies.shift();
    state.chartHistory.errors.shift();
  }
  drawLiveChart();

  // Animate Topology Fault Impact
  highlightActiveFaults(step.faults, step.slo_violated);
}

function addFeedItem(type, text) {
  const item = document.createElement('div');
  item.className = `feed-item item-${type}`;
  item.innerHTML = `<span class="feed-tick">[${new Date().toLocaleTimeString().slice(3)}]</span> <span class="feed-msg">${text}</span>`;
  elements.feedContainer.prepend(item);
}

function renderWeaknessScorecard(weaknesses) {
  elements.weaknessList.innerHTML = '';
  document.getElementById('score-found-badge').textContent = `0 / ${weaknesses.length} Discovered`;

  weaknesses.forEach(w => {
    const el = document.createElement('div');
    el.className = 'weakness-item';
    el.id = `w-item-${w.id}`;
    el.innerHTML = `
      <div class="w-meta">
        <span class="w-id">${w.id}</span>
        <span class="w-desc">${w.description}</span>
      </div>
      <span class="w-status-pill pill-locked">LOCKED</span>
    `;
    elements.weaknessList.appendChild(el);
  });
}

// ----------------------------------------------------
// TOPOLOGY SVG GRAPH RENDERING
// ----------------------------------------------------

function renderTopology(services, nodes) {
  const svg = elements.topologySvg;
  svg.innerHTML = '';

  const width = svg.clientWidth || 700;
  const height = svg.clientHeight || 560;

  // Separate services by layer
  const layers = { 0: [], 1: [], 2: [] };
  Object.values(services).forEach(s => {
    if (layers[s.layer] !== undefined) layers[s.layer].push(s);
  });

  const layerY = {
    0: height * 0.16,
    1: height * 0.50,
    2: height * 0.84,
  };

  const positions = {};

  // Compute node positions
  [0, 1, 2].forEach(l => {
    const list = layers[l];
    const spacing = width / (list.length + 1);
    list.forEach((s, i) => {
      positions[s.id] = {
        x: spacing * (i + 1),
        y: layerY[l],
        service: s,
      };
    });
  });

  // Draw Dependency Links
  const linksGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  linksGroup.setAttribute('id', 'topo-links-group');
  svg.appendChild(linksGroup);

  Object.values(services).forEach(s => {
    const fromPos = positions[s.id];
    if (!fromPos) return;

    s.dependencies.forEach(depId => {
      const toPos = positions[depId];
      if (!toPos) return;

      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      const d = `M ${fromPos.x} ${fromPos.y} C ${fromPos.x} ${(fromPos.y + toPos.y) / 2}, ${toPos.x} ${(fromPos.y + toPos.y) / 2}, ${toPos.x} ${toPos.y}`;
      path.setAttribute('d', d);
      path.setAttribute('class', 'topo-link');
      path.setAttribute('id', `link-${s.id}-${depId}`);
      path.setAttribute('fill', 'none');
      linksGroup.appendChild(path);
    });
  });

  // Draw Service Nodes
  const nodesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  nodesGroup.setAttribute('id', 'topo-nodes-group');
  svg.appendChild(nodesGroup);

  const rectW = 68;
  const rectH = 44;

  Object.values(positions).forEach(pos => {
    const s = pos.service;
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', 'svc-node-group');
    g.setAttribute('id', `node-group-${s.id}`);
    g.setAttribute('transform', `translate(${pos.x - rectW / 2}, ${pos.y - rectH / 2})`);

    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('width', rectW);
    rect.setAttribute('height', rectH);
    rect.setAttribute('class', 'svc-node-rect');
    rect.setAttribute('id', `node-rect-${s.id}`);

    // Layer-specific styling
    if (s.layer === 0) rect.style.stroke = 'rgba(255, 255, 255, 0.7)';
    else if (s.layer === 2) rect.style.stroke = 'rgba(113, 113, 122, 0.7)';

    // Label text
    const textTitle = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    textTitle.setAttribute('x', rectW / 2);
    textTitle.setAttribute('y', 18);
    textTitle.setAttribute('text-anchor', 'middle');
    textTitle.setAttribute('class', 'svc-title-text');
    textTitle.textContent = s.id;

    // Subtext (replicas & timeout)
    const textSub = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    textSub.setAttribute('x', rectW / 2);
    textSub.setAttribute('y', 34);
    textSub.setAttribute('text-anchor', 'middle');
    textSub.setAttribute('class', 'svc-sub-text');
    textSub.textContent = `${s.replicas.length} reps | ${s.timeout}ms`;

    g.appendChild(rect);
    g.appendChild(textTitle);
    g.appendChild(textSub);

    // Hover tooltip / info
    g.addEventListener('click', () => {
      addFeedItem('info', `[Inspect ${s.id}] Layer: ${s.layer}, Latency: ${s.base_latency.toFixed(1)}ms, Retries: ${s.max_retries}, CB: ${s.circuit_breaker ? 'Yes' : 'No'}`);
    });

    nodesGroup.appendChild(g);
  });
}

function highlightActiveFaults(faults, sloViolated) {
  // Clear previous highlights
  document.querySelectorAll('.svc-node-rect').forEach(r => r.classList.remove('under-attack'));
  document.querySelectorAll('.topo-link').forEach(l => l.setAttribute('class', 'topo-link'));

  if (!faults) return;

  faults.forEach(f => {
    const rect = document.getElementById(`node-rect-${f.target}`);
    if (rect) {
      rect.classList.add('under-attack');
    }

    // If node failure, highlight services on that node
    if (f.fault_type === 'NODE_FAILURE' && state.simulation) {
      Object.values(state.simulation.services).forEach(s => {
        if (s.replicas.some(r => r.node === f.target)) {
          const rEl = document.getElementById(`node-rect-${s.id}`);
          if (rEl) rEl.classList.add('under-attack');
        }
      });
    }

    // Highlight downstream links with cascading error style if SLO breached
    if (sloViolated && state.simulation && state.simulation.services[f.target]) {
      const callers = state.simulation.services[f.target].callers || [];
      callers.forEach(cid => {
        const link = document.getElementById(`link-${cid}-${f.target}`);
        if (link) link.setAttribute('class', 'topo-link flow-error');
      });
    }
  });
}

// ----------------------------------------------------
// LIVE TELEMETRY CANVAS CHART
// ----------------------------------------------------

function drawLiveChart() {
  const canvas = elements.liveChart;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  canvas.width = canvas.parentElement.clientWidth;
  canvas.height = canvas.parentElement.clientHeight;

  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const { steps, latencies, errors } = state.chartHistory;
  if (steps.length < 2) return;

  // Grid lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
  ctx.lineWidth = 1;
  for (let y = 0.25; y < 1; y += 0.25) {
    ctx.beginPath();
    ctx.moveTo(0, h * y);
    ctx.lineTo(w, h * y);
    ctx.stroke();
  }

  // Draw Latency Line (Cyan)
  const maxLat = Math.max(500, ...latencies);
  ctx.strokeStyle = '#06b6d4';
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  steps.forEach((step, i) => {
    const x = (i / (steps.length - 1)) * w;
    const y = h - (latencies[i] / maxLat) * (h * 0.85);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Draw Error Rate Line (Rose)
  const maxErr = 100;
  ctx.strokeStyle = '#f43f5e';
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  steps.forEach((step, i) => {
    const x = (i / (steps.length - 1)) * w;
    const y = h - (errors[i] / maxErr) * (h * 0.85);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

// ----------------------------------------------------
// TAB 2: BENCHMARK RESULTS LOADER
// ----------------------------------------------------

async function loadBenchmarks() {
  try {
    const res = await fetch('/api/benchmarks/summary');
    const data = await res.json();
    state.benchmarkData = data.grouped || [];
    renderBenchmarkTable('all');

    // Load Stats
    const statsRes = await fetch('/api/benchmarks/stats');
    const stats = await statsRes.json();
    renderStatsCards(stats);
  } catch (err) {
    console.error('Failed to load benchmarks:', err);
  }
}

function renderBenchmarkTable(scaleFilter) {
  elements.benchmarkTbody.innerHTML = '';
  const filtered = scaleFilter === 'all'
    ? state.benchmarkData
    : state.benchmarkData.filter(d => String(d.scale) === scaleFilter);

  filtered.forEach(row => {
    const tr = document.createElement('tr');
    const isRL = row.method.includes('PPO');
    tr.style.fontWeight = isRL ? '600' : 'normal';
    if (isRL) tr.style.background = 'rgba(255, 255, 255, 0.05)';

    tr.innerHTML = `
      <td>${row.scale}</td>
      <td style="color: ${isRL ? '#10b981' : '#fff'};">${row.method}</td>
      <td>${row.weaknesses_found_mean.toFixed(2)} ± ${row.weaknesses_found_std.toFixed(2)}</td>
      <td>${(row.discovery_rate_mean * 100).toFixed(1)}%</td>
      <td>${row.simple_weaknesses_mean.toFixed(2)}</td>
      <td>${row.compound_weaknesses_mean.toFixed(2)}</td>
      <td style="color: ${row.total_damage_mean > 3000 ? '#fca5a5' : '#86efac'};">${row.total_damage_mean.toFixed(2)}</td>
      <td>${row.exp_to_first_mean.toFixed(1)}</td>
      <td>${row.exp_to_50_mean.toFixed(1)}</td>
      <td>${row.mean_decision_ms_mean.toFixed(2)} ms</td>
    `;
    elements.benchmarkTbody.appendChild(tr);
  });
}

function renderStatsCards(stats) {
  elements.statsCardsGrid.innerHTML = '';
  Object.entries(stats).forEach(([pair, data]) => {
    const card = document.createElement('div');
    card.className = 'stat-card';
    const disc = data.discovery_comparison;
    const isSig = disc.significant_at_05;

    card.innerHTML = `
      <div class="stat-card-title">
        <span>${pair.replace('ChaosRL_vs_', 'ChaosRL vs. ')}</span>
        <span class="${isSig ? 'badge-sig' : 'badge-nosig'}">${isSig ? 'Significant (p<0.05)' : 'Not Significant'}</span>
      </div>
      <div class="stat-metrics-row">
        <span>Mann-Whitney U: <strong>${disc.u_stat.toFixed(1)}</strong></span>
        <span>p-value: <strong>${disc.p_value.toExponential(3)}</strong></span>
      </div>
      <div class="stat-metrics-row">
        <span>Cohen's d Effect Size: <strong>${disc.cohens_d.toFixed(2)}</strong></span>
        <span>Rank Biserial: <strong>${disc.rank_biserial.toFixed(2)}</strong></span>
      </div>
    `;
    elements.statsCardsGrid.appendChild(card);
  });
}

// ----------------------------------------------------
// TAB 3: PROJECT DOCUMENTATION VIEWER
// ----------------------------------------------------

async function loadDocument(docName) {
  elements.docContainer.innerHTML = '<div class="doc-loader">Loading document...</div>';
  try {
    const res = await fetch(`/api/docs/${docName}`);
    if (!res.ok) throw new Error('Document not found');
    const data = await res.json();
    elements.docContainer.innerHTML = marked.parse(data.content);
  } catch (err) {
    elements.docContainer.innerHTML = `<div class="error-msg">Failed to load ${docName}: ${err.message}</div>`;
  }
}
