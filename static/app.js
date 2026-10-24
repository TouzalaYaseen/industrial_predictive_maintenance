/**
 * app.js - Industrial Predictive Maintenance Cockpit Controller
 * =============================================================
 * Handles real-time telemetry streaming, Chart.js rolling strip-charts,
 * stream transport controls, calibrated failure alerting, and TreeSHAP diagnostics.
 */

// Global Dashboard State
const state = {
  currentIndex: 0,
  isPlaying: false,
  playbackSpeed: 1.0,
  timerId: null,
  maxTicks: 10000,
  windowSize: 30, // Visible window size on rolling strip-charts
  mechanicalChart: null,
  thermalChart: null,
  latestTick: null,
};

// Calibrated Operational Threshold ($t^* = 0.020$)
const THRESHOLD = 0.020;

// Document Ready Initialization
document.addEventListener('DOMContentLoaded', async () => {
  // Initialize Lucide icons
  if (window.lucide) {
    lucide.createIcons();
  }

  // Verify API health
  await checkApiHealth();

  // Initialize rolling Chart.js instances
  initCharts();

  // Attach all DOM event listeners
  setupTransportControls();
  setupTabControls();
  setupSandboxControls();
  setupShapInspector();

  // Load initial telemetry window starting at tick 0
  await loadInitialWindow(0);
});

/**
 * Checks FastAPI backend health and latency.
 */
async function checkApiHealth() {
  const statusBadge = document.getElementById('api-status-badge');
  const statusText = document.getElementById('api-status-text');
  const startTime = performance.now();

  try {
    const res = await fetch('/health');
    const data = await res.json();
    const latency = Math.round(performance.now() - startTime);

    if (data.status === 'healthy' && data.model_loaded) {
      statusBadge.className = 'flex items-center space-x-2 px-3 py-1.5 rounded-md bg-zinc-900 border border-zinc-800 font-mono text-zinc-300';
      statusText.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-500 inline-block mr-1"></span>FastAPI: Online (${latency}ms)`;
    } else {
      statusText.innerHTML = `<span class="w-2 h-2 rounded-full bg-amber-500 inline-block mr-1"></span>FastAPI: Degraded`;
    }
  } catch (err) {
    statusText.innerHTML = `<span class="w-2 h-2 rounded-full bg-rose-500 inline-block mr-1"></span>FastAPI: Offline`;
  }
}

/**
 * Initializes the two rolling Chart.js strip-charts with shadcn Zinc dark theme styling.
 */
function initCharts() {
  const chartFont = {
    family: "'Inter', sans-serif",
    size: 10,
  };

  const commonScales = {
    x: {
      grid: { color: 'rgba(39, 39, 42, 0.6)', drawBorder: false },
      ticks: { color: '#71717a', font: chartFont, maxTicksLimit: 8 },
    },
    y: {
      grid: { color: 'rgba(39, 39, 42, 0.6)', drawBorder: false },
      ticks: { color: '#71717a', font: chartFont },
    }
  };

  // 1. Mechanical Strip-Chart (Torque [Nm] & Scaled Speed [RPM / 50])
  const ctxMech = document.getElementById('mechanicalChart').getContext('2d');
  state.mechanicalChart = new Chart(ctxMech, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Torque [Nm]',
          data: [],
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.35,
          fill: false,
        },
        {
          label: 'Speed / 50 [rpm]',
          data: [],
          borderColor: '#c084fc',
          backgroundColor: 'rgba(192, 132, 252, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.35,
          fill: false,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#18181b',
          borderColor: '#27272a',
          borderWidth: 1,
          titleFont: { size: 11, weight: '600' },
          bodyFont: { size: 11 },
          padding: 8,
          boxPadding: 4,
        }
      },
      scales: {
        x: commonScales.x,
        y: {
          ...commonScales.y,
          suggestedMin: 0,
          suggestedMax: 80,
        }
      }
    }
  });

  // 2. Thermal Dynamics Strip-Chart (Air Temp vs Process Temp [K])
  const ctxTherm = document.getElementById('thermalChart').getContext('2d');
  state.thermalChart = new Chart(ctxTherm, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Process Temp [K]',
          data: [],
          borderColor: '#f59e0b',
          backgroundColor: 'rgba(245, 158, 11, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.35,
          fill: false,
        },
        {
          label: 'Air Temp [K]',
          data: [],
          borderColor: '#06b6d4',
          backgroundColor: 'rgba(6, 182, 212, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.35,
          fill: false,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#18181b',
          borderColor: '#27272a',
          borderWidth: 1,
          titleFont: { size: 11, weight: '600' },
          bodyFont: { size: 11 },
          padding: 8,
          boxPadding: 4,
        }
      },
      scales: {
        x: commonScales.x,
        y: {
          ...commonScales.y,
          suggestedMin: 295,
          suggestedMax: 315,
        }
      }
    }
  });
}

/**
 * Loads a continuous window of historical telemetry to prefill the rolling charts.
 */
async function loadInitialWindow(startIndex) {
  try {
    const res = await fetch(`/stream/window?start=${startIndex}&limit=${state.windowSize}`);
    if (!res.ok) throw new Error('Failed to load telemetry window');
    const data = await res.json();

    // Reset chart data buffers
    state.mechanicalChart.data.labels = [];
    state.mechanicalChart.data.datasets[0].data = [];
    state.mechanicalChart.data.datasets[1].data = [];

    state.thermalChart.data.labels = [];
    state.thermalChart.data.datasets[0].data = [];
    state.thermalChart.data.datasets[1].data = [];

    // Populate charts with batch window
    data.ticks.forEach(tick => {
      appendTickToCharts(tick);
    });

    state.mechanicalChart.update('none');
    state.thermalChart.update('none');

    // Update state to the last tick in the window
    const lastTick = data.ticks[data.ticks.length - 1];
    state.currentIndex = lastTick.index;
    state.latestTick = lastTick;
    updateDashboardUI(lastTick);
  } catch (err) {
    console.error('Error prefetching window:', err);
  }
}

/**
 * Fetches and processes a single tick at the requested index.
 */
async function loadTick(index) {
  try {
    const res = await fetch(`/stream/tick?index=${index}`);
    if (!res.ok) throw new Error(`Tick ${index} fetch failed`);
    const tick = await res.json();

    state.currentIndex = tick.index;
    state.latestTick = tick;

    // Append to charts with sliding window cap
    appendTickToCharts(tick);

    // Limit buffer length to windowSize
    if (state.mechanicalChart.data.labels.length > state.windowSize) {
      state.mechanicalChart.data.labels.shift();
      state.mechanicalChart.data.datasets[0].data.shift();
      state.mechanicalChart.data.datasets[1].data.shift();

      state.thermalChart.data.labels.shift();
      state.thermalChart.data.datasets[0].data.shift();
      state.thermalChart.data.datasets[1].data.shift();
    }

    state.mechanicalChart.update('none');
    state.thermalChart.update('none');

    // Update KPI cards and stream badges
    updateDashboardUI(tick);
  } catch (err) {
    console.error(`Error loading tick ${index}:`, err);
  }
}

/**
 * Helper to push a tick's sensor values into both Chart.js datasets.
 */
function appendTickToCharts(tick) {
  const label = `#${tick.index}`;
  const sensors = tick.telemetry;

  // Mechanical chart
  state.mechanicalChart.data.labels.push(label);
  state.mechanicalChart.data.datasets[0].data.push(sensors.torque_nm);
  state.mechanicalChart.data.datasets[1].data.push(Number((sensors.rotational_speed_rpm / 50).toFixed(1)));

  // Thermal chart
  state.thermalChart.data.labels.push(label);
  state.thermalChart.data.datasets[0].data.push(sensors.process_temperature_k);
  state.thermalChart.data.datasets[1].data.push(sensors.air_temperature_k);
}

/**
 * Updates KPI cards, status badges, risk gauge, and scrubber controls.
 */
function updateDashboardUI(tick) {
  const pred = tick.prediction;
  const sensors = tick.telemetry;
  const risk = pred.failure_probability;

  // 1. Stream Index & Timeline Slider
  const idx = tick.index;
  document.getElementById('stream-index-badge').textContent = `Tick: ${idx.toLocaleString()} / 10,000`;
  const slider = document.getElementById('timeline-slider');
  slider.value = idx;
  const pct = ((idx / state.maxTicks) * 100).toFixed(1);
  document.getElementById('slider-pct').textContent = `${pct}%`;

  // 2. KPI: Mechanical Load & Power
  const omega = (sensors.rotational_speed_rpm * 2 * Math.PI) / 60;
  const powerKw = ((sensors.torque_nm * omega) / 1000).toFixed(2);
  document.getElementById('kpi-power-text').textContent = `${powerKw} kW`;
  document.getElementById('kpi-torque-text').textContent = `${sensors.torque_nm.toFixed(1)} Nm`;
  document.getElementById('kpi-speed-text').textContent = `${sensors.rotational_speed_rpm} RPM`;
  document.getElementById('kpi-wear-text').textContent = `Wear: ${sensors.tool_wear_min} min`;

  // 3. KPI: Calibrated Risk & Progress Bar
  const riskPctStr = (risk * 100).toFixed(2);
  const riskTextEl = document.getElementById('kpi-risk-text');
  const riskBarEl = document.getElementById('kpi-risk-bar');
  riskTextEl.textContent = `${riskPctStr}%`;

  // Visual gauge width: scale relative to 5% max visual range for high sensitivity around t*=2.0%
  const visualBarWidth = Math.min(100, (risk / 0.05) * 100);
  riskBarEl.style.width = `${Math.max(4, visualBarWidth)}%`;

  // 4. KPI: Machine Health State & Alert Box
  const stateCard = document.getElementById('kpi-state-card');
  const stateText = document.getElementById('kpi-state-text');
  const stateDesc = document.getElementById('kpi-state-desc');
  const stateBar = document.getElementById('kpi-state-bar');
  const stateIcon = document.getElementById('kpi-state-icon');
  const tierBadge = document.getElementById('kpi-state-tier');

  const alertBox = document.getElementById('alert-callout-box');
  const alertTitle = document.getElementById('alert-callout-title');
  const alertDesc = document.getElementById('alert-callout-desc');
  const alertIcon = document.getElementById('alert-callout-icon');

  const tierName = sensors.machine_type === 'L' ? 'Low (L)' : sensors.machine_type === 'M' ? 'Medium (M)' : 'High (H)';
  tierBadge.textContent = `Tier: ${tierName}`;

  if (pred.risk_tier === 'CRITICAL' || risk >= THRESHOLD) {
    // 🔴 Critical Shutdown State
    stateText.textContent = 'CRITICAL SHUTDOWN';
    stateText.className = 'text-xl font-bold tracking-tight text-rose-400';
    stateDesc.textContent = 'Operational risk exceeds economic cutoff (t* = 2.0%). Preventative shutdown initiated.';
    stateBar.className = 'absolute bottom-0 left-0 right-0 h-0.5 bg-rose-500';
    stateCard.className = 'bg-zinc-900 border border-rose-500/50 glow-rose rounded-xl p-4.5 relative overflow-hidden transition-all duration-300';
    riskBarEl.className = 'bg-rose-500 h-1.5 rounded-full transition-all duration-300';
    
    alertBox.className = 'p-3.5 rounded-lg border border-rose-800/60 bg-rose-950/40 text-rose-400 space-y-1';
    alertTitle.textContent = 'EMERGENCY SHUTDOWN RECOMMENDED';
    alertDesc.textContent = `AI Risk (${riskPctStr}%) exceeds calibrated threshold (2.0%). Immediate inspection prevents $5,000 catastrophic breakdown.`;

    if (window.lucide) {
      stateIcon.setAttribute('data-lucide', 'alert-triangle');
      stateIcon.className = 'w-4 h-4 text-rose-400';
      alertIcon.setAttribute('data-lucide', 'alert-octagon');
      lucide.createIcons();
    }
  } else if (pred.risk_tier === 'WARNING' || risk >= 0.010) {
    // 🟡 Warning State (Approaching Threshold)
    stateText.textContent = 'PRE-FAILURE WARNING';
    stateText.className = 'text-xl font-bold tracking-tight text-amber-400';
    stateDesc.textContent = 'Thermal or mechanical stress is elevating near threshold limits. Monitor closely.';
    stateBar.className = 'absolute bottom-0 left-0 right-0 h-0.5 bg-amber-500';
    stateCard.className = 'bg-zinc-900 border border-amber-500/40 glow-amber rounded-xl p-4.5 relative overflow-hidden transition-all duration-300';
    riskBarEl.className = 'bg-amber-500 h-1.5 rounded-full transition-all duration-300';

    alertBox.className = 'p-3.5 rounded-lg border border-amber-800/40 bg-amber-950/20 text-amber-400 space-y-1';
    alertTitle.textContent = 'ELEVATED STRESS ENVELOPE';
    alertDesc.textContent = `Telemetry shows elevated strain (Risk: ${riskPctStr}%). Nearing cutoff threshold.`;

    if (window.lucide) {
      stateIcon.setAttribute('data-lucide', 'alert-circle');
      stateIcon.className = 'w-4 h-4 text-amber-400';
      alertIcon.setAttribute('data-lucide', 'alert-triangle');
      lucide.createIcons();
    }
  } else {
    // 🟢 Optimal Normal State
    stateText.textContent = 'OPTIMAL';
    stateText.className = 'text-xl font-bold tracking-tight text-emerald-400';
    stateDesc.textContent = 'All sensor envelopes within safe mechanical limits.';
    stateBar.className = 'absolute bottom-0 left-0 right-0 h-0.5 bg-emerald-500';
    stateCard.className = 'bg-zinc-900 border border-zinc-800 rounded-xl p-4.5 relative overflow-hidden transition-all duration-300';
    riskBarEl.className = 'bg-emerald-500 h-1.5 rounded-full transition-all duration-300';

    alertBox.className = 'p-3.5 rounded-lg border border-emerald-800/40 bg-emerald-950/20 text-emerald-400 space-y-1';
    alertTitle.textContent = 'CONTINUE RUNNING';
    alertDesc.textContent = 'Risk probability is below calibrated threshold (t* = 0.020). Preventative inspection not required.';

    if (window.lucide) {
      stateIcon.setAttribute('data-lucide', 'shield-check');
      stateIcon.className = 'w-4 h-4 text-emerald-400';
      alertIcon.setAttribute('data-lucide', 'check-circle');
      lucide.createIcons();
    }
  }

  // 5. Ground Truth Badge
  const gt = tick.ground_truth;
  const gtBadge = document.getElementById('ground-truth-badge');
  if (gt && gt.machine_failure) {
    gtBadge.className = 'px-2 py-0.5 text-[10px] font-mono rounded bg-rose-950 text-rose-400 border border-rose-800/70';
    const modes = gt.active_failure_modes && gt.active_failure_modes.length > 0 ? gt.active_failure_modes.join(', ') : 'Failure';
    gtBadge.textContent = `Ground Truth: FAULT (${modes})`;
  } else {
    gtBadge.className = 'px-2 py-0.5 text-[10px] font-mono rounded bg-zinc-800 text-zinc-400';
    gtBadge.textContent = 'Ground Truth: Normal (0)';
  }
}

/**
 * Attaches transport controls (Play, Pause, Step, Scrub, Speed, Jump).
 */
function setupTransportControls() {
  const btnPlayPause = document.getElementById('btn-play-pause');
  const btnStep = document.getElementById('btn-step');
  const btnReset = document.getElementById('btn-reset');
  const timelineSlider = document.getElementById('timeline-slider');

  // Play / Pause toggle
  btnPlayPause.addEventListener('click', () => {
    if (state.isPlaying) {
      pausePlayback();
    } else {
      startPlayback();
    }
  });

  // Step +1 tick
  btnStep.addEventListener('click', () => {
    pausePlayback();
    const nextIdx = (state.currentIndex + 1) % state.maxTicks;
    loadTick(nextIdx);
  });

  // Reset to tick 0
  btnReset.addEventListener('click', () => {
    pausePlayback();
    loadInitialWindow(0);
  });

  // Timeline slider scrubbing
  timelineSlider.addEventListener('input', (e) => {
    pausePlayback();
    const targetIdx = parseInt(e.target.value, 10);
    loadTick(targetIdx);
  });

  // Speed toggles (1x, 2x, 5x)
  const speedButtons = document.querySelectorAll('.speed-btn');
  speedButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      speedButtons.forEach(b => {
        b.className = 'speed-btn px-2 py-1 rounded bg-zinc-900 text-zinc-400 hover:bg-zinc-800 border border-zinc-800';
      });
      btn.className = 'speed-btn px-2 py-1 rounded bg-zinc-800 text-zinc-200 border border-zinc-700';

      state.playbackSpeed = parseFloat(btn.dataset.speed);
      if (state.isPlaying) {
        // Restart timer at new speed
        clearInterval(state.timerId);
        state.timerId = setInterval(tickStep, Math.round(1000 / state.playbackSpeed));
      }
    });
  });

  // Jump to historical failure events
  const jumpButtons = document.querySelectorAll('.jump-btn');
  jumpButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      pausePlayback();
      const targetIdx = parseInt(btn.dataset.index, 10);
      // Load window leading up to target failure for rich context
      const startIdx = Math.max(0, targetIdx - 15);
      loadInitialWindow(startIdx).then(() => {
        loadTick(targetIdx);
      });
    });
  });
}

/**
 * Starts continuous playback timer.
 */
function startPlayback() {
  state.isPlaying = true;
  const playIcon = document.getElementById('play-pause-icon');
  const playText = document.getElementById('play-pause-text');
  playText.textContent = 'Pause Replay';
  if (window.lucide) {
    playIcon.setAttribute('data-lucide', 'pause');
    lucide.createIcons();
  }

  const intervalMs = Math.round(1000 / state.playbackSpeed);
  state.timerId = setInterval(tickStep, intervalMs);
}

/**
 * Pauses continuous playback timer.
 */
function pausePlayback() {
  state.isPlaying = false;
  clearInterval(state.timerId);
  state.timerId = null;

  const playIcon = document.getElementById('play-pause-icon');
  const playText = document.getElementById('play-pause-text');
  playText.textContent = 'Stream Replay';
  if (window.lucide) {
    playIcon.setAttribute('data-lucide', 'play');
    lucide.createIcons();
  }
}

/**
 * Single tick step executed by the playback timer.
 */
function tickStep() {
  const nextIdx = (state.currentIndex + 1) % state.maxTicks;
  loadTick(nextIdx);
}

/**
 * Tab switching (TreeSHAP vs What-If Sandbox).
 */
function setupTabControls() {
  const btnShap = document.getElementById('tab-btn-shap');
  const btnSandbox = document.getElementById('tab-btn-sandbox');
  const contentShap = document.getElementById('tab-content-shap');
  const contentSandbox = document.getElementById('tab-content-sandbox');

  btnShap.addEventListener('click', () => {
    btnShap.className = 'px-3.5 py-1.5 rounded-lg bg-zinc-800 text-zinc-100 font-medium text-xs flex items-center space-x-2 border border-zinc-700';
    btnSandbox.className = 'px-3.5 py-1.5 rounded-lg text-zinc-400 hover:text-zinc-200 hover:bg-zinc-850 font-medium text-xs flex items-center space-x-2 transition-colors';
    contentShap.classList.remove('hidden');
    contentSandbox.classList.add('hidden');
  });

  btnSandbox.addEventListener('click', () => {
    btnSandbox.className = 'px-3.5 py-1.5 rounded-lg bg-zinc-800 text-zinc-100 font-medium text-xs flex items-center space-x-2 border border-zinc-700';
    btnShap.className = 'px-3.5 py-1.5 rounded-lg text-zinc-400 hover:text-zinc-200 hover:bg-zinc-850 font-medium text-xs flex items-center space-x-2 transition-colors';
    contentSandbox.classList.remove('hidden');
    contentShap.classList.add('hidden');
  });
}

/**
 * Sets up manual sandbox slider bindings and live /predict evaluation.
 */
function setupSandboxControls() {
  const syncSlider = (id, valId, suffix) => {
    const el = document.getElementById(id);
    const valEl = document.getElementById(valId);
    if (!el || !valEl) return;
    el.addEventListener('input', () => {
      valEl.textContent = `${el.value} ${suffix}`;
      debouncedEvaluateSandbox();
    });
  };

  syncSlider('sb-air', 'sb-air-val', 'K');
  syncSlider('sb-proc', 'sb-proc-val', 'K');
  syncSlider('sb-speed', 'sb-speed-val', 'rpm');
  syncSlider('sb-torque', 'sb-torque-val', 'Nm');
  syncSlider('sb-wear', 'sb-wear-val', 'min');

  const typeSelect = document.getElementById('sb-type');
  if (typeSelect) {
    typeSelect.addEventListener('change', () => {
      debouncedEvaluateSandbox();
    });
  }

  const evalBtn = document.getElementById('btn-sandbox-eval');
  if (evalBtn) {
    evalBtn.addEventListener('click', () => {
      evaluateSandbox();
    });
  }
}

// Simple debounce utility for smooth real-time slider responsiveness
let debounceTimer = null;
function debouncedEvaluateSandbox() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    evaluateSandbox();
  }, 200);
}

/**
 * Sends hypothetical physical parameters to POST /predict and updates live feedback pill.
 */
async function evaluateSandbox() {
  const resultPill = document.getElementById('sandbox-result-pill');
  const evalBtn = document.getElementById('btn-sandbox-eval');
  if (!resultPill) return;

  const payload = {
    machine_type: document.getElementById('sb-type').value,
    air_temperature_k: parseFloat(document.getElementById('sb-air').value),
    process_temperature_k: parseFloat(document.getElementById('sb-proc').value),
    rotational_speed_rpm: parseFloat(document.getElementById('sb-speed').value),
    torque_nm: parseFloat(document.getElementById('sb-torque').value),
    tool_wear_min: parseFloat(document.getElementById('sb-wear').value),
  };

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) throw new Error(`Prediction failed (${res.status})`);
    const pred = await res.json();

    const probPct = (pred.failure_probability * 100).toFixed(2);
    const cutoffPct = (pred.decision_threshold * 100).toFixed(2);

    if (pred.risk_tier === 'CRITICAL' || pred.failure_predicted) {
      resultPill.innerHTML = `
        <div class="flex items-center space-x-2 text-rose-400 font-medium">
          <span class="w-2 h-2 rounded-full bg-rose-500 animate-pulse"></span>
          <span>CRITICAL RISK: ${probPct}% (Cutoff: ${cutoffPct}%)</span>
          <span class="text-zinc-500 font-normal">· ${pred.recommended_action}</span>
        </div>
      `;
    } else if (pred.risk_tier === 'WARNING' || pred.failure_probability >= 0.010) {
      resultPill.innerHTML = `
        <div class="flex items-center space-x-2 text-amber-400 font-medium">
          <span class="w-2 h-2 rounded-full bg-amber-500"></span>
          <span>WARNING RISK: ${probPct}% (Cutoff: ${cutoffPct}%)</span>
          <span class="text-zinc-500 font-normal">· ${pred.recommended_action}</span>
        </div>
      `;
    } else {
      resultPill.innerHTML = `
        <div class="flex items-center space-x-2 text-emerald-400 font-medium">
          <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
          <span>OPTIMAL: ${probPct}%</span>
          <span class="text-zinc-500 font-normal">· Nominal physical parameters</span>
        </div>
      `;
    }
  } catch (err) {
    console.error('Error evaluating sandbox scenario:', err);
    resultPill.innerHTML = `<span class="text-rose-400">Evaluation error: ${err.message}</span>`;
  }
}

/**
 * Attaches the TreeSHAP Root Cause Inspector trigger and renders feature attributions.
 */
function setupShapInspector() {
  const btnInspect = document.getElementById('btn-inspect-shap');
  if (!btnInspect) return;

  btnInspect.addEventListener('click', async () => {
    if (!state.latestTick) {
      alert('No telemetry tick selected yet. Play or step the stream first.');
      return;
    }

    // Automatically switch to TreeSHAP tab
    const btnShap = document.getElementById('tab-btn-shap');
    if (btnShap) btnShap.click();

    // Show loading state
    const originalText = btnInspect.innerHTML;
    btnInspect.disabled = true;
    btnInspect.innerHTML = `
      <svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-purple-400 inline-block" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
      </svg>
      <span>Computing TreeSHAP...</span>
    `;

    try {
      const sensors = state.latestTick.telemetry;
      const payload = {
        machine_type: sensors.machine_type,
        air_temperature_k: sensors.air_temperature_k,
        process_temperature_k: sensors.process_temperature_k,
        rotational_speed_rpm: sensors.rotational_speed_rpm,
        torque_nm: sensors.torque_nm,
        tool_wear_min: sensors.tool_wear_min,
      };

      const res = await fetch('/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error(`SHAP explanation failed (${res.status})`);
      const explanation = await res.json();

      renderShapAttributions(explanation, state.latestTick.index);
    } catch (err) {
      console.error('Error fetching SHAP explanation:', err);
      const container = document.getElementById('shap-bars-list');
      container.innerHTML = `<div class="text-xs text-rose-400 py-4 text-center">Failed to compute TreeSHAP: ${err.message}</div>`;
    } finally {
      btnInspect.disabled = false;
      btnInspect.innerHTML = originalText;
      if (window.lucide) lucide.createIcons();
    }
  });
}

/**
 * Human-readable feature name mapping.
 */
const FEATURE_PRETTY_NAMES = {
  overstrain_ratio: 'Overstrain Stress Ratio',
  power_w: 'Mechanical Power (W)',
  temp_ratio: 'Temp Ratio (Process / Air)',
  temp_diff: 'Thermal Gradient ΔT (K)',
  overstrain: 'Absolute Overstrain (τ × Wear)',
  torque_nm: 'Torque (Nm)',
  rotational_speed_rpm: 'Rotational Speed (rpm)',
  tool_wear_min: 'Tool Wear (min)',
  air_temperature_k: 'Air Temperature (K)',
  process_temperature_k: 'Process Temperature (K)',
  Type: 'Machine Quality Tier',
};

/**
 * Formats a raw feature value cleanly with appropriate units.
 */
function formatFeatureVal(feature, val) {
  if (feature === 'power_w') return `${Math.round(val).toLocaleString()} W`;
  if (feature === 'torque_nm') return `${val.toFixed(1)} Nm`;
  if (feature === 'rotational_speed_rpm') return `${Math.round(val)} rpm`;
  if (feature === 'tool_wear_min') return `${Math.round(val)} min`;
  if (feature.includes('temp') && !feature.includes('ratio')) return `${val.toFixed(1)} K`;
  if (feature.includes('ratio')) return val.toFixed(3);
  return typeof val === 'number' ? val.toFixed(2) : String(val);
}

/**
 * Renders the TreeSHAP feature attributions and diagnostic explanation.
 */
function renderShapAttributions(explanation, tickIdx) {
  // Update base value log-odds
  const baseValEl = document.getElementById('shap-base-val');
  baseValEl.textContent = `Fleet Base Value: φ₀ = ${explanation.base_value.toFixed(2)} log-odds`;

  const container = document.getElementById('shap-bars-list');
  container.innerHTML = '';

  const drivers = explanation.top_risk_drivers || [];
  if (drivers.length === 0) {
    container.innerHTML = '<div class="text-xs text-zinc-500 py-4 text-center">No feature attributions available.</div>';
    return;
  }

  // Find max absolute SHAP value for proportional bar scaling
  const maxAbsShap = Math.max(...drivers.map(d => Math.abs(d.shap_value)), 0.1);

  drivers.forEach(d => {
    const isRiskDriver = d.shap_value > 0;
    const absVal = Math.abs(d.shap_value);
    const barPct = Math.min(100, Math.max(6, Math.round((absVal / maxAbsShap) * 100)));
    const prettyName = FEATURE_PRETTY_NAMES[d.feature] || d.feature;
    const formattedVal = formatFeatureVal(d.feature, d.value);

    const barColorClass = isRiskDriver ? 'bg-rose-500' : 'bg-emerald-500';
    const textColorClass = isRiskDriver ? 'text-rose-400' : 'text-emerald-400';
    const sign = d.shap_value >= 0 ? '+' : '';

    const row = document.createElement('div');
    row.className = 'p-2 rounded bg-zinc-950 border border-zinc-850 space-y-1.5 transition-all';
    row.innerHTML = `
      <div class="flex items-center justify-between text-xs font-mono">
        <span class="text-zinc-200 font-medium">${prettyName}</span>
        <div class="flex items-center space-x-2">
          <span class="text-zinc-500">${formattedVal}</span>
          <span class="font-semibold ${textColorClass}">${sign}${d.shap_value.toFixed(3)} φ</span>
        </div>
      </div>
      <div class="w-full bg-zinc-800 rounded-full h-1.5 overflow-hidden">
        <div class="${barColorClass} h-1.5 rounded-full transition-all duration-500" style="width: ${barPct}%"></div>
      </div>
    `;
    container.appendChild(row);
  });

  // Update Engineering Explanation Narrative
  const explanationEl = document.getElementById('shap-explanation-text');
  const probaPct = (explanation.failure_probability * 100).toFixed(2);
  const cutoffPct = (explanation.decision_threshold * 100).toFixed(2);

  let statusLead = '';
  if (explanation.failure_predicted) {
    statusLead = `<span class="text-rose-400 font-bold">[CRITICAL ALERT at Tick #${tickIdx}]</span> Calibrated failure risk of <strong class="text-zinc-100">${probaPct}%</strong> breached economic cutoff (${cutoffPct}%).`;
  } else {
    statusLead = `<span class="text-emerald-400 font-bold">[NOMINAL at Tick #${tickIdx}]</span> Machine operating safely with risk of <strong class="text-zinc-100">${probaPct}%</strong> (below ${cutoffPct}% threshold).`;
  }

  explanationEl.innerHTML = `
    <div class="space-y-2">
      <p>${statusLead}</p>
      <div class="p-2.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-300">
        <span class="text-zinc-400 font-semibold block mb-0.5">Primary Root Cause Diagnosis:</span>
        <span class="text-zinc-100">${explanation.primary_concern}</span>
      </div>
      <p class="text-[10px] text-zinc-500">
        Game-theoretic Shapley values guarantee additive fairness: base log-odds (φ₀) plus the sum of all individual feature attributions (Σφᵢ) exactly equals the model's uncalibrated output.
      </p>
    </div>
  `;
}
