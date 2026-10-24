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

  // Initialize rolling Chart.js instances
  initCharts();

  // Attach all DOM event listeners
  setupTransportControls();
  setupTabControls();
  setupSandboxControls();
  setupShapInspector();

  // Verify API health and initiate periodic polling (every 8 seconds)
  await checkApiHealth();
  setInterval(checkApiHealth, 8000);

  // Load initial telemetry window starting at tick 0
  await loadInitialWindow(0);
});

let hasBeenOnline = false;

/**
 * Checks FastAPI backend health and latency with cold-start resilience.
 */
async function checkApiHealth() {
  const statusBadge = document.getElementById('api-status-badge');
  const statusText = document.getElementById('api-status-text');
  if (!statusText) return;
  const startTime = performance.now();

  try {
    const res = await fetch('/health', { cache: 'no-store' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const latency = Math.round(performance.now() - startTime);

    if (statusBadge) statusBadge.className = 'flex items-center justify-between text-xs font-mono';

    if (data.status === 'healthy' && data.model_loaded) {
      hasBeenOnline = true;
      statusText.innerHTML = `<span class="px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 font-mono text-[11px] font-semibold flex items-center gap-1.5"><span class="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>Online (${latency}ms)</span>`;

      // If initial window failed during cold start, auto-load tick 0 now that server is awake
      if (!state.latestTick && state.mechanicalChart && state.thermalChart) {
        await loadInitialWindow(0);
      }
    } else {
      statusText.innerHTML = `<span class="px-2.5 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200 font-mono text-[11px] font-semibold flex items-center gap-1.5"><span class="w-1.5 h-1.5 rounded-full bg-amber-500"></span>Degraded</span>`;
    }
  } catch (err) {
    if (statusBadge) statusBadge.className = 'flex items-center justify-between text-xs font-mono';
    if (!hasBeenOnline) {
      statusText.innerHTML = `<span class="px-2.5 py-0.5 rounded-full bg-amber-50 text-amber-700 border border-amber-200 font-mono text-[11px] font-semibold flex items-center gap-1.5"><span class="w-1.5 h-1.5 rounded-full bg-amber-500 animate-ping"></span>Waking Up...</span>`;
    } else {
      statusText.innerHTML = `<span class="px-2.5 py-0.5 rounded-full bg-rose-50 text-rose-700 border border-rose-200 font-mono text-[11px] font-semibold flex items-center gap-1.5"><span class="w-1.5 h-1.5 rounded-full bg-rose-500"></span>Offline</span>`;
    }
  }
}

/**
 * Initializes the two rolling Chart.js strip-charts matching the reference design:
 * clean white surface, smooth curves, horizontal-only gridlines, and circular dot legends.
 */
function initCharts() {
  const chartFont = {
    family: "'Inter', sans-serif",
    size: 11,
  };

  const commonScales = {
    x: {
      grid: { display: false },
      ticks: { color: '#94a3b8', font: chartFont, maxTicksLimit: 8, padding: 8 },
    },
    y: {
      grid: { color: 'rgba(0, 0, 0, 0.05)', drawBorder: false },
      ticks: { color: '#94a3b8', font: chartFont, padding: 8 },
    }
  };

  // 1. Mechanical Dynamics (Torque in Royal Blue, Speed in Warm Orange)
  const canvasMech = document.getElementById('mechanicalChart');
  const ctxMech = canvasMech.getContext('2d');

  state.mechanicalChart = new Chart(ctxMech, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Torque (Nm)',
          data: [],
          borderColor: '#2563eb', // Restored Vibrant Royal Blue
          borderWidth: 2.5,
          pointRadius: 0,
          pointHoverRadius: 6,
          pointBackgroundColor: '#2563eb',
          tension: 0.45,
          fill: false,
        },
        {
          label: 'Speed / 50 (rpm)',
          data: [],
          borderColor: '#f59e0b', // Warm Orange/Amber
          borderWidth: 2.5,
          pointRadius: 0,
          pointHoverRadius: 6,
          pointBackgroundColor: '#f59e0b',
          tension: 0.45,
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
          backgroundColor: '#ffffff',
          titleColor: '#0f172a',
          bodyColor: '#334155',
          borderColor: '#e2e8f0',
          borderWidth: 1,
          cornerRadius: 12,
          titleFont: { size: 12, weight: '700' },
          bodyFont: { size: 11 },
          padding: 12,
          boxPadding: 6,
          usePointStyle: true,
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

  // 2. Thermal Dynamics (Process Temp in Warm Amber, Air Temp in Cyan/Blue)
  const canvasTherm = document.getElementById('thermalChart');
  const ctxTherm = canvasTherm.getContext('2d');

  state.thermalChart = new Chart(ctxTherm, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Process Temp [K]',
          data: [],
          borderColor: '#f59e0b',
          borderWidth: 2.5,
          pointRadius: 0,
          pointHoverRadius: 6,
          pointBackgroundColor: '#f59e0b',
          tension: 0.45,
          fill: false,
        },
        {
          label: 'Air Temp [K]',
          data: [],
          borderColor: '#0284c7',
          borderWidth: 2.5,
          pointRadius: 0,
          pointHoverRadius: 6,
          pointBackgroundColor: '#0284c7',
          tension: 0.45,
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
          backgroundColor: '#ffffff',
          titleColor: '#0f172a',
          bodyColor: '#334155',
          borderColor: '#e2e8f0',
          borderWidth: 1,
          cornerRadius: 12,
          titleFont: { size: 12, weight: '700' },
          bodyFont: { size: 11 },
          padding: 12,
          boxPadding: 6,
          usePointStyle: true,
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

  // 3. KPI: Calibrated Risk (Dynamic Card: Clean Slate in Nominal, Amber in Warning, Orange in High Risk, Blood Crimson in Critical)
  const riskPctStr = (risk * 100).toFixed(2);
  const riskTextEl = document.getElementById('kpi-risk-text');
  if (riskTextEl) riskTextEl.textContent = `${riskPctStr}%`;

  const riskCard = document.getElementById('kpi-risk-card');
  const riskTitle = document.getElementById('kpi-risk-title');
  const riskIcon = document.getElementById('kpi-risk-icon');
  const riskThresholdSub = document.getElementById('kpi-risk-threshold-subtitle');
  const riskStatusLabel = document.getElementById('kpi-risk-status-label');
  const riskCutoffLabel = document.getElementById('kpi-risk-cutoff-label');
  const riskBar = document.getElementById('kpi-risk-bar');

  // 4. KPI: Machine Health State (Dark Slate Navy in Nominal, Amber in Warning, Orange in High Risk, Deep Blood Maroon in Critical)
  const stateCard = document.getElementById('kpi-state-card');
  const stateTitle = document.getElementById('kpi-state-title');
  const stateIcon = document.getElementById('kpi-state-icon');
  const stateText = document.getElementById('kpi-state-text');
  const stateDesc = document.getElementById('kpi-state-desc');
  const tierBadge = document.getElementById('kpi-state-tier');

  const alertBox = document.getElementById('alert-callout-box');
  const alertTitle = document.getElementById('alert-callout-title');
  const alertDesc = document.getElementById('alert-callout-desc');
  const alertIcon = document.getElementById('alert-callout-icon');

  const tierName = sensors.machine_type === 'L' ? 'Low (L)' : sensors.machine_type === 'M' ? 'Medium (M)' : 'High (H)';
  if (tierBadge) tierBadge.textContent = `Tier: ${tierName}`;

  if (risk >= 0.60) {
    // 🔴 4. Extreme / Catastrophic State (e.g. 90% Risk) - Blood Red Maroon + Emergency Pulse
    if (stateText) stateText.textContent = 'CRITICAL';
    if (stateDesc) {
      stateDesc.textContent = `Catastrophic breakdown imminent (${riskPctStr}%)! Automatic shutdown initiated.`;
      stateDesc.className = 'text-xs text-white font-medium leading-relaxed truncate';
    }
    if (stateTitle) stateTitle.className = 'text-xs font-medium text-white/80 tracking-wide';
    if (stateIcon) stateIcon.className = 'w-4 h-4 text-white/80';
    if (tierBadge) tierBadge.className = 'text-xs font-mono text-white/85 mt-0.5';
    if (stateCard) stateCard.className = 'bg-[#7f1d1d] text-white rounded-3xl p-6 flex flex-col justify-between shadow-xl shadow-rose-950/30 ring-2 ring-rose-500/50 transition-all duration-300 relative overflow-hidden';

    if (riskCard) riskCard.className = 'bg-[#b91c1c] text-white rounded-3xl p-6 flex flex-col justify-between shadow-xl shadow-rose-900/30 transition-all duration-300 relative overflow-hidden';
    if (riskTitle) riskTitle.className = 'text-xs font-medium text-white/80 tracking-wide';
    if (riskIcon) riskIcon.className = 'w-4 h-4 text-white/80';
    if (riskTextEl) riskTextEl.className = 'text-2xl font-bold font-mono tracking-tight text-white';
    if (riskThresholdSub) riskThresholdSub.className = 'text-xs font-mono text-white/85 mt-0.5';
    if (riskCutoffLabel) riskCutoffLabel.className = 'font-mono text-[11px] text-white/80';
    if (riskStatusLabel) {
      riskStatusLabel.className = 'px-2.5 py-0.5 rounded-full bg-rose-950/60 text-white font-mono text-[11px] font-bold border border-rose-400/40 animate-pulse';
      riskStatusLabel.textContent = 'CRITICAL SHUTDOWN';
    }
    if (riskBar) {
      riskBar.className = 'absolute bottom-0 left-0 h-1 bg-rose-400 animate-pulse transition-all duration-500';
      riskBar.style.width = `${Math.min(100, Math.max(5, risk * 100))}%`;
    }

    if (alertBox) alertBox.className = 'p-4 rounded-2xl bg-rose-100 border-2 border-rose-600 text-rose-950 flex flex-col gap-1.5 transition-all';
    if (alertTitle) alertTitle.textContent = 'EMERGENCY CRITICAL SHUTDOWN';
    if (alertDesc) alertDesc.textContent = `Extreme catastrophic failure risk (${riskPctStr}%). Automatic machine halt recommended immediately to avoid $5,000+ spindle ruin.`;

    if (window.lucide && alertIcon) {
      alertIcon.setAttribute('data-lucide', 'alert-octagon');
      alertIcon.className = 'w-4 h-4 text-rose-600 animate-pulse';
      lucide.createIcons();
    }
  } else if (risk >= THRESHOLD || pred.risk_tier === 'CRITICAL') {
    // 🟠 3. High Risk / Actionable Mid-Range Fault (e.g. 40% Risk) - Vivid Orange (Clearly different from 90%)
    if (stateText) stateText.textContent = 'HIGH RISK';
    if (stateDesc) {
      stateDesc.textContent = `Failure risk (${riskPctStr}%) breached economic cutoff (t* = 2.0%). Tool maintenance required.`;
      stateDesc.className = 'text-xs text-white font-medium leading-relaxed truncate';
    }
    if (stateTitle) stateTitle.className = 'text-xs font-medium text-white/80 tracking-wide';
    if (stateIcon) stateIcon.className = 'w-4 h-4 text-white/80';
    if (tierBadge) tierBadge.className = 'text-xs font-mono text-white/85 mt-0.5';
    if (stateCard) stateCard.className = 'bg-[#ea580c] text-white rounded-3xl p-6 flex flex-col justify-between shadow-lg shadow-orange-500/25 transition-all duration-300 relative overflow-hidden';

    if (riskCard) riskCard.className = 'bg-[#f97316] text-white rounded-3xl p-6 flex flex-col justify-between shadow-lg shadow-orange-500/20 transition-all duration-300 relative overflow-hidden';
    if (riskTitle) riskTitle.className = 'text-xs font-medium text-white/80 tracking-wide';
    if (riskIcon) riskIcon.className = 'w-4 h-4 text-white/80';
    if (riskTextEl) riskTextEl.className = 'text-2xl font-bold font-mono tracking-tight text-white';
    if (riskThresholdSub) riskThresholdSub.className = 'text-xs font-mono text-white/85 mt-0.5';
    if (riskCutoffLabel) riskCutoffLabel.className = 'font-mono text-[11px] text-white/80';
    if (riskStatusLabel) {
      riskStatusLabel.className = 'px-2.5 py-0.5 rounded-full bg-white/25 text-white font-mono text-[11px] font-bold';
      riskStatusLabel.textContent = `HIGH RISK (${riskPctStr}%)`;
    }
    if (riskBar) {
      riskBar.className = 'absolute bottom-0 left-0 h-1 bg-orange-200 transition-all duration-500';
      riskBar.style.width = `${Math.min(100, Math.max(5, risk * 100))}%`;
    }

    if (alertBox) alertBox.className = 'p-4 rounded-2xl bg-orange-50 border-2 border-orange-400 text-orange-950 flex flex-col gap-1.5 transition-all';
    if (alertTitle) alertTitle.textContent = 'ACTIONABLE ALERT: MAINTENANCE REQUIRED';
    if (alertDesc) alertDesc.textContent = `AI Risk (${riskPctStr}%) exceeds calibrated cutoff (2.0%). Schedule inspection and tool swap before next operational shift.`;

    if (window.lucide && alertIcon) {
      alertIcon.setAttribute('data-lucide', 'alert-triangle');
      alertIcon.className = 'w-4 h-4 text-orange-600';
      lucide.createIcons();
    }
  } else if (pred.risk_tier === 'WARNING' || risk >= 0.010) {
    // 🟡 2. Warning / Advisory State (1.0% - 2.0%) - Warm Amber
    if (stateText) stateText.textContent = 'WARNING';
    if (stateDesc) {
      stateDesc.textContent = 'Thermal or mechanical stress is elevating near threshold limits. Monitor closely.';
      stateDesc.className = 'text-xs text-white font-medium leading-relaxed truncate';
    }
    if (stateTitle) stateTitle.className = 'text-xs font-medium text-white/80 tracking-wide';
    if (stateIcon) stateIcon.className = 'w-4 h-4 text-white/80';
    if (tierBadge) tierBadge.className = 'text-xs font-mono text-white/85 mt-0.5';
    if (stateCard) stateCard.className = 'bg-[#d97706] text-white rounded-3xl p-6 flex flex-col justify-between shadow-lg shadow-amber-500/20 transition-all duration-300 relative overflow-hidden';

    if (riskCard) riskCard.className = 'bg-[#f59e0b] text-white rounded-3xl p-6 flex flex-col justify-between shadow-lg shadow-amber-500/20 transition-all duration-300 relative overflow-hidden';
    if (riskTitle) riskTitle.className = 'text-xs font-medium text-white/80 tracking-wide';
    if (riskIcon) riskIcon.className = 'w-4 h-4 text-white/80';
    if (riskTextEl) riskTextEl.className = 'text-2xl font-bold font-mono tracking-tight text-white';
    if (riskThresholdSub) riskThresholdSub.className = 'text-xs font-mono text-white/85 mt-0.5';
    if (riskCutoffLabel) riskCutoffLabel.className = 'font-mono text-[11px] text-white/80';
    if (riskStatusLabel) {
      riskStatusLabel.className = 'px-2.5 py-0.5 rounded-full bg-white/20 text-white font-mono text-[11px] font-semibold';
      riskStatusLabel.textContent = 'ELEVATED RISK';
    }
    if (riskBar) {
      riskBar.className = 'absolute bottom-0 left-0 h-1 bg-amber-200 transition-all duration-500';
      riskBar.style.width = `${Math.min(100, Math.max(5, risk * 100))}%`;
    }

    if (alertBox) alertBox.className = 'p-4 rounded-2xl bg-amber-50 border-2 border-amber-400 text-amber-950 flex flex-col gap-1.5 transition-all';
    if (alertTitle) alertTitle.textContent = 'PREVENTATIVE INSPECTION ADVISORY';
    if (alertDesc) alertDesc.textContent = `Telemetry shows elevated strain (Risk: ${riskPctStr}%). Approaching 2.0% cutoff threshold.`;

    if (window.lucide && alertIcon) {
      alertIcon.setAttribute('data-lucide', 'alert-triangle');
      alertIcon.className = 'w-4 h-4 text-amber-600';
      lucide.createIcons();
    }
  } else {
    // 🟢 1. Optimal Normal State (< 1.0%) - Clean Shadcn Black + White Card (No Red Anywhere)
    if (stateText) stateText.textContent = 'OPTIMAL';
    if (stateDesc) {
      stateDesc.textContent = 'Safe mechanical limits.';
      stateDesc.className = 'text-xs text-slate-300 leading-relaxed truncate';
    }
    if (stateTitle) stateTitle.className = 'text-xs font-medium text-slate-400 tracking-wide';
    if (stateIcon) stateIcon.className = 'w-4 h-4 text-slate-400';
    if (tierBadge) tierBadge.className = 'text-xs font-mono text-slate-400 mt-0.5';
    if (stateCard) stateCard.className = 'bg-slate-900 text-white rounded-3xl p-6 flex flex-col justify-between shadow-lg shadow-slate-900/10 transition-all duration-300 relative overflow-hidden';

    if (riskCard) riskCard.className = 'bg-white border border-slate-200/90 text-slate-900 rounded-3xl p-6 flex flex-col justify-between shadow-sm transition-all duration-300 relative overflow-hidden';
    if (riskTitle) riskTitle.className = 'text-xs font-medium text-slate-500 tracking-wide';
    if (riskIcon) riskIcon.className = 'w-4 h-4 text-slate-400';
    if (riskTextEl) riskTextEl.className = 'text-2xl font-bold font-mono tracking-tight text-slate-900';
    if (riskThresholdSub) riskThresholdSub.className = 'text-xs font-mono text-slate-400 mt-0.5';
    if (riskCutoffLabel) riskCutoffLabel.className = 'font-mono text-[11px] text-slate-400';
    if (riskStatusLabel) {
      riskStatusLabel.className = 'px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 font-mono text-[11px] font-medium';
      riskStatusLabel.textContent = 'Safe (<1%)';
    }
    if (riskBar) {
      riskBar.className = 'absolute bottom-0 left-0 h-1 bg-emerald-400 transition-all duration-500';
      riskBar.style.width = `${Math.max(4, risk * 100)}%`;
    }

    if (alertBox) alertBox.className = 'p-4 rounded-2xl bg-emerald-50 border border-emerald-200 text-emerald-800 flex flex-col gap-1.5 transition-all';
    if (alertTitle) alertTitle.textContent = 'CONTINUE RUNNING';
    if (alertDesc) alertDesc.textContent = 'Risk probability is below calibrated threshold (t* = 2.0%). Machine running nominal.';

    if (window.lucide && alertIcon) {
      alertIcon.setAttribute('data-lucide', 'check-circle');
      alertIcon.className = 'w-4 h-4 text-emerald-600';
      lucide.createIcons();
    }
  }

  // 5. Ground Truth Badge
  const gt = tick.ground_truth;
  const gtBadge = document.getElementById('ground-truth-badge');
  if (gt && gt.machine_failure) {
    gtBadge.className = 'px-2.5 py-1 text-[11px] font-mono rounded-lg bg-rose-100 text-rose-700 border border-rose-200 font-semibold';
    const modes = gt.active_failure_modes && gt.active_failure_modes.length > 0 ? gt.active_failure_modes.join(', ') : 'Failure';
    gtBadge.textContent = `Ground Truth: FAULT (${modes})`;
  } else {
    gtBadge.className = 'px-2.5 py-1 text-[11px] font-mono rounded-lg bg-slate-100 text-slate-600 font-medium';
    gtBadge.textContent = 'Ground Truth: Normal (0)';
  }
}

/**
 * Attaches transport controls (Play, Pause, Step, Scrub, Speed, Jump).
 */
function setupTransportControls() {
  const btnPlayPause = document.getElementById('btn-play-pause');
  const btnStepBack = document.getElementById('btn-step-back');
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

  // Step -1 tick backward
  if (btnStepBack) {
    btnStepBack.addEventListener('click', () => {
      pausePlayback();
      const prevIdx = (state.currentIndex - 1 + state.maxTicks) % state.maxTicks;
      loadTick(prevIdx);
    });
  }

  // Step +1 tick forward
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

  // Speed toggles (1x, 2x, 5x) - Unified rounded-2xl styling
  const speedButtons = document.querySelectorAll('.speed-btn');
  speedButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      speedButtons.forEach(b => {
        b.className = 'speed-btn px-3 py-1 rounded-2xl bg-slate-100 text-slate-600 hover:bg-slate-200 font-medium cursor-pointer transition-colors';
      });
      btn.className = 'speed-btn px-3 py-1 rounded-2xl bg-slate-900 text-white font-medium cursor-pointer shadow-sm';

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

  const activeClasses = 'w-full py-2.5 px-4 rounded-2xl bg-slate-900 text-white font-bold text-xs flex items-center justify-center gap-2 shadow-sm transition-all cursor-pointer';
  const inactiveClasses = 'w-full py-2.5 px-4 rounded-2xl text-slate-600 hover:text-slate-900 hover:bg-slate-200/60 font-semibold text-xs flex items-center justify-center gap-2 transition-all cursor-pointer';

  btnShap.addEventListener('click', () => {
    btnShap.className = activeClasses;
    const iconShap = btnShap.querySelector('i');
    if (iconShap) iconShap.className = 'w-3.5 h-3.5 text-white';

    btnSandbox.className = inactiveClasses;
    const iconSb = btnSandbox.querySelector('i');
    if (iconSb) iconSb.className = 'w-3.5 h-3.5 text-slate-500';

    contentShap.classList.remove('hidden');
    contentSandbox.classList.add('hidden');
  });

  btnSandbox.addEventListener('click', () => {
    btnSandbox.className = activeClasses;
    const iconSb = btnSandbox.querySelector('i');
    if (iconSb) iconSb.className = 'w-3.5 h-3.5 text-white';

    btnShap.className = inactiveClasses;
    const iconShap = btnShap.querySelector('i');
    if (iconShap) iconShap.className = 'w-3.5 h-3.5 text-slate-500';

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

  // Reactive initial evaluation on startup
  evaluateSandbox();
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

    if (pred.failure_probability >= 0.60) {
      resultPill.innerHTML = `
        <div class="w-full flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-rose-950 font-bold text-xs bg-rose-100 border-2 border-rose-600 p-4 rounded-2xl shadow-sm">
          <div class="flex items-center gap-3">
            <span class="w-3 h-3 rounded-full bg-rose-600 animate-pulse shrink-0"></span>
            <div class="flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-3">
              <span class="font-bold text-sm text-rose-950">CRITICAL SHUTDOWN: ${probPct}%</span>
              <span class="text-rose-900 font-normal">· Catastrophic failure imminent! ${pred.recommended_action}</span>
            </div>
          </div>
          <span class="text-xs font-mono font-semibold text-rose-900/80 shrink-0">Cutoff: ${cutoffPct}% (Breached)</span>
        </div>
      `;
    } else if (pred.risk_tier === 'CRITICAL' || pred.failure_probability >= pred.decision_threshold) {
      resultPill.innerHTML = `
        <div class="w-full flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-orange-950 font-bold text-xs bg-orange-50 border-2 border-orange-400 p-4 rounded-2xl shadow-sm">
          <div class="flex items-center gap-3">
            <span class="w-3 h-3 rounded-full bg-orange-500 shrink-0"></span>
            <div class="flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-3">
              <span class="font-bold text-sm text-orange-950">HIGH RISK: ${probPct}%</span>
              <span class="text-orange-900 font-normal">· Actionable fault. ${pred.recommended_action}</span>
            </div>
          </div>
          <span class="text-xs font-mono font-semibold text-orange-900/80 shrink-0">Cutoff: ${cutoffPct}% (Breached)</span>
        </div>
      `;
    } else if (pred.risk_tier === 'WARNING' || pred.failure_probability >= 0.010) {
      resultPill.innerHTML = `
        <div class="w-full flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-amber-950 font-bold text-xs bg-amber-50 border-2 border-amber-400 p-4 rounded-2xl shadow-sm">
          <div class="flex items-center gap-3">
            <span class="w-3 h-3 rounded-full bg-amber-500 shrink-0"></span>
            <div class="flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-3">
              <span class="font-bold text-sm text-amber-950">ELEVATED WARNING: ${probPct}%</span>
              <span class="text-amber-900 font-normal">· Approaching threshold. ${pred.recommended_action}</span>
            </div>
          </div>
          <span class="text-xs font-mono font-semibold text-amber-900/80 shrink-0">Cutoff: ${cutoffPct}%</span>
        </div>
      `;
    } else {
      resultPill.innerHTML = `
        <div class="w-full flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-emerald-950 font-bold text-xs bg-emerald-50 border border-emerald-300 p-4 rounded-2xl shadow-sm">
          <div class="flex items-center gap-3">
            <span class="w-3 h-3 rounded-full bg-emerald-600 shrink-0"></span>
            <div class="flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-3">
              <span class="font-bold text-sm text-emerald-950">OPTIMAL OPERATION: ${probPct}%</span>
              <span class="text-emerald-900 font-normal">· Nominal physical parameters</span>
            </div>
          </div>
          <span class="text-xs font-mono font-semibold text-emerald-800 shrink-0">Threshold: ${cutoffPct}%</span>
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

  drivers.forEach((d, idx) => {
    const isRiskDriver = d.shap_value > 0;
    const absVal = Math.abs(d.shap_value);
    const barPct = Math.min(100, Math.max(4, Math.round((absVal / maxAbsShap) * 100)));
    const prettyName = FEATURE_PRETTY_NAMES[d.feature] || d.feature;
    const formattedVal = formatFeatureVal(d.feature, d.value);

    // Primary positive driver gets Shadcn Black, secondary positive gets Rose, negative gets Emerald
    let barColorClass = 'bg-rose-500';
    let textColorClass = 'text-rose-600';
    if (idx === 0 && isRiskDriver) {
      barColorClass = 'bg-slate-900';
      textColorClass = 'text-slate-900';
    } else if (!isRiskDriver) {
      barColorClass = 'bg-emerald-500';
      textColorClass = 'text-emerald-600';
    }

    const sign = d.shap_value >= 0 ? '+' : '';

    const row = document.createElement('div');
    row.className = 'flex items-center gap-4 py-1.5 px-2 hover:bg-slate-50/80 rounded-xl transition-colors group text-xs';
    row.innerHTML = `
      <!-- Left: Feature Name (Fixed width label) -->
      <div class="w-44 sm:w-52 shrink-0 text-slate-600 font-sans font-medium truncate group-hover:text-slate-900 transition-colors" title="${prettyName}">
        ${prettyName}
      </div>

      <!-- Center: Full-Width Horizontal Capsule Track (Styled like "Umur Mesin" reference) -->
      <div class="flex-1 bg-slate-100 rounded-full h-3.5 relative overflow-hidden flex items-center">
        <div class="${barColorClass} h-3.5 rounded-full transition-all duration-500 shadow-xs" style="width: ${barPct}%"></div>
      </div>

      <!-- Right: Raw Sensor Value + Shapley Attribution Delta -->
      <div class="w-32 sm:w-36 shrink-0 text-right font-mono flex items-center justify-end gap-2 text-xs">
        <span class="text-slate-400 font-normal text-[11px]">${formattedVal}</span>
        <span class="font-bold ${textColorClass}">${sign}${d.shap_value.toFixed(2)} φ</span>
      </div>
    `;
    container.appendChild(row);
  });

  // Update Engineering Explanation Narrative (Full-width banner on top)
  const explanationEl = document.getElementById('shap-explanation-text');
  const probaPct = (explanation.failure_probability * 100).toFixed(2);
  const cutoffPct = (explanation.decision_threshold * 100).toFixed(2);

  let statusLead = '';
  if (explanation.failure_predicted) {
    statusLead = `<span class="text-rose-600 font-bold">[CRITICAL ALERT at Tick #${tickIdx}]</span> Calibrated failure risk of <strong class="text-slate-900">${probaPct}%</strong> breached economic cutoff (${cutoffPct}%). Immediate shutdown advised.`;
  } else {
    statusLead = `<span class="text-emerald-600 font-bold">[NOMINAL at Tick #${tickIdx}]</span> Machine operating safely with risk of <strong class="text-slate-900">${probaPct}%</strong> (below ${cutoffPct}% threshold).`;
  }

  explanationEl.innerHTML = `
    <div class="flex flex-col gap-2.5">
      <p class="leading-relaxed">${statusLead}</p>
      <div class="p-3.5 rounded-xl bg-white border border-slate-200/90 text-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-2 shadow-xs">
        <div class="flex items-center gap-2">
          <span class="text-slate-400 font-mono font-bold text-[11px] uppercase tracking-wider">Root Cause:</span>
          <span class="text-slate-900 text-xs font-sans font-semibold">${explanation.primary_concern}</span>
        </div>
        <span class="text-[11px] font-mono text-slate-400">Additive Fairness: φ₀ + Σφᵢ = log-odds</span>
      </div>
    </div>
  `;
}
