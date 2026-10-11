// AURA-GRID AI: Real-Time TinyML Inference & Alert System Logic
// Grounded on 1,750-sample high-voltage insulator research paper dataset

// State variables
let currentSensors = {
  Voltage_kV: 11.05,
  Leakage_Current_mA: 0.15,
  Temperature_C: 35.2,
  Relative_Humidity_pct: 65.0,
  Pollution_Index: 0.08,
  Leakage_Current_THD_pct: 0.35,
  Phase_Angle_deg: -2.4,
  Partial_Discharge_Count: 1.0
};

let isStreaming = false;
let streamInterval = null;
let streamIndex = 0;
let audioEnabled = true;
let audioCtx = null;
let lastClass = null;
let incidentLogs = [];

// Audio Alert Synthesizer using Web Audio API
function initAudio() {
  if (!audioCtx) {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (AudioContext) {
      audioCtx = new AudioContext();
    }
  }
}

function playAlarmTone(alertType) {
  if (!audioEnabled) return;
  try {
    initAudio();
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }

    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain);
    gain.connect(audioCtx.destination);

    const now = audioCtx.currentTime;

    if (alertType === 'CRITICAL' || alertType === 'TRIP') {
      // Emergency two-tone alternating siren
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(880, now);
      osc.frequency.linearRampToValueAtTime(440, now + 0.25);
      osc.frequency.linearRampToValueAtTime(880, now + 0.5);

      gain.gain.setValueAtTime(0.25, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.5);

      osc.start(now);
      osc.stop(now + 0.5);
    } else if (alertType === 'WARNING') {
      // Double beep for moderate fault warning
      osc.type = 'square';
      osc.frequency.setValueAtTime(659.25, now);
      gain.gain.setValueAtTime(0.15, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.15);

      osc.start(now);
      osc.stop(now + 0.15);
    } else if (alertType === 'ADVISORY') {
      // Single soft chirp
      osc.type = 'sine';
      osc.frequency.setValueAtTime(523.25, now);
      gain.gain.setValueAtTime(0.1, now);
      gain.gain.exponentialRampToValueAtTime(0.01, now + 0.1);

      osc.start(now);
      osc.stop(now + 0.1);
    }
  } catch (err) {
    console.warn("Audio play prevented or unavailable:", err);
  }
}

// Neural Network Forward Pass in pure JavaScript
function predictTinyML(sensorObj) {
  const model = TINYML_MODEL_DATA;
  const features = model.input_features;
  const norm = model.normalization;

  // 1. Normalize input features strictly using training parameters
  const x_norm = [];
  for (let i = 0; i < features.length; i++) {
    const fName = features[i];
    const val = sensorObj[fName] !== undefined ? sensorObj[fName] : norm[fName].mean;
    const mean = norm[fName].mean;
    const std = norm[fName].std || 1.0;
    x_norm.push((val - mean) / std);
  }

  // 2. Layer 1: Dense(16, ReLU)
  const l1_W = model.weights[0].weights; // [num_features][16]
  const l1_b = model.weights[0].bias;    // [16]
  const h1 = new Array(16).fill(0);

  for (let j = 0; j < 16; j++) {
    let sum = l1_b[j];
    for (let i = 0; i < features.length; i++) {
      sum += x_norm[i] * l1_W[i][j];
    }
    h1[j] = Math.max(0, sum); // ReLU activation
  }

  // 3. Layer 2: Dense(8, ReLU)
  const l2_W = model.weights[1].weights; // [16][8]
  const l2_b = model.weights[1].bias;    // [8]
  const h2 = new Array(8).fill(0);

  for (let k = 0; k < 8; k++) {
    let sum = l2_b[k];
    for (let j = 0; j < 16; j++) {
      sum += h1[j] * l2_W[j][k];
    }
    h2[k] = Math.max(0, sum); // ReLU activation
  }

  // 4. Output Layer: Dense(4, Softmax)
  const out_W = model.weights[2].weights; // [8][4]
  const out_b = model.weights[2].bias;    // [4]
  const logits = new Array(4).fill(0);

  let maxLogit = -Infinity;
  for (let c = 0; c < 4; c++) {
    let sum = out_b[c];
    for (let k = 0; k < 8; k++) {
      sum += h2[k] * out_W[k][c];
    }
    logits[c] = sum;
    if (sum > maxLogit) maxLogit = sum;
  }

  // Softmax with numerical stability
  const expValues = logits.map(v => Math.exp(v - maxLogit));
  const sumExp = expValues.reduce((a, b) => a + b, 0);
  const probabilities = expValues.map(v => v / sumExp);

  // 5. Determine Predicted Class
  let maxIdx = 0;
  for (let c = 1; c < 4; c++) {
    if (probabilities[c] > probabilities[maxIdx]) {
      maxIdx = c;
    }
  }

  return {
    predictedClass: maxIdx,
    confidence: probabilities[maxIdx],
    probabilities: probabilities
  };
}

// Deterministic Safety Threshold Checking (Runs alongside TinyML)
function checkSafetyInterlock(sensorObj) {
  const safety = TINYML_MODEL_DATA.safety_thresholds;

  if (sensorObj.Leakage_Current_mA >= safety.Leakage_Current_mA.trip_threshold) {
    return {
      tripped: true,
      reason: `OVERCURRENT TRIP: Leakage Current (${sensorObj.Leakage_Current_mA.toFixed(2)} mA) exceeds absolute hardware limit of 25.0 mA!`
    };
  }

  if (sensorObj.Voltage_kV >= safety.Voltage_kV.danger_max) {
    return {
      tripped: true,
      reason: `OVERVOLTAGE TRIP: Grid Voltage (${sensorObj.Voltage_kV.toFixed(2)} kV) breached 14.0 kV surge threshold!`
    };
  }

  if (sensorObj.Temperature_C >= safety.Temperature_C.trip_threshold) {
    return {
      tripped: true,
      reason: `THERMAL TRIP: Temperature (${sensorObj.Temperature_C.toFixed(1)} °C) breached 85.0 °C equipment threshold!`
    };
  }

  return { tripped: false };
}

// Update the entire UI state based on current sensors and AI decision
function updateDashboard() {
  // Update sensor numerical displays
  for (const [key, val] of Object.entries(currentSensors)) {
    const el = document.getElementById(`val_${key}`);
    if (el) el.textContent = typeof val === 'number' ? val.toFixed(key === 'Partial_Discharge_Count' ? 0 : (key === 'Pollution_Index' || key === 'Voltage_kV' || key === 'Leakage_Current_mA' ? 2 : 1)) : val;

    // Update mini gauge bars
    const bar = document.getElementById(`bar_${key}`);
    const norm = TINYML_MODEL_DATA.normalization[key];
    if (bar && norm) {
      const min = norm.min || 0;
      const max = norm.max || 100;
      const pct = Math.max(4, Math.min(100, ((val - min) / (max - min)) * 100));
      bar.style.width = `${pct}%`;
    }
  }

  // Update slider displays
  if (document.getElementById('sliderVal_Leakage')) {
    document.getElementById('sliderVal_Leakage').textContent = `${currentSensors.Leakage_Current_mA.toFixed(2)} mA`;
    document.getElementById('slider_Leakage').value = currentSensors.Leakage_Current_mA;
  }
  if (document.getElementById('sliderVal_PD')) {
    document.getElementById('sliderVal_PD').textContent = `${Math.round(currentSensors.Partial_Discharge_Count)} events`;
    document.getElementById('slider_PD').value = currentSensors.Partial_Discharge_Count;
  }
  if (document.getElementById('sliderVal_Pollution')) {
    document.getElementById('sliderVal_Pollution').textContent = currentSensors.Pollution_Index.toFixed(2);
    document.getElementById('slider_Pollution').value = currentSensors.Pollution_Index;
  }
  if (document.getElementById('sliderVal_THD')) {
    document.getElementById('sliderVal_THD').textContent = `${currentSensors.Leakage_Current_THD_pct.toFixed(2)} %`;
    document.getElementById('slider_THD').value = currentSensors.Leakage_Current_THD_pct;
  }

  // Check hardware interlock first (Conventional protection)
  const safety = checkSafetyInterlock(currentSensors);
  
  // Run AI Inference
  const aiResult = predictTinyML(currentSensors);
  const classes = TINYML_MODEL_DATA.classes;
  const predClassInfo = classes[aiResult.predictedClass];

  // Update probabilities breakdown bars
  for (let c = 0; c < 4; c++) {
    const pPct = (aiResult.probabilities[c] * 100).toFixed(1);
    const pText = document.getElementById(`prob_${c}_text`);
    const pBar = document.getElementById(`prob_${c}_bar`);
    if (pText) pText.textContent = `${pPct}%`;
    if (pBar) pBar.style.width = `${pPct}%`;
  }

  // References to alert UI elements
  const banner = document.getElementById('alertBanner');
  const beacon = document.getElementById('alarmBeacon');
  const beaconIcon = document.getElementById('alarmIcon');
  const badge = document.getElementById('alertSeverityBadge');
  const headline = document.getElementById('alertHeadline');
  const subtext = document.getElementById('alertSubtext');
  const confText = document.getElementById('primaryConfidenceText');
  const tripInd = document.getElementById('safetyTripIndicator');
  const resCard = document.getElementById('aiResultCard');
  const predName = document.getElementById('predictedClassName');
  const predLabel = document.getElementById('predictedClassLabel');
  const predBadge = document.getElementById('predictedSeverityBadge');

  const nowTime = new Date().toLocaleTimeString();
  document.getElementById('alertTimestamp').textContent = nowTime;

  // Handle Safety Trip Override
  if (safety.tripped) {
    banner.className = "rounded-2xl p-4 md:p-5 border transition-all duration-300 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 bg-pink-950/40 border-pink-500 text-pink-200 alarm-pulse";
    beacon.className = "w-12 h-12 rounded-xl flex items-center justify-center bg-pink-500/20 text-pink-400 border border-pink-500 alarm-flash flex-shrink-0";
    beaconIcon.setAttribute('data-lucide', 'alert-octagon');
    badge.className = "px-2.5 py-0.5 text-xs font-bold uppercase tracking-wider rounded-md bg-pink-500/30 text-pink-200 font-mono";
    badge.textContent = "SAFETY TRIP TRIGGERED";
    headline.textContent = "HARDWARE SAFETY INTERLOCK BREACHED";
    subtext.textContent = safety.reason;
    confText.textContent = "100.0%";
    tripInd.innerHTML = '<span class="w-2 h-2 rounded-full bg-pink-500 animate-ping"></span><span class="text-pink-300 font-bold">TRIPPED (ISOLATED)</span>';
    
    predName.textContent = "HARDWARE INTERLOCK TRIP";
    predLabel.textContent = "Conventional Safety Circuit Override";
    predBadge.className = "inline-block px-2.5 py-1 text-xs font-extrabold rounded-lg bg-pink-500/30 text-pink-300 border border-pink-500";
    predBadge.textContent = "EMERGENCY TRIP";

    if (lastClass !== 'TRIP') {
      playAlarmTone('TRIP');
      addIncidentLog('Hardware Safety Trip', 'EMERGENCY', '100%', 'Circuit Breaker Open');
      lastClass = 'TRIP';
    }
    lucide.createIcons();
    return;
  }

  // Otherwise, Normal TinyML Prediction State
  tripInd.innerHTML = '<span class="w-2 h-2 rounded-full bg-emerald-400"></span><span>Interlock: ARMED</span>';
  confText.textContent = `${(aiResult.confidence * 100).toFixed(1)}%`;
  predName.textContent = predClassInfo.name;
  predLabel.textContent = predClassInfo.label;

  // Render according to AI Class
  if (aiResult.predictedClass === 0) {
    // HEALTHY
    banner.className = "rounded-2xl p-4 md:p-5 border transition-all duration-300 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 bg-emerald-950/30 border-emerald-500/40 text-emerald-200";
    beacon.className = "w-12 h-12 rounded-xl flex items-center justify-center bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 flex-shrink-0";
    beaconIcon.setAttribute('data-lucide', 'shield-check');
    badge.className = "px-2.5 py-0.5 text-xs font-bold uppercase tracking-wider rounded-md bg-emerald-500/20 text-emerald-300 font-mono";
    badge.textContent = "NORMAL GRID STATUS";
    headline.textContent = "All Parameters Nominal - System Healthy";
    subtext.textContent = "Leakage current, partial discharges, and harmonics are within baseline IEEE specifications.";

    predBadge.className = "inline-block px-2.5 py-1 text-xs font-extrabold rounded-lg bg-emerald-500/20 text-emerald-400 border border-emerald-500/40";
    predBadge.textContent = "NORMAL";

  } else if (aiResult.predictedClass === 1) {
    // LIGHT FAULT (ADVISORY)
    banner.className = "rounded-2xl p-4 md:p-5 border transition-all duration-300 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 bg-yellow-950/30 border-yellow-500/50 text-yellow-200";
    beacon.className = "w-12 h-12 rounded-xl flex items-center justify-center bg-yellow-500/20 text-yellow-400 border border-yellow-500/40 flex-shrink-0";
    beaconIcon.setAttribute('data-lucide', 'alert-triangle');
    badge.className = "px-2.5 py-0.5 text-xs font-bold uppercase tracking-wider rounded-md bg-yellow-500/20 text-yellow-300 font-mono";
    badge.textContent = "ADVISORY: LIGHT POLLUTION";
    headline.textContent = "Early-Stage Insulator Pollution Detected";
    subtext.textContent = "Slight elevation in surface contamination index. Schedule routine maintenance wash.";

    predBadge.className = "inline-block px-2.5 py-1 text-xs font-extrabold rounded-lg bg-yellow-500/20 text-yellow-300 border border-yellow-500/40";
    predBadge.textContent = "LOW / ADVISORY";

  } else if (aiResult.predictedClass === 2) {
    // MODERATE FAULT (WARNING)
    banner.className = "rounded-2xl p-4 md:p-5 border transition-all duration-300 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 bg-orange-950/40 border-orange-500/60 text-orange-200";
    beacon.className = "w-12 h-12 rounded-xl flex items-center justify-center bg-orange-500/20 text-orange-400 border border-orange-500/50 flex-shrink-0";
    beaconIcon.setAttribute('data-lucide', 'alert-circle');
    badge.className = "px-2.5 py-0.5 text-xs font-bold uppercase tracking-wider rounded-md bg-orange-500/20 text-orange-300 font-mono";
    badge.textContent = "WARNING: MODERATE CONTAMINATION";
    headline.textContent = "Noticeable Insulation Degradation Active";
    subtext.textContent = "Harmonic leakage current and dry band arcing risk elevated. Prepare line inspection.";

    predBadge.className = "inline-block px-2.5 py-1 text-xs font-extrabold rounded-lg bg-orange-500/20 text-orange-300 border border-orange-500/40";
    predBadge.textContent = "MEDIUM / WARNING";

  } else if (aiResult.predictedClass === 3) {
    // HEAVY FAULT (CRITICAL FLASHOVER RISK)
    banner.className = "rounded-2xl p-4 md:p-5 border transition-all duration-300 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 bg-rose-950/50 border-rose-500 text-rose-200 alarm-pulse";
    beacon.className = "w-12 h-12 rounded-xl flex items-center justify-center bg-rose-500/30 text-rose-400 border border-rose-500 alarm-flash flex-shrink-0";
    beaconIcon.setAttribute('data-lucide', 'flame');
    badge.className = "px-2.5 py-0.5 text-xs font-bold uppercase tracking-wider rounded-md bg-rose-500/30 text-rose-200 font-mono";
    badge.textContent = "CRITICAL: FLASHOVER HAZARD";
    headline.textContent = "Severe Fault / Wet Flashover Imminent!";
    subtext.textContent = "Excessive leakage current & severe partial discharge counts detected. Immediate line protection recommended.";

    predBadge.className = "inline-block px-2.5 py-1 text-xs font-extrabold rounded-lg bg-rose-500/30 text-rose-200 border border-rose-500";
    predBadge.textContent = "CRITICAL HAZARD";
  }

  // Play audio sound if state changed to a non-zero fault
  if (lastClass !== aiResult.predictedClass) {
    if (aiResult.predictedClass > 0) {
      playAlarmTone(predClassInfo.alert_level);
      addIncidentLog(predClassInfo.label, predClassInfo.severity, `${(aiResult.confidence * 100).toFixed(1)}%`, aiResult.predictedClass === 3 ? "Alert Dispatched" : "Logged");
    }
    lastClass = aiResult.predictedClass;
  }

  lucide.createIcons();
}

// Add incident to audit log
function addIncidentLog(condition, severity, confidence, action) {
  const timestamp = new Date().toLocaleTimeString();
  const entry = {
    timestamp: timestamp,
    condition: condition,
    severity: severity,
    confidence: confidence,
    leakage: currentSensors.Leakage_Current_mA.toFixed(2),
    pd: Math.round(currentSensors.Partial_Discharge_Count),
    action: action
  };

  incidentLogs.unshift(entry);
  if (incidentLogs.length > 50) incidentLogs.pop(); // keep last 50

  renderIncidentTable();
}

function renderIncidentTable() {
  const tbody = document.getElementById('incidentLogBody');
  if (!tbody) return;

  if (incidentLogs.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" class="p-4 text-center text-slate-500 font-mono">No incident alerts recorded yet. System running smoothly.</td></tr>`;
    return;
  }

  tbody.innerHTML = incidentLogs.map(log => {
    let colorClass = "text-emerald-400";
    if (log.severity.includes('Critical') || log.severity === 'EMERGENCY') colorClass = "text-rose-400 font-bold";
    else if (log.severity.includes('Medium')) colorClass = "text-orange-400";
    else if (log.severity.includes('Low')) colorClass = "text-yellow-400";

    return `
      <tr class="hover:bg-slate-800/40 transition-colors">
        <td class="p-3 text-slate-400">${log.timestamp}</td>
        <td class="p-3 font-semibold text-white">${log.condition}</td>
        <td class="p-3 ${colorClass}">${log.severity}</td>
        <td class="p-3 text-cyan-400">${log.confidence}</td>
        <td class="p-3">${log.leakage} mA</td>
        <td class="p-3">${log.pd}</td>
        <td class="p-3 text-slate-400">${log.action}</td>
      </tr>
    `;
  }).join('');
}

// Quick Load Preset from Research Dataset
function loadPreset(presetIdx) {
  if (presetIdx === 'trip') {
    currentSensors = {
      Voltage_kV: 11.2,
      Leakage_Current_mA: 32.5, // > 25 mA trip threshold
      Temperature_C: 48.0,
      Relative_Humidity_pct: 75.0,
      Pollution_Index: 0.85,
      Leakage_Current_THD_pct: 22.5,
      Phase_Angle_deg: 18.0,
      Partial_Discharge_Count: 165.0
    };
  } else {
    const preset = TINYML_MODEL_DATA.sample_presets[presetIdx];
    if (preset) {
      currentSensors = { ...preset.features };
    }
  }
  updateDashboard();
}

function resetToBaseline() {
  loadPreset(0);
}

// Telemetry Streamer Logic
function toggleStream() {
  isStreaming = !isStreaming;
  const btn = document.getElementById('streamToggleBtn');
  const btnText = document.getElementById('streamBtnText');
  const icon = document.getElementById('streamIcon');

  if (isStreaming) {
    btn.className = "flex-1 sm:flex-initial px-4 py-2 rounded-xl text-xs font-bold bg-amber-600 hover:bg-amber-500 text-white flex items-center justify-center gap-2 shadow-lg shadow-amber-600/20 transition-all";
    btnText.textContent = "Pause Stream";
    icon.setAttribute('data-lucide', 'pause');

    initAudio(); // user interaction allows audio context

    streamInterval = setInterval(() => {
      // Pick next sample preset with slight realistic sensor noise
      const presets = TINYML_MODEL_DATA.sample_presets;
      const base = presets[streamIndex % presets.length].features;
      streamIndex++;

      // Inject small realistic fluctuation (+- 2%)
      const noisySensors = {};
      for (const [k, v] of Object.entries(base)) {
        const noise = (Math.random() - 0.5) * 0.04 * v;
        noisySensors[k] = Math.max(0, v + noise);
      }
      currentSensors = noisySensors;
      updateDashboard();
    }, 2000);
  } else {
    clearInterval(streamInterval);
    btn.className = "flex-1 sm:flex-initial px-4 py-2 rounded-xl text-xs font-bold bg-cyan-600 hover:bg-cyan-500 text-white flex items-center justify-center gap-2 shadow-lg shadow-cyan-600/20 transition-all";
    btnText.textContent = "Start Stream";
    icon.setAttribute('data-lucide', 'play');
  }
  lucide.createIcons();
}

function nextStreamSample() {
  const presets = TINYML_MODEL_DATA.sample_presets;
  currentSensors = { ...presets[streamIndex % presets.length].features };
  streamIndex++;
  updateDashboard();
}

// Export Incident Log to CSV
function exportLogsToCsv() {
  if (incidentLogs.length === 0) {
    alert("No incident logs available to export.");
    return;
  }

  const headers = ["Timestamp", "Condition", "Severity", "Confidence", "Leakage_Current_mA", "Partial_Discharge", "Action"];
  const rows = incidentLogs.map(l => [
    `"${l.timestamp}"`,
    `"${l.condition}"`,
    `"${l.severity}"`,
    `"${l.confidence}"`,
    l.leakage,
    l.pd,
    `"${l.action}"`
  ]);

  const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map(e => e.join(","))].join("\n");
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", `auragrid_fault_incidents_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// Setup Event Listeners on DOM Load
document.addEventListener('DOMContentLoaded', () => {
  lucide.createIcons();

  // Initial dashboard render
  updateDashboard();
  renderIncidentTable();

  // Slider Listeners
  const sLeakage = document.getElementById('slider_Leakage');
  if (sLeakage) {
    sLeakage.addEventListener('input', (e) => {
      currentSensors.Leakage_Current_mA = parseFloat(e.target.value);
      updateDashboard();
    });
  }

  const sPD = document.getElementById('slider_PD');
  if (sPD) {
    sPD.addEventListener('input', (e) => {
      currentSensors.Partial_Discharge_Count = parseFloat(e.target.value);
      updateDashboard();
    });
  }

  const sPollution = document.getElementById('slider_Pollution');
  if (sPollution) {
    sPollution.addEventListener('input', (e) => {
      currentSensors.Pollution_Index = parseFloat(e.target.value);
      updateDashboard();
    });
  }

  const sTHD = document.getElementById('slider_THD');
  if (sTHD) {
    sTHD.addEventListener('input', (e) => {
      currentSensors.Leakage_Current_THD_pct = parseFloat(e.target.value);
      updateDashboard();
    });
  }

  // Buttons
  document.getElementById('streamToggleBtn').addEventListener('click', toggleStream);
  document.getElementById('streamNextBtn').addEventListener('click', nextStreamSample);
  document.getElementById('exportCsvBtn').addEventListener('click', exportLogsToCsv);
  document.getElementById('clearLogBtn').addEventListener('click', () => {
    incidentLogs = [];
    renderIncidentTable();
  });

  // Audio Toggle
  document.getElementById('toggleAudioBtn').addEventListener('click', () => {
    audioEnabled = !audioEnabled;
    const icon = document.getElementById('audioIcon');
    const text = document.getElementById('audioText');
    if (audioEnabled) {
      initAudio();
      icon.setAttribute('data-lucide', 'volume-2');
      icon.className = 'w-4 h-4 text-emerald-400';
      text.textContent = 'Siren: Enabled';
    } else {
      icon.setAttribute('data-lucide', 'volume-x');
      icon.className = 'w-4 h-4 text-slate-500';
      text.textContent = 'Siren: Muted';
    }
    lucide.createIcons();
  });

  // Test Siren Audio
  document.getElementById('testAudioBtn').addEventListener('click', () => {
    initAudio();
    playAlarmTone('CRITICAL');
  });

  // Modal handlers
  document.getElementById('openModelModalBtn').addEventListener('click', () => {
    document.getElementById('modelModal').classList.remove('hidden');
  });
  document.getElementById('closeModelModalBtn').addEventListener('click', () => {
    document.getElementById('modelModal').classList.add('hidden');
  });
});
