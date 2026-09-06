// ============================================================================
// SAFIRI PORTPULSE — app.js
// 3-step assessment workflow, scenario presets, /predict API integration.
// Plain vanilla JS — no frameworks, no dependencies.
// ============================================================================

const API_BASE = "http://127.0.0.1:8000";

// ── Application state ────────────────────────────────────────────────────────
// Holds the current form values across all 3 steps.
// Updated on every field change; never cleared by step navigation.
const state = {
  // Step 1 — Current Operations
  total_berths: null,
  available_berths: null,
  berth_utilization: null,
  vessels_currently_in_port: null,
  vessels_anchored: null,
  vessels_approaching: null,
  queue_length: null,
  avg_waiting_time_hours: null,

  // Step 2 — Traffic & Resources
  arrivals_last_1h: null,
  arrivals_last_6h: null,
  arrivals_last_24h: null,
  arrival_rate: null,
  cranes_operational: null,
  crane_utilization: null,
  equipment_failure_count: null,
  labor_availability_pct: null,

  // Step 3 — Conditions & Context
  weather_severity: null,
  storm_flag: 0,
  hour: null,
  day_of_week: null,
  is_weekend: 0,
  historical_avg_wait_time: null,
  historical_congestion_rate: null,
  traffic_pressure: null,
  traffic_weather_interaction: null,
  vessel_type: "",
  cargo_type: "",
};

// Current active step (1–3)
let currentStep = 1;

// Prevent double-submission
let isSubmitting = false;

// ── Scenario presets ─────────────────────────────────────────────────────────
// Values drawn from smoke-test rows and plausible mid-range data.
// All values satisfy the API validation ranges.
const SCENARIOS = {
  normal: {
    label: "NORMAL",
    total_berths: 11,
    available_berths: 10,
    berth_utilization: 0.09,
    vessels_currently_in_port: 2,
    vessels_anchored: 1,
    vessels_approaching: 3,
    queue_length: 0.82,
    avg_waiting_time_hours: 0.38,
    arrivals_last_1h: 1,
    arrivals_last_6h: 12,
    arrivals_last_24h: 77,
    arrival_rate: 2.98,
    cranes_operational: 14,
    crane_utilization: 0.62,
    equipment_failure_count: 0,
    labor_availability_pct: 1.0,
    weather_severity: 0.0,
    storm_flag: 0,
    hour: 5,
    day_of_week: 4,
    is_weekend: 0,
    historical_avg_wait_time: 1.18,
    historical_congestion_rate: 0.22,
    traffic_pressure: 0.72,
    traffic_weather_interaction: 0.0,
    vessel_type: "bulk",
    cargo_type: "dry_bulk",
  },

  busy: {
    label: "BUSY PORT",
    total_berths: 15,
    available_berths: 3,
    berth_utilization: 0.80,
    vessels_currently_in_port: 18,
    vessels_anchored: 5,
    vessels_approaching: 8,
    queue_length: 4.5,
    avg_waiting_time_hours: 2.8,
    arrivals_last_1h: 2,
    arrivals_last_6h: 15,
    arrivals_last_24h: 85,
    arrival_rate: 3.2,
    cranes_operational: 16,
    crane_utilization: 0.75,
    equipment_failure_count: 1,
    labor_availability_pct: 0.92,
    weather_severity: 0.15,
    storm_flag: 0,
    hour: 14,
    day_of_week: 2,
    is_weekend: 0,
    historical_avg_wait_time: 1.8,
    historical_congestion_rate: 0.28,
    traffic_pressure: 0.85,
    traffic_weather_interaction: 0.13,
    vessel_type: "container",
    cargo_type: "containerized",
  },

  critical: {
    label: "CRITICAL",
    total_berths: 6,
    available_berths: 0,
    berth_utilization: 1.0,
    vessels_currently_in_port: 21,
    vessels_anchored: 13,
    vessels_approaching: 2,
    queue_length: 18.89,
    avg_waiting_time_hours: 18.02,
    arrivals_last_1h: 2,
    arrivals_last_6h: 19,
    arrivals_last_24h: 40,
    arrival_rate: 1.30,
    cranes_operational: 8,
    crane_utilization: 0.58,
    equipment_failure_count: 0,
    labor_availability_pct: 0.96,
    weather_severity: 0.30,
    storm_flag: 0,
    hour: 22,
    day_of_week: 4,
    is_weekend: 0,
    historical_avg_wait_time: 1.51,
    historical_congestion_rate: 0.11,
    traffic_pressure: 0.75,
    traffic_weather_interaction: 0.23,
    vessel_type: "container",
    cargo_type: "containerized",
  },
};

// ── Field definitions per step (used for sync & validation) ─────────────────
const STEP_FIELDS = {
  1: [
    "total_berths", "available_berths", "berth_utilization",
    "vessels_currently_in_port", "vessels_anchored", "vessels_approaching",
    "queue_length", "avg_waiting_time_hours",
  ],
  2: [
    "arrivals_last_1h", "arrivals_last_6h", "arrivals_last_24h", "arrival_rate",
    "cranes_operational", "crane_utilization", "equipment_failure_count",
    "labor_availability_pct",
  ],
  3: [
    "weather_severity", "hour", "day_of_week",
    "historical_avg_wait_time", "historical_congestion_rate",
    "traffic_pressure", "traffic_weather_interaction",
    "vessel_type", "cargo_type",
  ],
};

// Integer fields (parsed with parseInt, not parseFloat)
const INTEGER_FIELDS = new Set([
  "total_berths", "available_berths",
  "vessels_currently_in_port", "vessels_anchored", "vessels_approaching",
  "arrivals_last_1h", "arrivals_last_6h", "arrivals_last_24h",
  "cranes_operational", "equipment_failure_count",
  "hour", "day_of_week",
]);

// ── Initialization ───────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  checkApiHealth();
  // Load the busy scenario as a sensible default for the snapshot overview
  applyScenarioToState(SCENARIOS.busy);
  updateSnapshot();
  wireInputListeners();
  populateFormFromState();
  wireButtonListeners();
});

// ── Button wiring (all event listeners here, no inline onclick in HTML) ──────
function wireButtonListeners() {
  // Home screen
  const btnBegin = document.getElementById("btnBeginAssessment");
  if (btnBegin) btnBegin.addEventListener("click", () => setStep(1));

  document.getElementById("scenarioNormal")?.addEventListener("click", () => loadScenario("normal"));
  document.getElementById("scenarioBusy")?.addEventListener("click",   () => loadScenario("busy"));
  document.getElementById("scenarioCritical")?.addEventListener("click", () => loadScenario("critical"));

  // Assessment steps
  document.getElementById("step1Back")?.addEventListener("click", () => goHome());
  document.getElementById("step1Next")?.addEventListener("click", () => nextStep(1));
  document.getElementById("step2Back")?.addEventListener("click", () => prevStep(2));
  document.getElementById("step2Next")?.addEventListener("click", () => nextStep(2));
  document.getElementById("step3Back")?.addEventListener("click", () => prevStep(3));
  document.getElementById("btnAssess")?.addEventListener("click", () => submitAssessment());

  // Error bar dismiss
  document.getElementById("errorBarClose")?.addEventListener("click", () => hideError());

  // Results screen
  document.getElementById("btnModifyTop")?.addEventListener("click",    () => modifyConditions());
  document.getElementById("btnNewTop")?.addEventListener("click",       () => resetAssessment());
  document.getElementById("btnModifyBottom")?.addEventListener("click", () => modifyConditions());
  document.getElementById("btnNewBottom")?.addEventListener("click",    () => resetAssessment());
}

// ── Screen management ────────────────────────────────────────────────────────
function showScreen(id) {
  document.querySelectorAll(".screen").forEach((s) => s.classList.remove("active"));
  const el = document.getElementById(id);
  if (el) {
    el.classList.add("active");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
}

function goHome() {
  syncStateFromForm(currentStep);
  showScreen("screenHome");
  updateSnapshot();
}

function modifyConditions() {
  // Return to step 1 with existing values intact
  setStep(1);
}

function resetAssessment() {
  applyScenarioToState(SCENARIOS.busy);
  populateFormFromState();
  updateSnapshot();
  clearAllErrors();
  hideError();
  setStep(1);
}

// ── Step management ──────────────────────────────────────────────────────────
function setStep(n) {
  currentStep = n;
  showScreen("screenAssessment");
  hideError();
  clearAllErrors();

  // Show/hide step panels
  for (let i = 1; i <= 3; i++) {
    const panel = document.getElementById(`stepPanel${i}`);
    const nav   = document.getElementById(`stepNav${i}`);
    if (!panel || !nav) continue;

    panel.classList.toggle("active", i === n);
    nav.classList.remove("active", "done");
    if (i === n)      nav.classList.add("active");
    else if (i < n)   nav.classList.add("done");
  }

  // Re-populate the visible step's inputs from state
  populateStepFromState(n);
}

function nextStep(fromStep) {
  syncStateFromForm(fromStep);
  if (!validateStep(fromStep)) return;
  setStep(fromStep + 1);
}

function prevStep(fromStep) {
  syncStateFromForm(fromStep);
  setStep(fromStep - 1);
}

// ── State ↔ form synchronization ────────────────────────────────────────────

// Read the active step's DOM inputs into state
function syncStateFromForm(step) {
  const fields = STEP_FIELDS[step] || [];
  fields.forEach((key) => {
    const el = document.getElementById(key);
    if (!el) return;

    if (el.type === "checkbox") {
      state[key] = el.checked ? 1 : 0;
    } else if (el.tagName === "SELECT") {
      state[key] = el.value;
    } else {
      const raw = el.value.trim();
      if (raw === "") {
        state[key] = null;
      } else {
        state[key] = INTEGER_FIELDS.has(key) ? parseInt(raw, 10) : parseFloat(raw);
      }
    }
  });

  // Derived: is_weekend
  if (state.day_of_week !== null) {
    state.is_weekend = state.day_of_week >= 5 ? 1 : 0;
  }

  // storm_flag is on step 3 but handled separately via checkbox
  const stormEl = document.getElementById("storm_flag");
  if (stormEl) state.storm_flag = stormEl.checked ? 1 : 0;
}

// Push state values into the DOM for a given step
function populateStepFromState(step) {
  const fields = STEP_FIELDS[step] || [];
  fields.forEach((key) => {
    const el = document.getElementById(key);
    if (!el) return;

    const val = state[key];
    if (el.tagName === "SELECT") {
      el.value = val !== null && val !== undefined ? val : "";
    } else if (el.type === "checkbox") {
      el.checked = val === 1 || val === true;
    } else {
      el.value = val !== null && val !== undefined ? val : "";
    }
  });

  // storm_flag lives in step 3
  if (step === 3) {
    const stormEl = document.getElementById("storm_flag");
    if (stormEl) stormEl.checked = state.storm_flag === 1;
    updateWeekendDisplay();
  }
}

// Push full state into all form inputs (called on first load / scenario load)
function populateFormFromState() {
  for (let s = 1; s <= 3; s++) {
    populateStepFromState(s);
  }
  updateWeekendDisplay();
}

function applyScenarioToState(scenario) {
  Object.keys(state).forEach((k) => {
    if (scenario[k] !== undefined) state[k] = scenario[k];
  });
  state.is_weekend = state.day_of_week >= 5 ? 1 : 0;
}

// ── Weekend display ──────────────────────────────────────────────────────────
function updateWeekendDisplay() {
  const dow = parseInt(document.getElementById("day_of_week")?.value, 10);
  const isWeekend = !isNaN(dow) && dow >= 5;
  state.is_weekend = isWeekend ? 1 : 0;

  const el = document.getElementById("weekendDisplay");
  if (el) {
    el.textContent = isWeekend ? "Yes" : "No";
  }
}

// ── Snapshot cards (home screen) ─────────────────────────────────────────────
function updateSnapshot() {
  const s = state;

  // Berths
  const snBerths = document.getElementById("snap-berths");
  const snBerthsD = document.getElementById("snap-berths-detail");
  if (snBerths && s.total_berths !== null) {
    snBerths.textContent = `${s.available_berths ?? "—"} / ${s.total_berths}`;
    snBerthsD.textContent = "available / total";
  }

  // Vessels
  const snV = document.getElementById("snap-vessels");
  const snVD = document.getElementById("snap-vessels-detail");
  if (snV && s.vessels_currently_in_port !== null) {
    snV.textContent = `${s.vessels_currently_in_port} · ${s.vessels_anchored} · ${s.vessels_approaching}`;
    snVD.textContent = "in port · anchored · approaching";
  }

  // Queue
  const snQ = document.getElementById("snap-queue");
  const snQD = document.getElementById("snap-queue-detail");
  if (snQ && s.queue_length !== null) {
    snQ.textContent = `${s.queue_length}`;
    snQD.textContent = `vessels · ${s.avg_waiting_time_hours ?? "—"} h avg wait`;
  }

  // Operations
  const snO = document.getElementById("snap-ops");
  const snOD = document.getElementById("snap-ops-detail");
  if (snO && s.crane_utilization !== null) {
    const craneUtilPct = Math.round(s.crane_utilization * 100);
    const labourPct    = Math.round(s.labor_availability_pct * 100);
    snO.textContent = `${craneUtilPct}% · ${labourPct}%`;
    snOD.textContent = "crane util · labour";
  }
}

// ── Input event wiring ───────────────────────────────────────────────────────
function wireInputListeners() {
  // Live sync + clear errors on change
  document.querySelectorAll("input[type='number'], input[type='checkbox'], select").forEach((el) => {
    el.addEventListener("change", () => {
      const step = getFieldStep(el.id);
      if (step) syncStateFromForm(step);
      clearFieldError(el.id);
      if (el.id === "day_of_week") updateWeekendDisplay();
      if (el.id === "total_berths" || el.id === "available_berths") {
        // Immediate cross-field hint
        const total = parseInt(document.getElementById("total_berths")?.value, 10);
        const avail = parseInt(document.getElementById("available_berths")?.value, 10);
        if (!isNaN(total) && !isNaN(avail) && avail > total) {
          showFieldError("available_berths", "Cannot exceed total berths");
        }
      }
    });

    el.addEventListener("input", () => {
      clearFieldError(el.id);
      if (el.id === "day_of_week") updateWeekendDisplay();
    });
  });
}

function getFieldStep(fieldId) {
  for (let s = 1; s <= 3; s++) {
    if (STEP_FIELDS[s].includes(fieldId)) return s;
    if (fieldId === "storm_flag") return 3;
  }
  return null;
}

// ── Validation ───────────────────────────────────────────────────────────────
function validateStep(step) {
  clearAllErrors();
  let ok = true;

  const fields = STEP_FIELDS[step] || [];
  fields.forEach((key) => {
    const val = state[key];
    if (val === null || val === undefined || val === "" ||
        (typeof val === "number" && isNaN(val))) {
      showFieldError(key, "Required");
      ok = false;
    }
  });

  // Step 1 cross-field: available_berths ≤ total_berths
  if (step === 1) {
    const total = state.total_berths;
    const avail = state.available_berths;
    if (total !== null && avail !== null && avail > total) {
      showFieldError("available_berths", "Cannot exceed total berths");
      ok = false;
    }
  }

  if (!ok) {
    showError("Please fill in all required fields before continuing.");
  }

  return ok;
}

// Validate the full 27-field payload before API submission
function validateAll() {
  let ok = true;
  for (let s = 1; s <= 3; s++) {
    const fields = STEP_FIELDS[s] || [];
    fields.forEach((key) => {
      const val = state[key];
      if (val === null || val === undefined || val === "" ||
          (typeof val === "number" && isNaN(val))) {
        ok = false;
      }
    });
  }
  const total = state.total_berths;
  const avail = state.available_berths;
  if (total !== null && avail !== null && avail > total) ok = false;
  return ok;
}

// ── Scenarios ─────────────────────────────────────────────────────────────────
function loadScenario(name) {
  const scenario = SCENARIOS[name];
  if (!scenario) return;
  applyScenarioToState(scenario);
  populateFormFromState();
  updateSnapshot();
  clearAllErrors();
  hideError();
}

// ── API health ────────────────────────────────────────────────────────────────
async function checkApiHealth() {
  const dot   = document.getElementById("apiDot");
  const label = document.getElementById("apiLabel");

  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
    const data = await res.json();
    if (res.ok && data.status === "ok") {
      dot.className   = "api-dot online";
      label.className = "api-label online";
      label.textContent = "API ONLINE";
    } else {
      setApiOffline(dot, label);
    }
  } catch {
    setApiOffline(dot, label);
  }
}

function setApiOffline(dot, label) {
  dot.className     = "api-dot offline";
  label.className   = "api-label offline";
  label.textContent = "API OFFLINE";
}

// ── Collect the 27-field payload ─────────────────────────────────────────────
function collectFormData() {
  // Ensure all steps are synced (user may not have navigated back)
  for (let s = 1; s <= 3; s++) syncStateFromForm(s);

  // Exactly the 27 fields accepted by POST /predict
  // The five server-derived fields are never included here.
  return {
    total_berths:               state.total_berths,
    available_berths:           state.available_berths,
    berth_utilization:          state.berth_utilization,
    vessels_currently_in_port:  state.vessels_currently_in_port,
    vessels_anchored:           state.vessels_anchored,
    vessels_approaching:        state.vessels_approaching,
    arrivals_last_1h:           state.arrivals_last_1h,
    arrivals_last_6h:           state.arrivals_last_6h,
    arrivals_last_24h:          state.arrivals_last_24h,
    arrival_rate:               state.arrival_rate,
    queue_length:               state.queue_length,
    avg_waiting_time_hours:     state.avg_waiting_time_hours,
    cranes_operational:         state.cranes_operational,
    crane_utilization:          state.crane_utilization,
    equipment_failure_count:    state.equipment_failure_count,
    labor_availability_pct:     state.labor_availability_pct,
    weather_severity:           state.weather_severity,
    storm_flag:                 state.storm_flag,
    hour:                       state.hour,
    day_of_week:                state.day_of_week,
    is_weekend:                 state.is_weekend,
    historical_avg_wait_time:   state.historical_avg_wait_time,
    historical_congestion_rate: state.historical_congestion_rate,
    traffic_pressure:           state.traffic_pressure,
    traffic_weather_interaction: state.traffic_weather_interaction,
    vessel_type:                state.vessel_type,
    cargo_type:                 state.cargo_type,
  };
}

// ── Submit assessment ─────────────────────────────────────────────────────────
async function submitAssessment() {
  if (isSubmitting) return;

  // Sync step 3 first
  syncStateFromForm(3);

  // Validate step 3 before sending
  if (!validateStep(3)) return;

  const payload = collectFormData();

  const btn     = document.getElementById("btnAssess");
  const btnText = document.getElementById("assessBtnText");
  const btnArrow = document.getElementById("assessBtnArrow");

  setLoadingState(btn, btnText, btnArrow, true);
  hideError();

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (res.ok) {
      renderAssessment(data);
    } else if (res.status === 422) {
      handle422(data);
    } else if (res.status === 503) {
      showError("Prediction service unavailable. Check that the API server has loaded its models.");
    } else {
      showError(`Unable to generate prediction (HTTP ${res.status}). Check that the API is running.`);
    }
  } catch (err) {
    showError("API unavailable. Make sure the FastAPI server is running on port 8765.");
  } finally {
    setLoadingState(btn, btnText, btnArrow, false);
  }
}

function handle422(data) {
  if (data.detail && Array.isArray(data.detail)) {
    // Show field-level messages where possible
    data.detail.forEach((err) => {
      const loc = err.loc ?? [];
      const field = loc[loc.length - 1];
      if (field && typeof field === "string") {
        showFieldError(field, err.msg ?? "Invalid value");
      }
    });
    showError("Some fields contain invalid values. Please review the highlighted inputs.");
  } else {
    showError("Validation failed. Please check your inputs.");
  }
}

function setLoadingState(btn, textEl, arrowEl, loading) {
  isSubmitting = loading;
  btn.disabled = loading;
  if (loading) {
    btn.classList.add("btn-loading");
    if (textEl)  textEl.textContent  = "ANALYZING";
    if (arrowEl) arrowEl.textContent = "";
  } else {
    btn.classList.remove("btn-loading");
    if (textEl)  textEl.textContent  = "ASSESS PORT RISK";
    if (arrowEl) arrowEl.textContent = "→";
  }
}

// ── Render results ────────────────────────────────────────────────────────────
function renderAssessment(data) {
  renderTimestamp();
  renderRisk(data.congestion_probability, data.risk_level);
  renderDelay(data.predicted_delay_hours);
  renderEscalation(data.escalation_triggers);
  renderReason(data.reason);
  renderFactors(data);
  renderActions(data.actions);
  showScreen("screenResults");
}

function renderTimestamp() {
  const el = document.getElementById("resultsTimestamp");
  if (!el) return;
  const now  = new Date();
  const time = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const date = now.toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" });
  el.textContent = `${date} · ${time}`;
}

function renderRisk(probability, riskLevel) {
  const pct      = Math.round(probability * 100);
  const ringEl   = document.getElementById("ringSvg");
  const pctEl    = document.getElementById("ringPct");
  const badgeEl  = document.getElementById("riskBadge");

  // Ring circumference = 2π × 50 ≈ 314.16
  const circumference = 314.16;
  const offset        = circumference - (probability * circumference);

  // Colour-code the ring stroke by risk level
  const strokeColor = riskLevel === "HIGH"   ? "var(--high)"
                    : riskLevel === "MEDIUM" ? "var(--med)"
                    : "var(--low)";

  if (ringEl) {
    ringEl.style.stroke            = strokeColor;
    // Trigger CSS transition: set offset after a brief delay so the animation fires
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        ringEl.style.strokeDashoffset = String(offset);
      });
    });
  }

  if (pctEl) {
    pctEl.textContent = `${pct}%`;
    pctEl.style.color = strokeColor;
  }

  if (badgeEl) {
    badgeEl.textContent = riskLevel;
    badgeEl.className   = `risk-badge ${riskLevel}`;
  }
}

function renderDelay(hours) {
  const el = document.getElementById("delayValue");
  if (el) el.textContent = hours.toFixed(1);
}

function renderEscalation(triggers) {
  const block = document.getElementById("escalationBlock");
  const list  = document.getElementById("escalationList");
  if (!block || !list) return;

  if (triggers && triggers.length > 0) {
    list.innerHTML = "";
    triggers.forEach((t) => {
      const li = document.createElement("li");
      li.textContent = t;
      list.appendChild(li);
    });
    block.hidden = false;
  } else {
    block.hidden = true;
  }
}

function renderReason(reason) {
  const el = document.getElementById("reasonBlock");
  if (el) el.textContent = reason;
}

function renderFactors(data) {
  const shapAvail    = data.shap_available;
  const riskFactors  = data.risk_increasing_factors || [];
  const protFactors  = data.protective_factors      || [];

  const factorsRow   = document.getElementById("factorsRow");
  const unavail      = document.getElementById("shapUnavailable");

  if (!shapAvail || (riskFactors.length === 0 && protFactors.length === 0)) {
    if (factorsRow) factorsRow.style.display = "none";
    if (unavail)    unavail.hidden = false;
    return;
  }

  if (factorsRow) factorsRow.style.display = "grid";
  if (unavail)    unavail.hidden = true;

  // Maximum absolute SHAP across all factors — used to scale bar widths
  const allAbs = [...riskFactors, ...protFactors]
    .filter((f) => f.shap_value !== null)
    .map((f) => Math.abs(f.shap_value));
  const maxAbs = allAbs.length > 0 ? Math.max(...allAbs) : 1;

  renderFactorList("riskFactorList",    riskFactors, "risk",    maxAbs);
  renderFactorList("protectFactorList", protFactors, "protect", maxAbs);
}

function renderFactorList(containerId, factors, type, maxAbs) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = "";

  if (factors.length === 0) {
    el.innerHTML = `<p style="color:var(--tx-3);font-size:0.85rem;font-style:italic;">None identified.</p>`;
    return;
  }

  factors.forEach((f) => {
    const shapVal  = f.shap_value !== null ? f.shap_value : 0;
    const barWidth = maxAbs > 0 ? (Math.abs(shapVal) / maxAbs) * 100 : 0;
    const sign     = shapVal >= 0 ? "+" : "−";
    const absStr   = Math.abs(shapVal).toFixed(3);

    // Use backend-provided human label; fall back to feature_name if blank
    const displayLabel = (f.label && f.label.trim()) ? f.label : f.feature_name;

    const item = document.createElement("div");
    item.className = "factor-item";

    item.innerHTML = `
      <div class="factor-meta">
        <span class="factor-label">${escHtml(displayLabel)}</span>
        <div class="factor-stats">
          <span class="factor-value">${escHtml(String(f.raw_value))}</span>
          <span class="factor-contrib ${type}">${sign}${absStr}</span>
        </div>
      </div>
      <div class="factor-track">
        <div class="factor-bar ${type}" style="width:0%" data-target="${barWidth.toFixed(1)}%"></div>
      </div>
    `;
    el.appendChild(item);

    // Animate bars after DOM insertion
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        const bar = item.querySelector(".factor-bar");
        if (bar) bar.style.width = bar.dataset.target;
      });
    });
  });
}

function renderActions(actions) {
  const el = document.getElementById("actionList");
  if (!el) return;
  el.innerHTML = "";

  (actions || []).forEach((action) => {
    const li = document.createElement("li");
    li.textContent = action;
    el.appendChild(li);
  });
}

// ── Error display ─────────────────────────────────────────────────────────────
function showError(msg) {
  const bar  = document.getElementById("errorBar");
  const text = document.getElementById("errorBarText");
  if (!bar || !text) return;
  text.textContent = msg;
  bar.hidden = false;
}

function hideError() {
  const bar = document.getElementById("errorBar");
  if (bar) bar.hidden = true;
}

function showFieldError(fieldId, msg) {
  const el  = document.getElementById(fieldId);
  const err = document.getElementById(`err_${fieldId}`);
  if (el)  el.classList.add("invalid");
  if (err) err.textContent = msg;
}

function clearFieldError(fieldId) {
  const el  = document.getElementById(fieldId);
  const err = document.getElementById(`err_${fieldId}`);
  if (el)  el.classList.remove("invalid");
  if (err) err.textContent = "";
}

function clearAllErrors() {
  document.querySelectorAll(".field-err").forEach((e) => { e.textContent = ""; });
  document.querySelectorAll(".invalid").forEach((e) => e.classList.remove("invalid"));
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
