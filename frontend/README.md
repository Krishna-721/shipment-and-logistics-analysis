# Safiri PortPulse Frontend

A vanilla HTML/CSS/JavaScript operations dashboard for port congestion and delay prediction.

## Quick Start

### 1. Start the FastAPI Backend

```bash
cd d:\Aiqon-takehome\safiri-port-pulse
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8765
```

### 2. Serve the Frontend

```bash
cd d:\Aiqon-takehome\safiri-port-pulse
python -m http.server 8000 --directory frontend
```

### 3. Open in Browser

Navigate to: **http://localhost:8000/**

## Features

### Status Indicator
- Green "API ONLINE" when backend is healthy
- Red "API OFFLINE" when backend is unavailable
- Automatic health check on page load

### Input Form
- 27 validated fields organized into 9 semantic groups
- Two example data buttons (Low-Risk, High-Risk)
- Client-side validation for `available_berths ≤ total_berths`
- Auto-calculated weekend status from day of week
- Storm flag checkbox
- Vessel type and cargo type dropdowns

### Results Dashboard
- **Metrics:** Congestion probability, predicted delay, risk level
- **Visual Probability Bar:** Gradient from green → amber → red
- **Reason:** Human-readable explanation from backend
- **Escalation Triggers:** Conditional warning section
- **SHAP Factors:** Risk-increasing and protective factors with horizontal bars
- **Actions:** Numbered list of operational recommendations
- **Human-in-the-Loop Callout:** Explicit decision support framing

### Error Handling
- 422: Field-level validation errors displayed inline
- 500/503: Dismissible error banner
- Network errors: Clear messaging about API availability

### Responsive Design
- Desktop: Two-column grid (inputs left, results right)
- Tablet: Adjusted layout
- Mobile: Single column stack

## Architecture

### Files
- `index.html` — Semantic HTML structure
- `style.css` — Pure CSS with custom properties
- `app.js` — Vanilla JavaScript, no frameworks

### API Contract
The frontend sends exactly 27 fields to `POST /predict`:
- Berth capacity (3 fields)
- Vessel traffic (3 fields)
- Arrivals (4 fields)
- Queue & waiting (2 fields)
- Equipment (3 fields)
- Labour & weather (3 fields)
- Temporal (3 fields)
- Historical context (2 fields)
- Port profile (4 fields: traffic_pressure, traffic_weather_interaction, vessel_type, cargo_type)

**Five server-derived fields are never sent:**
- capacity_pressure
- queue_pressure
- equipment_pressure
- weather_pressure
- queue_capacity_interaction

These are computed internally by the backend using the exact training generator formulas.

### CORS
The backend is configured to accept requests from:
- http://localhost:8000
- http://127.0.0.1:8000
- http://localhost:5500 (VS Code Live Server)
- http://127.0.0.1:5500

## Testing

### Automated Tests
All automated integration tests pass:
- Health endpoint connectivity
- CORS headers
- Moderate/low/high-risk examples
- 422 validation error handling
- Derived field rejection
- Response structure verification

### Manual Testing Checklist
1. ✓ Open http://localhost:8000/
2. ✓ Verify "API ONLINE" status
3. ✓ Default moderate-risk example pre-loaded
4. ✓ Submit form → prediction renders
5. ✓ Metrics display correctly
6. ✓ SHAP factors show human-readable labels
7. ✓ Actions list displays
8. ✓ Human decision callout visible
9. ✓ Load low-risk example → submit
10. ✓ Load high-risk example → submit
11. ✓ Responsive layout on narrow window

## Design Language

**Dark maritime operations terminal aesthetic:**
- Near-black background (#0f1117)
- Subtle gradients and borders
- System font stack (no web fonts)
- Risk colors: Green (LOW), Amber (MEDIUM), Red (HIGH)
- Accent blue (#4a90d9) for interactive elements
- Professional operations control interface, not consumer web

## Browser Compatibility

Tested in modern browsers supporting:
- CSS Grid
- CSS Custom Properties
- Fetch API
- ES6+ JavaScript

No IE11 support required.
