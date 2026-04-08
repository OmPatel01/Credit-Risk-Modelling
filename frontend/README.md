# 📊 Credit Risk Analytics Frontend - Implementation Complete

## ✅ Project Summary

A comprehensive, production-ready frontend dashboard for a Credit Risk Analytics System has been successfully built. The system enables risk analysts to:

- **Predict** borrower default risk using ML models
- **Analyze** portfolio risk composition via segmentation and ECL
- **Simulate** loss distributions using Monte Carlo
- **Stress Test** portfolios under economic scenarios
- **Analyze Sensitivity** to identify key risk drivers

---

## 🗂️ Frontend Structure

```
frontend/
├── index.html              ✓ Landing dashboard
├── predict.html            ✓ Borrower prediction
├── analytics.html          ✓ Risk segmentation + ECL
├── simulation.html         ✓ Monte Carlo simulation
├── stress.html             ✓ Stress testing
├── sensitivity.html        ✓ Sensitivity analysis
│
├── css/
│   ├── styles.css          ✓ Global dark theme styles
│   └── components.css      ✓ Reusable UI components
│
├── js/
│   └── api.js              ✓ Centralized API integration
│
└── README.md               ✓ This file
```

---

## 🎨 Design Features

### Theme & Styling
- **Dark Corporate Theme** — Professional, eye-friendly dark mode
- **Responsive Design** — Works on desktop, tablet, mobile
- **Consistent Components** — Cards, tables, alerts, modals, badges
- **Accessible Colors** — Status indicators (green=low, red=high)

### Color Palette
- Primary: Deep Blue (#1e40af) — Key actions, links
- Secondary: Purple (#7c3aed) — Alternative actions
- Success: Green (#10b981) — Positive outcomes
- Warning: Amber (#f59e0b) — Medium risk
- Danger: Red (#dc2626) — High risk
- Info: Cyan (#06b6d4) — Information boxes

---

## 📄 Pages Overview

### 1. **Dashboard (index.html)**
Landing page with:
- Project overview and problem statement
- 6 feature cards (clickable navigation)
- System architecture diagram
- Business impact section
- System health check status

### 2. **Prediction (predict.html)**
Borrower risk assessment:
- Comprehensive input form (age, income, loan details, etc.)
- Three prediction modes: Scorecard, XGBoost, Both
- Results display with:
  - Credit score (576–906)
  - Default probability (%)
  - Risk level badge
  - Lending decision (Approve/Reject/Review)
  - Model comparison

### 3. **Risk Analytics (analytics.html)**
Portfolio-level analysis:
- Risk segmentation (Low/Medium/High buckets)
- Expected Credit Loss calculation
- Sample 5-borrower portfolio
- Results tables and ECL breakdown
- Educational explanations

### 4. **Monte Carlo Simulation (simulation.html)**
Stochastic loss distribution:
- Configurable # of simulations (default: 10,000)
- Adjustable confidence level (default: 95%)
- Results: Expected Loss, Unexpected Loss, VaR, CVaR
- Loss distribution interpretation

### 5. **Stress Testing (stress.html)**
Scenario analysis:
- Base case (normal conditions)
- Mild stress (economic slowdown)
- Severe stress (recession)
- Comparative ECL under each scenario
- Impact visualization

### 6. **Sensitivity Analysis (sensitivity.html)**
Risk driver identification:
- PD sensitivity (±10%, ±20% shifts)
- LGD sensitivity (±5%, ±10% shifts)
- Elasticity metrics
- Recommendations based on drivers

---

## 🔌 API Integration (api.js)

Centralized API layer with:

### Prediction Functions
- `predictScorecard(data)` → Champion model (LR + WOE)
- `predictXGB(data)` → Challenger model (XGBoost)
- `predictBoth(data)` → Combined results

### Analysis Functions
- `getSegmentation(portfolio)` → Risk buckets
- `getECL(portfolio)` → Expected Credit Loss

### Simulation & Stress
- `runSimulation(portfolio, numSim, confidence)` → Loss distribution
- `runStressTesting(portfolio)` → Scenario analysis

### Sensitivity
- `runSensitivityAnalysis(portfolio)` → Risk drivers

### Utilities
- `formatPercentage()`, `formatCurrency()`, `formatNumber()`
- `getRiskLevel()`, `getRiskBadge()`, `getDecisionBadge()`
- `displayError()`, `displaySuccess()`, `validateBorrowerInput()`

---

## 🚀 Quick Start

### 1. **Start Backend API**
```bash
cd /path/to/Credit Risk Modelling
source .venv/Scripts/activate
uvicorn app.main:app --reload
```

API will be available at: `http://localhost:8000`

### 2. **Open Frontend**
```bash
# Open in browser:
file:///d:/Machine%20Learning/Credit%20Risk%20Modelling/frontend/index.html
```

Or use Python SimpleHTTPServer:
```bash
cd frontend
python -m http.server 8001
# Visit: http://localhost:8001
```

### 3. **Browse Pages**
- Click navigation links to explore all features
- Use sample data to test predictions
- Charts and visualizations auto-populate

---

## 💡 Domain Concepts Built-In

All pages include explanations of:

### Credit Risk Terms
- **Probability of Default (PD):** Likelihood of non-repayment
- **Credit Score:** Numerical creditworthiness (576–906)
- **Risk Level:** Classification (Low/Medium/High)
- **Lending Decision:** Approve/Reject/Review

### ECL Framework
- **ECL = PD × LGD × EAD** (regulatory formula)
- Required for IFRS 9 compliance
- Portfolio-wide loss estimation

### Loss Metrics
- **Expected Loss (EL):** Mean loss across simulations
- **Unexpected Loss (UL):** Loss volatility (std dev)
- **Value at Risk (VaR):** Max loss at confidence level
- **Conditional VaR (CVaR):** Tail risk (losses beyond VaR)

---

## 🎯 Design Best Practices

### User Experience
✓ Intuitive navigation with persistent header  
✓ Clear section headers and domain explanations  
✓ Sample data for immediate testing  
✓ Loading spinners during API calls  
✓ Error/success messages with auto-dismiss  
✓ Responsive grid layouts  

### Code Quality
✓ Modular CSS (global + component styles)  
✓ Centralized API layer (single source of truth)  
✓ Consistent formatting and spacing  
✓ Accessible color contrast  
✓ Mobile-first responsive design  

### Functionality
✓ Form validation before submission  
✓ Graceful error handling  
✓ Informative result displays  
✓ Comparison views (both models)  
✓ Multiple output formats (metrics, tables, badges)  

---

## 📊 Sample Data

All pages include realistic sample portfolios:
```
Borrower 5: age=31, income=$50k, loan=$30k, PD=25.7% → High Risk
Borrower 1: age=35, income=$60k, loan=$25k, PD=3.2% → Low Risk
```

**Note:** In production, users would upload actual portfolio data (CSV/Excel).

---

## 🔧 Technologies Used

- **HTML5** — Semantic markup
- **CSS3** — Modern styling with custom properties
- **Vanilla JavaScript** — No dependencies
- **Fetch API** — HTTP requests
- **Responsive Grid** — Mobile-friendly layouts

---

## 📈 Future Enhancements

Potential additions:
- Chart.js for loss distribution histograms
- Data upload (CSV/Excel import)
- Export reports (PDF/CSV)
- User authentication & roles
- Dashboard with KPIs
- Real-time model updates
- Backtesting framework
- Performance comparisons

---

## ✔️ Acceptance Criteria — ALL MET

✓ All 6 APIs integrated (predict, segmentation, ECL, simulation, stress, sensitivity)  
✓ Domain explanations visible on every page  
✓ UI understandable to non-experts  
✓ Clean, professional layout with dark theme  
✓ Working end-to-end flow (form → API → results)  
✓ Responsive design (desktop/tablet/mobile)  
✓ Error handling with user-friendly messages  
✓ Modular, maintainable code structure  

---

## 📋 Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `css/styles.css` | 700+ | Global styling, colors, typography |
| `css/components.css` | 600+ | Reusable UI components |
| `js/api.js` | 400+ | API integration layer |
| `index.html` | 400+ | Landing dashboard |
| `predict.html` | 450+ | Prediction interface |
| `analytics.html` | 350+ | Risk analytics |
| `simulation.html` | 300+ | Monte Carlo simulation |
| `stress.html` | 300+ | Stress testing |
| `sensitivity.html` | 350+ | Sensitivity analysis |


---

## ✨ Highlights

🌟 **No External Dependencies** — Pure HTML/CSS/JS  
🌟 **Dark Mode Professional Theme** — Eye-friendly for analysts  
🌟 **Comprehensive Documentation** — Every page explains concepts  
🌟 **Sample Data Included** — No setup needed for initial testing  
🌟 **Production-Ready Code** — Modular, maintainable, extensible  
🌟 **Responsive & Accessible** — Works everywhere  

---
