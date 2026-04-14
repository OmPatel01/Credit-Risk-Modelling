# 📌 Credit Risk Modeling System

> A full-stack, end-to-end credit risk intelligence platform built with machine learning, financial engineering, and modern web technologies. Predicts borrower default risk, calculates portfolio losses, and stress-tests under adverse economic scenarios.

**Live Demo:** [Frontend on Vercel](https://credit-risk-modelling-3zjb.onrender.com) · [API on Render](https://credit-risk-modelling-3zjb.onrender.com/docs)

---

## ❗ Problem Statement

Banks approve thousands of loans daily — with no consistent way to decide who is likely to default.

> **Bad loans cost money. Rejected good borrowers cost revenue. And regulators demand proof that both risks are measured.**

Traditional rule-based credit systems fail on three fronts:
- They are **inconsistent** — two analysts reviewing the same borrower may reach different decisions
- They are **blind to portfolio risk** — no way to know total exposure or worst-case losses across all borrowers
- They are **non-compliant** — IFRS 9 requires forward-looking, model-driven Expected Credit Loss (ECL) provisions; Basel III requires stress testing

**This project solves all three.** It builds a complete, production-grade credit risk system — from individual borrower scoring to portfolio loss simulation to regulatory stress testing — using the same methodology used in real banks.

---

## 📋 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Project Overview](#-project-overview)
3. [Business Context](#-business-context)
4. [System Architecture](#-system-architecture)
5. [Dataset & Features](#-dataset--features)
6. [Data Processing & EDA](#-data-processing--eda)
7. [Model Development (PD Model)](#-model-development-pd-model)
8. [Risk Segmentation](#-risk-segmentation)
9. [Expected Credit Loss (ECL)](#-expected-credit-loss-ecl)
10. [Monte Carlo Simulation](#-monte-carlo-simulation)
11. [Sensitivity & Stress Testing](#-sensitivity--stress-testing)
12. [API & Backend (FastAPI)](#-api--backend-fastapi)
13. [Frontend Dashboard](#-frontend-dashboard)
14. [Deployment](#-deployment)
15. [Project Structure](#-project-structure)
16. [How to Run Locally](#-how-to-run-locally)
17. [Results & Metrics](#-results--metrics)
18. [Key Highlights](#-key-highlights)
19. [Future Improvements](#-future-improvements)

---

## 🚀 Project Overview

This project implements a **production-grade credit risk analytics system** that mirrors what you would find in a real bank or fintech company. It goes beyond a simple ML model — it covers the complete lifecycle of credit risk management:

| Capability | What It Does |
|---|---|
| **Default Prediction** | Estimates probability of default (PD) for individual borrowers |
| **Credit Scoring** | Produces interpretable credit scores (576–906) via a WOE scorecard |
| **Risk Segmentation** | Groups borrowers into Low / Medium / High risk buckets |
| **ECL Calculation** | Computes Expected Credit Loss = PD × LGD × EAD (IFRS 9 compliant) |
| **Monte Carlo Simulation** | Estimates full loss distribution; calculates VaR and CVaR |
| **Stress Testing** | Evaluates portfolio under Base / Mild / Severe economic scenarios |
| **Sensitivity Analysis** | Identifies which risk drivers (PD, LGD) most impact losses |
| **What-If Analysis** | Shows how changing one input shifts a borrower's score |

---

## 🏦 Business Context

### The Problem

Financial institutions making lending decisions face significant challenges:

- **No consistent methodology** — decisions depend on individual judgment
- **Poor portfolio visibility** — no aggregate view of risk exposure
- **Regulatory pressure** — IFRS 9 requires forward-looking ECL provisions
- **Tail risk blindness** — expected loss is easy to calculate, but what about worst-case scenarios?

### The Solution

This system provides a **data-driven, standardized framework** for credit risk:

- **Automated scoring** ensures consistent decisions across all borrowers
- **Interpretable credit scores** with clear score-to-decision mapping
- **Portfolio analytics** give a full picture of aggregate risk exposure
- **Regulatory-compliant ECL** calculations following IFRS 9 / Basel III methodology
- **Stress testing and VaR** quantify tail risks that ECL alone cannot capture

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────┐
│              User (Browser / Frontend)           │
│           Vercel — Static HTML/CSS/JS            │
└─────────────────────┬───────────────────────────┘
                      │  HTTP / REST API
┌─────────────────────▼───────────────────────────┐
│              FastAPI Backend                     │
│              Render (Production)                 │
│         /predict  /ecl  /simulate                │
│         /stress-test  /sensitivity               │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────┐
│           Business Input Mapper                  │
│    Translates 10 plain-English fields →          │
│    21 raw dataset columns                        │
└──────────┬──────────────────────┬───────────────┘
           │                      │
┌──────────▼──────────┐  ┌───────▼───────────────┐
│  WOE Scorecard      │  │  XGBoost Classifier    │
│  (Champion)         │  │  (Challenger)           │
│  LR + scorecardpy   │  │  sklearn Pipeline       │
│  Interpretable      │  │  Higher AUC             │
│  Credit Score       │  │  PD only                │
└─────────────────────┘  └───────────────────────┘
```

### Key Design Decisions

**Champion/Challenger Architecture:** The WOE Logistic Regression scorecard is the production champion — it's interpretable, auditable, and generates a credit score. The XGBoost model runs in parallel as a challenger for benchmarking and potential future promotion.

**Business Input Layer:** The API exposes a clean 10-field interface (`recent_delay`, `avg_bill_amount`, etc.) instead of requiring 21 raw dataset columns. A mapper converts these business-friendly fields deterministically to raw features at inference time — no data leakage possible.

**Artifact Freeze:** WOE bins and the scorecard points table are computed once on training data and saved to disk. Inference loads them as frozen artifacts — recomputing on inference data would cause data leakage.

---

## 📊 Dataset & Features

### Source Dataset

**UCI Taiwan Credit Card Default Dataset**
- 30,000 credit card customers, April–September 2005
- Target: `DEFAULT_NEXT_MONTH` (binary — 1 = defaulted, 0 = did not)
- Default rate: ~22%

### Raw Features (21 columns)

| Category | Features | Description |
|---|---|---|
| **Demographics** | `LIMIT_BAL`, `AGE`, `EDUCATION` | Credit limit, age, education level |
| **Repayment Status** | `PAY_0`, `PAY_2`–`PAY_6` | Monthly payment status (−2 to 8 scale) |
| **Bill Amounts** | `BILL_AMT1`–`BILL_AMT6` | Outstanding balance each month |
| **Payment Amounts** | `PAY_AMT1`–`PAY_AMT6` | Amount paid each month |

> Note: `PAY_1` is missing — this is a known quirk of the original dataset.

### Engineered Features (15+ derived)

| Feature | Formula / Logic | Why It Matters |
|---|---|---|
| `MAX_DELAY` | `max(PAY_0..PAY_6)` | Worst single-month delinquency — strongest default predictor |
| `NUM_DELAYS` | Count of PAY_X ≥ 1 | How frequently the borrower was late |
| `PAST_DELAY_AVG` | Mean of `PAY_2`–`PAY_6` | Chronic vs one-off lateness |
| `AVG_BILL_AMT` | Mean of 6 bill columns | Replaces 6 collinear columns (r > 0.90) |
| `BILL_GROWTH` | `BILL_AMT1 − BILL_AMT6` | Is debt growing or shrinking? |
| `NUM_ZERO_PAYMENTS` | Count of zero PAY_AMTs | Strong signal — chronic non-payers default far more |
| `PAY_BILL_RATIO_i` | `PAY_AMTi / BILL_AMTi` | Repayment coverage per month (clipped at p99 = 15) |
| `AVG_PAY_BILL_RATIO` | Mean of 6 ratios | Smoothed repayment behavior |
| `UTILIZATION` | `AVG_BILL_AMT / LIMIT_BAL` | How much credit is being used (capped at 1.05) |

---

## 🧹 Data Processing & EDA

### Data Cleaning

- **Duplicate header row** removed (`iloc[1:]`)
- **Undocumented EDUCATION codes** (0, 5, 6) remapped to 4 (Others) to prevent singleton WOE bins
- **Undocumented MARRIAGE code** (0) remapped to 3 (Others)
- **ID column** dropped — no predictive signal

### Feature Selection

Columns removed at each stage:

| Removed | Reason |
|---|---|
| `GENDER`, `MARRIAGE` | IV < 0.02; legally sensitive in credit models |
| `BILL_AMT2`–`BILL_AMT6` | r > 0.90 with each other; replaced by `AVG_BILL_AMT` + `BILL_GROWTH` |
| `PAY_2`–`PAY_6` | Summarized by `MAX_DELAY`, `NUM_DELAYS`, `PAST_DELAY_AVG` |
| `PAY_AMT2`–`PAY_AMT6` | Replaced by pay-to-bill ratios |
| `NUM_DELAYS` | r = 0.75 with `MAX_DELAY`; removed to reduce collinearity |
| `AVG_BILL_AMT` | Negative LR coefficient — captured better by `UTILIZATION` |
| `PAY_BILL_RATIO_1` | Negative coefficient; overlaps with `NUM_ZERO_PAYMENTS` |
| `EDUCATION` | Near-zero coefficient and low IV (0.036) |

### WOE Binning & Information Value

Binning uses `scorecardpy` on training data only. Features are ranked by Information Value (IV):

- IV > 0.30 → strong predictor
- IV 0.10–0.30 → medium predictor
- IV < 0.02 → useless (removed)

Manual bin breaks were applied to `UTILIZATION` to enforce **monotonic WOE** (a scorecard requirement).

---

## 🤖 Model Development (PD Model)

### Champion: WOE Logistic Regression Scorecard

**Why a scorecard?**
Scorecards have been the industry standard in retail credit for decades. They are:
- Fully interpretable — every point can be explained to a regulator or borrower
- Auditable — the points table is a fixed, human-readable artifact
- Monotonic — higher score always means lower risk

**Training Pipeline:**

```
Raw Data
  ↓ engineer_features()         — 15+ derived features
  ↓ select_features()           — remove collinear/weak/leaky
  ↓ train_test_split()          — stratified, 80/20
  ↓ woebin() [train only]       — compute WOE bins
  ↓ woebin_ply()                — apply WOE transform
  ↓ LogisticRegression.fit()    — all coefficients must be positive
  ↓ scorecard()                 — convert LR coefficients to points table
  ↓ scorecard_ply()             — score test set; validate monotonicity
```

**Two-pass LR approach:**
- Pass 1: identify features with negative coefficients and remove them
- Pass 2: refit on the cleaned feature set — this is the production model

**Score calibration:**
- Anchor: 600 points at average odds
- PDO (Points to Double Odds): 50
- Score range on test set: ~576–906

**Score Bands:**

| Score Range | Risk Label | Decision |
|---|---|---|
| < 650 | High Risk | Decline |
| 650–700 | Elevated | Decline |
| 700–750 | Moderate | Review |
| 750–800 | Low | Approve |
| 800–850 | Lower | Approve |
| > 850 | Minimal | Approve |

### Challenger: XGBoost Classifier

Wrapped in a `sklearn Pipeline` with `OrdinalEncoder` for the `EDUCATION` categorical feature. The pipeline handles preprocessing + inference in a single object.

Key hyperparameters:
- `scale_pos_weight = 3.52` — corrects for 22% default rate imbalance
- 5-fold CV validates that test AUC is not a lucky partition

**Business input mapping** (`core/input_mapper.py`):

Instead of asking users for 21 raw fields, both models accept 10 business-friendly inputs:

| Business Field | Maps To |
|---|---|
| `recent_delay` | `PAY_0` |
| `avg_past_delay` + `num_delays` | `PAY_2`–`PAY_6` |
| `avg_bill_amount` + `bill_growth_rate` | `BILL_AMT1`–`BILL_AMT6` |
| `payment_amount` + `zero_payment_count` | `PAY_AMT1`–`PAY_AMT6` |

---

## ⚖️ Risk Segmentation

Borrowers are grouped into risk buckets using two strategies:

**Quantile segmentation** — equal-sized groups (quintiles by default). Useful when the PD distribution is highly skewed and you want evenly populated monitoring groups.

**Fixed-threshold segmentation** — PD cut-points with business meaning:

| Bucket | PD Range | Risk Level |
|---|---|---|
| A | < 5% | Very Low |
| B | 5%–15% | Low |
| C | 15%–30% | Medium |
| D | 30%–50% | High |
| E | > 50% | Very High |

Each bucket returns count and average PD — giving risk managers a portfolio composition view at a glance.

---

## 💸 Expected Credit Loss (ECL)

### Formula

```
ECL = PD × LGD × EAD
```

| Component | Definition | Default Assumption |
|---|---|---|
| **PD** | Probability of Default | From model output |
| **LGD** | Loss Given Default | 45% (Basel II standard for unsecured retail) |
| **EAD** | Exposure at Default | Outstanding loan amount |

This follows **IFRS 9** methodology for Stage 1 (12-month) expected credit loss. The system computes:

- `individual_ecl` — per-borrower ECL
- `total_ecl` — portfolio provision amount (sum of all individual ECLs)
- `mean_ecl` — average per borrower
- `segment_ecl` — ECL breakdown by risk bucket (when segment labels are provided)

---

## 🎲 Monte Carlo Simulation

### Why Monte Carlo?

ECL gives the *expected* loss — the long-run average. But a risk manager also needs to know the *distribution* of losses, especially the tail. Monte Carlo simulates thousands of default/no-default scenarios and reveals that distribution.

### Simulation Approach

For each of `N` simulations:
1. Draw a random U[0,1] for each borrower
2. If U < PD → borrower defaults in this scenario
3. Borrower loss = `LGD × EAD` if defaulted, else 0
4. Portfolio loss = sum across all borrowers

After N simulations, compute statistics over the portfolio loss distribution.

### Key Metrics

| Metric | Definition |
|---|---|
| **Expected Loss (EL)** | Mean simulated loss — should approximate deterministic ECL |
| **Unexpected Loss (UL)** | Standard deviation — measures volatility around EL |
| **VaR (95%)** | Loss not exceeded in 95% of simulations |
| **CVaR** | Average loss in the worst 5% of simulations (Expected Shortfall) |

> **Note:** The simulation assumes independent defaults between borrowers. A correlated model (e.g. Vasicek) would require a shared systematic factor — a planned future improvement.

---

## 🌡️ Sensitivity & Stress Testing

### Sensitivity Analysis

Sweeps small perturbations of PD and LGD to identify which assumption has the greatest impact on ECL:

- **PD shifts** — applied as relative multipliers: `adjusted_PD = PD × (1 + shift)`
  - e.g. shift = +0.20 means "our PD estimates are 20% too low"
- **LGD shifts** — applied as absolute additions: `adjusted_LGD = LGD + shift`
  - e.g. shift = +0.10 means "recovery is 10 percentage points worse than assumed"

The key output is **elasticity** — % change in ECL per % change in the driver.

### Stress Testing

Applies predefined macroeconomic scenarios to the entire portfolio:

| Scenario | PD Multiplier | LGD Override | Interpretation |
|---|---|---|---|
| **Base Case** | 1.0× | None | Normal economic conditions |
| **Mild Stress** | 1.3× | 45% | Economic slowdown, early recession |
| **Severe Stress** | 1.8× | 55% | Deep recession, systemic shock |

Results show ECL and % change vs base for each scenario — the key input for capital planning and ICAAP (Internal Capital Adequacy Assessment Process).

---

## 🌐 API & Backend (FastAPI)

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | API liveness check |
| GET | `/model-info` | Training metadata (AUC, features) |
| POST | `/predict` | Scorecard prediction (raw input) |
| POST | `/predict/xgb` | XGBoost prediction (raw input) |
| POST | `/predict/business` | Scorecard prediction (business-friendly input) |
| POST | `/predict/business/xgb` | XGBoost prediction (business-friendly input) |
| POST | `/predict/both` | Both models, side-by-side |
| POST | `/whatif/scorecard` | Score/PD delta between two scenarios |
| POST | `/segmentation` | Risk bucket assignment |
| POST | `/ecl` | Expected Credit Loss calculation |
| POST | `/simulate` | Monte Carlo simulation |
| POST | `/stress-test` | Scenario stress testing |
| POST | `/sensitivity` | Sensitivity analysis |

### API Design Principles

- **Two input formats:** Raw 21-column `ClientInput` for technical integrations; 10-field `BusinessInput` for the frontend — translated by `input_mapper.py`
- **Thin route handlers:** Routes only validate and delegate; all business logic lives in `services/`
- **Pydantic everywhere:** All inputs and outputs are typed — Pydantic handles validation and generates the Swagger schema automatically
- **Stateless:** Every request is independent; no session state stored server-side
- **CORS enabled:** Frontend can call from any origin (restrict in production)

### Interactive Docs

Visit `/docs` on any running instance for the full Swagger UI with live request testing.

---

## 🖥️ Frontend Dashboard

A pure HTML/CSS/JavaScript dashboard — no framework dependency, runs as static files.

### Pages

| Page | URL | What It Does |
|---|---|---|
| **Dashboard** | `index.html` | Project overview, architecture diagram, feature navigation |
| **Prediction** | `predict.html` | Single-borrower risk prediction (Scorecard + XGBoost tabs); What-If analysis |
| **Risk Analysis** | `analytics.html` | Portfolio table with editable PD/EAD; segmentation + ECL calculation |
| **Simulation** | `simulation.html` | Monte Carlo simulation with configurable parameters and risk slider |
| **Stress Test** | `stress.html` | Three-scenario comparison with chart visualization |
| **Sensitivity** | `sensitivity.html` | PD and LGD sensitivity tables with elasticity |

### Tech Stack

- **CSS:** Custom design system with CSS variables; dark "financial terminal" aesthetic
- **Fonts:** Syne (display), DM Mono (data/code), DM Sans (body)
- **Charts:** Chart.js (stress testing bar chart)
- **API integration:** `api.js` — centralized fetch layer with error handling

---

## ⚙️ Deployment

### Production (Public Access)

The system is deployed on two free-tier platforms for public access:

| Layer | Platform | URL |
|---|---|---|
| **Frontend** | Vercel | Static HTML/CSS/JS served via CDN |
| **Backend API** | Render | FastAPI with all ML artifacts |

This split keeps costs at zero while giving a production-like separation of frontend and backend.

### AWS Deployment (Learning Exercise)

A full production-style deployment was also completed on AWS to understand cloud infrastructure:

#### Step-by-Step AWS Deployment

**1. EC2 Instance Setup**
- Launched `t3.micro` (free-tier eligible) with Ubuntu OS
- Configured key pair for secure SSH access
- Enabled auto-assign public IP

**2. Security Group Configuration**
- Inbound rules configured:
  - Port 22 (SSH) — remote terminal access
  - Port 80 (HTTP) — web traffic via Nginx
  - Port 8000 (Custom TCP) — FastAPI backend direct access
- Learned firewall-level access control via security groups

**3. Remote Server Access**
- Connected via SSH using key pair from local machine
- Managed server entirely through terminal to simulate a real production environment

**4. Backend Deployment**
```bash
# Clone the repository
git clone <repo-url>
cd credit-risk-modelling

# Create isolated Python environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
- Verified API at `http://<EC2-IP>:8000/docs`

**5. Real-World Debugging**
Encountered and resolved multiple production deployment issues:
- SSH connection failures due to security group misconfiguration
- Missing Python dependencies not in `requirements.txt`
- `ModuleNotFoundError` from incorrect working directory
- Port accessibility issues from firewall rules

**6. Nginx Reverse Proxy**
```nginx
server {
    listen 80;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```
- Hides the backend port from public exposure
- Enables clean URL access without `:8000` in the address

**7. Final AWS Architecture**
```
Browser → Nginx (Port 80) → FastAPI (Port 8000)
```

**8. Cost Management**
- Monitored AWS billing dashboard throughout
- Terminated EC2 instance after testing
- Verified cleanup of: EC2 instances, EBS volumes, Elastic IPs
- Ensured zero-cost state post-deployment

#### AWS Key Learnings
- Cloud compute and networking fundamentals (VPC, security groups, EC2)
- Reverse proxy architecture with Nginx
- Debugging real deployment failures (not just local errors)
- Cost awareness and resource lifecycle management in cloud environments

---

## 📂 Project Structure

```
credit-risk-modelling/
│
├── app/                          # FastAPI application layer
│   ├── main.py                   # App setup, all route definitions
│   ├── schema.py                 # Pydantic models: ClientInput, BusinessInput, responses
│   ├── routes/                   # One file per analytics endpoint
│   │   ├── ecl.py
│   │   ├── segmentation.py
│   │   ├── simulation.py
│   │   ├── stress.py
│   │   └── sensitivity.py
│   └── schemas/
│       └── risk_schemas.py       # Pydantic schemas for risk analytics endpoints
│
├── core/                         # Shared inference utilities
│   ├── config.py                 # All file paths and constants (single source of truth)
│   ├── preprocessing.py          # Feature engineering + WOE transform for inference
│   ├── input_mapper.py           # BusinessInput → raw 21-column dict
│   └── utils.py                  # KS statistic, validation helpers
│
├── services/                     # Business logic layer (no FastAPI dependencies)
│   ├── pd_model.py               # Scorecard + XGBoost inference; artifact loading
│   ├── ecl_service.py            # ECL = PD × LGD × EAD computation
│   ├── monte_carlo_service.py    # Monte Carlo loss simulation
│   ├── scenario_service.py       # Stress testing scenarios
│   ├── sensitivity_service.py    # PD/LGD sensitivity sweeps
│   ├── segmentation_service.py   # Risk bucket assignment
│   └── risk_config.py            # Tunable parameters: scenarios, thresholds, defaults
│
├── mlops/                        # Offline training pipeline
│   ├── train.py                  # Full end-to-end training: scorecard + XGBoost
│   └── evaluate.py               # Metrics, confusion matrix, ROC curve; MLflow logging
│
├── frontend/                     # Static HTML/CSS/JS dashboard
│   ├── index.html
│   ├── predict.html
│   ├── analytics.html
│   ├── simulation.html
│   ├── stress.html
│   ├── sensitivity.html
│   ├── css/
│   │   ├── styles.css            # Design system (variables, layout, typography)
│   │   └── components.css        # Reusable UI components
│   └── js/
│       ├── config.js             # API base URL (localhost vs production)
│       ├── api.js                # Centralized fetch layer with error handling
│       ├── index.js
│       ├── predict.js
│       ├── analytics.js
│       ├── simulation.js
│       ├── stress.js
│       └── sensitivity.js
│
├── artifacts/                    # Generated during training (not committed)
│   ├── models/
│   │   ├── scorecard_lr_model.joblib
│   │   └── xgb_pipeline.joblib
│   └── preprocessing/
│       ├── woe_bins.joblib
│       ├── scorecard.joblib
│       ├── feature_columns_scorecard.json
│       └── feature_columns_xgb.json
│
├── data/raw/credit_risk.xls      # Source dataset (not committed)
├── params.yaml                   # Hyperparameters (DVC-tracked)
├── requirements.txt
└── README.md
```

---

## 🧪 How to Run Locally

### Prerequisites

- Python 3.9+
- pip

### 1. Clone and Install

```bash
git clone <repo-url>
cd credit-risk-modelling
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train Models (or use pre-trained artifacts)

```bash
# Download dataset to data/raw/credit_risk.xls first
python mlops/train.py
```

This produces all artifacts in `artifacts/`. If you have pre-trained artifacts, skip this step.

### 3. Start the Backend

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- API: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`

### 4. Open the Frontend

Open `frontend/index.html` directly in your browser, **or** serve it with any static file server:

```bash
cd frontend
python -m http.server 3000
# then open http://localhost:3000
```

The frontend auto-detects `localhost` and points to `http://localhost:8000` (configured in `js/config.js`).

---

## 📈 Results & Metrics

### Scorecard (Champion) — Logistic Regression + WOE

| Metric | Value |
|---|---|
| AUC | ~0.77 |
| KS Statistic | ~0.44 |
| Gini Coefficient | ~0.54 |
| Score Range | 576–906 |
| Monotonic Score Bands | ✅ Yes |
| Positive Coefficients Only | ✅ Yes |
| Features Used | 10 WOE features |

### XGBoost (Challenger)

| Metric | Value |
|---|---|
| AUC | ~0.78 |
| KS Statistic | ~0.45 |
| CV Mean AUC (5-fold) | ~0.77 |
| CV Std AUC | ~0.004 |

> The XGBoost challenger achieves marginally higher AUC but is not interpretable as a points table. The scorecard remains the production champion for its explainability and regulatory compliance.

---

## 📌 Key Highlights

**Production-grade ML pipeline:**
- WOE binning with monotonicity enforcement
- Two-pass LR to guarantee all-positive scorecard coefficients
- Frozen artifact loading at startup — no per-request overhead

**Clean software architecture:**
- Strict separation: routes → services → core
- No business logic in route handlers
- Services are FastAPI-agnostic (pure Python functions)

**Regulatory alignment:**
- ECL follows IFRS 9 (PD × LGD × EAD)
- Stress testing follows Basel III / ICAAP principles
- Score bands are monotonic and explainable

**End-to-end MLOps:**
- MLflow experiment tracking for every training run
- `params.yaml` for hyperparameter versioning (DVC-compatible)
- Artifacts separated into `models/` and `preprocessing/` for independent retraining

**Real deployment experience:**
- AWS EC2 with Nginx reverse proxy (learning exercise)
- Render (FastAPI) + Vercel (frontend) for zero-cost public access
- Hands-on debugging of real cloud infrastructure issues

---

## 🔮 Future Improvements

| Area | Improvement |
|---|---|
| **Model** | Add correlated default simulation (Vasicek model) to Monte Carlo for more realistic tail risk |
| **Model** | Implement SHAP values for XGBoost feature explanation |
| **Model** | Periodic model monitoring and drift detection (PSI, CSI) |
| **Data** | Connect to a live data source or database instead of a static XLS file |
| **API** | Add JWT authentication and rate limiting for production use |
| **API** | Async endpoints for simulation (currently blocking for large `num_simulations`) |
| **Frontend** | Add interactive loss distribution histogram for Monte Carlo results |
| **Frontend** | Portfolio upload (CSV) instead of hard-coded sample data |
| **MLOps** | DVC pipeline for full data + model versioning |
| **Deployment** | Dockerize the backend for consistent environment across deployments |
| **Deployment** | CI/CD pipeline (GitHub Actions) for automated retraining and deployment |

---

## 📬 Contact

Built by **Om Patel**

- 📧 [ompatel2587@gmail.com](mailto:ompatel2587@gmail.com)
- 🔗 [LinkedIn](https://linkedin.com/in/om-patel-tech)

---

*Built with FastAPI · scikit-learn · XGBoost · scorecardpy · NumPy · Pandas · MLflow*