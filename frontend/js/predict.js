/**
 * PREDICTION PAGE — js/predict.js
 * ─────────────────────────────────────────────────────────────────
 *
 * WHAT CHANGED (Business Input Layer):
 *   - Forms now collect ~10 business-friendly inputs (not 20+ raw features)
 *   - Payload is sent to /predict/business or /predict/business/xgb
 *   - Backend (input_mapper.py) handles the mapping to raw dataset format
 *   - No raw PAY_X / BILL_AMT_X / PAY_AMT_X fields in the frontend anymore
 *
 * Field mapping reference:
 *   LIMIT_BAL       → passed through
 *   AGE             → passed through
 *   EDUCATION       → passed through (1–4 dropdown)
 *   recent_delay    → maps to PAY_0
 *   avg_past_delay  → maps to PAY_2..PAY_6 (controlled by num_delays)
 *   num_delays      → controls how many past months carry a delay
 *   avg_bill_amount → maps to BILL_AMT1..6
 *   bill_growth_rate→ trend variation across BILL_AMT months
 *   payment_amount  → maps to PAY_AMT1..6
 *   zero_payment_count → oldest N months set to $0 payment
 */

'use strict';

const API_BASE_URL = 'http://localhost:8000';

/** Currently active model tab */
let currentModel = 'scorecard';

console.log('[PREDICT] Page script loaded');

// ─────────────────────────────────────────────────────────────────
// NAVIGATION
// ─────────────────────────────────────────────────────────────────

function navigateTo(page) {
    console.log('[PREDICT] Navigating to:', page);
    window.location.href = page;
}

// ─────────────────────────────────────────────────────────────────
// MODEL TAB SWITCHING
// ─────────────────────────────────────────────────────────────────

/**
 * Switch the active model tab.
 * @param {string} model   - 'scorecard' | 'xgboost'
 * @param {Element} clickedBtn - The button element that was clicked
 */
function switchModel(model, clickedBtn) {
    console.log('[PREDICT] switchModel():', { from: currentModel, to: model });
    currentModel = model;

    // Update tab button styles
    document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
    clickedBtn.classList.add('active');

    // Show the correct form section
    document.querySelectorAll('.form-section').forEach(s => s.classList.remove('active'));
    const sectionId = model === 'scorecard' ? 'scorecard-section' : 'xgboost-section';
    document.getElementById(sectionId).classList.add('active');

    // Reset results and messages
    document.getElementById('results-section').classList.add('hidden');
    document.getElementById('message-container').innerHTML = '';

    console.log('[PREDICT] Active model:', model);
}

// ─────────────────────────────────────────────────────────────────
// FORM DATA COLLECTION
// ─────────────────────────────────────────────────────────────────

/**
 * Read the Scorecard form and return a business input payload.
 * All fields map to /predict/business (BusinessInput schema).
 */
function getScorecardBusinessPayload() {
    const payload = {
        // Demographics
        LIMIT_BAL:          parseFloat(document.getElementById('sc-limit-bal').value),
        AGE:                parseInt(document.getElementById('sc-age').value, 10),
        EDUCATION:          parseInt(document.getElementById('sc-education').value, 10),

        // Repayment behavior
        recent_delay:       parseFloat(document.getElementById('sc-recent-delay').value),
        avg_past_delay:     parseFloat(document.getElementById('sc-avg-past-delay').value),
        num_delays:         parseInt(document.getElementById('sc-num-delays').value, 10),

        // Billing history
        avg_bill_amount:    parseFloat(document.getElementById('sc-avg-bill').value),
        bill_growth_rate:   parseFloat(document.getElementById('sc-bill-growth').value),

        // Payment history
        payment_amount:     parseFloat(document.getElementById('sc-payment-amt').value),
        zero_payment_count: parseInt(document.getElementById('sc-zero-payments').value, 10),
    };
    console.log('[PREDICT] Scorecard business payload:', payload);
    return payload;
}

/**
 * Read the XGBoost form and return a business input payload.
 * All fields map to /predict/business/xgb (BusinessInput schema).
 */
function getXGBBusinessPayload() {
    const payload = {
        // Demographics
        LIMIT_BAL:          parseFloat(document.getElementById('xgb-limit-bal').value),
        AGE:                parseInt(document.getElementById('xgb-age').value, 10),
        EDUCATION:          parseInt(document.getElementById('xgb-education').value, 10),

        // Repayment behavior
        recent_delay:       parseFloat(document.getElementById('xgb-recent-delay').value),
        avg_past_delay:     parseFloat(document.getElementById('xgb-avg-past-delay').value),
        num_delays:         parseInt(document.getElementById('xgb-num-delays').value, 10),

        // Billing history
        avg_bill_amount:    parseFloat(document.getElementById('xgb-avg-bill').value),
        bill_growth_rate:   parseFloat(document.getElementById('xgb-bill-growth').value),

        // Payment history
        payment_amount:     parseFloat(document.getElementById('xgb-payment-amt').value),
        zero_payment_count: parseInt(document.getElementById('xgb-zero-payments').value, 10),
    };
    console.log('[PREDICT] XGBoost business payload:', payload);
    return payload;
}

// ─────────────────────────────────────────────────────────────────
// CLIENT-SIDE VALIDATION
// ─────────────────────────────────────────────────────────────────

/**
 * Validate a business payload before sending.
 * Mirrors the backend validate_business_input() rules so the user
 * gets fast, friendly feedback without a round-trip.
 *
 * @param {Object} payload
 * @returns {string|null} - Error message string, or null if valid
 */
function validatePayload(payload) {
    if (!payload.LIMIT_BAL || payload.LIMIT_BAL <= 0)
        return 'Credit Limit must be a positive number.';

    if (isNaN(payload.AGE) || payload.AGE < 18 || payload.AGE > 100)
        return 'Age must be between 18 and 100.';

    if (isNaN(payload.EDUCATION) || payload.EDUCATION < 1 || payload.EDUCATION > 4)
        return 'Education level must be 1–6.';

    if (isNaN(payload.recent_delay) || payload.recent_delay < -2 || payload.recent_delay > 8)
        return 'Recent Delay must be between −2 and 8.';

    if (isNaN(payload.avg_past_delay) || payload.avg_past_delay < -2 || payload.avg_past_delay > 8)
        return 'Avg Past Delay must be between −2 and 8.';

    if (isNaN(payload.num_delays) || payload.num_delays < 0 || payload.num_delays > 5)
        return 'Number of Delayed Months must be 0–5.';

    if (isNaN(payload.avg_bill_amount) || payload.avg_bill_amount < 0)
        return 'Avg Bill Amount must be a non-negative number.';

    if (isNaN(payload.bill_growth_rate) || payload.bill_growth_rate < -1 || payload.bill_growth_rate > 1)
        return 'Bill Growth Rate must be between −1 and 1.';

    if (isNaN(payload.payment_amount) || payload.payment_amount < 0)
        return 'Payment Amount must be a non-negative number.';

    if (isNaN(payload.zero_payment_count) || payload.zero_payment_count < 0 || payload.zero_payment_count > 6)
        return 'Zero-Payment Months must be 0–6.';

    return null; // all good
}

// ─────────────────────────────────────────────────────────────────
// UI HELPERS
// ─────────────────────────────────────────────────────────────────

function displayMessage(type, message) {
    const container = document.getElementById('message-container');
    const icon = type === 'success' ? '✓' : '⚠️';
    container.innerHTML = `
        <div class="alert alert-${type}">
            <span>${icon}</span><span>${message}</span>
        </div>`;
    if (type !== 'danger') {
        setTimeout(() => { container.innerHTML = ''; }, 5000);
    }
}

function showLoading() {
    const c = document.getElementById('loading-container');
    c.innerHTML = `<div class="loading"><div class="spinner"></div><span>Predicting...</span></div>`;
    c.classList.remove('hidden');
}

function hideLoading() {
    const c = document.getElementById('loading-container');
    c.innerHTML = '';
    c.classList.add('hidden');
}

function formatPercentage(value) {
    return (value * 100).toFixed(2) + '%';
}

function getRiskLevel(pd) {
    if (pd < 0.05) return 'low';
    if (pd < 0.20) return 'medium';
    return 'high';
}

function getRiskLabel(level) {
    return { low: 'Low', medium: 'Medium', high: 'High' }[level] || level;
}

function getDecision(riskLevel) {
    return { low: 'Approve', medium: 'Review', high: 'Reject' }[riskLevel] || 'Review';
}

// ─────────────────────────────────────────────────────────────────
// RESULT RENDERING
// ─────────────────────────────────────────────────────────────────

function renderScorecardResult(result) {
    const pd         = result.default_probability;
    const riskLevel = result.risk_level.toLowerCase();
    const riskLabel = result.risk_level;
    const decision  = result.decision;
    // const riskLevel  = getRiskLevel(pd);
    // const riskLabel  = getRiskLabel(riskLevel);
    // const decision   = getDecision(riskLevel);

    return `
        <div class="result-card">
            <div class="result-header">🔵 Scorecard Model Results</div>

            <div class="result-row">
                <span class="result-label">Credit Score</span>
                <span class="result-value" style="font-size: 1.4rem; font-weight: 700; color: var(--primary-color);">
                    ${result.credit_score ?? '—'}
                </span>
            </div>

            <div class="result-row">
                <span class="result-label">Probability of Default (PD)</span>
                <span class="result-value">${formatPercentage(pd)}</span>
            </div>

            <div class="result-row">
                <span class="result-label">Risk Level</span>
                <span class="result-value">
                    <span class="risk-indicator risk-${riskLevel}">● ${riskLabel} Risk</span>
                </span>
            </div>

            <div class="result-row">
                <span class="result-label">Lending Decision</span>
                <span class="result-value">
                    <span class="decision-badge decision-${decision.toLowerCase()}">${decision}</span>
                </span>
            </div>
        </div>`;
}

function renderXGBResult(result) {
    const pd         = result.default_probability;
    const riskLevel = result.risk_level.toLowerCase();
    const riskLabel = result.risk_level;
    const decision  = result.decision;
    // const riskLevel  = getRiskLevel(pd);
    // const riskLabel  = getRiskLabel(riskLevel);
    // const decision   = getDecision(riskLevel);

    return `
        <div class="result-card">
            <div class="result-header">🟠 XGBoost Model Results</div>

            <div class="result-row">
                <span class="result-label">Probability of Default (PD)</span>
                <span class="result-value">${formatPercentage(pd)}</span>
            </div>

            <div class="result-row">
                <span class="result-label">Risk Level</span>
                <span class="result-value">
                    <span class="risk-indicator risk-${riskLevel}">● ${riskLabel} Risk</span>
                </span>
            </div>

            <div class="result-row">
                <span class="result-label">Lending Decision</span>
                <span class="result-value">
                    <span class="decision-badge decision-${decision.toLowerCase()}">${decision}</span>
                </span>
            </div>
        </div>`;
}

// ─────────────────────────────────────────────────────────────────
// MAIN PREDICT FUNCTION
// ─────────────────────────────────────────────────────────────────

/**
 * Collect form data → validate → POST to /predict/business or
 * /predict/business/xgb → render result.
 *
 * The backend's input_mapper.py converts the business payload to
 * raw dataset features before passing to the model.
 */
async function predictRisk() {
    console.log('[PREDICT] predictRisk() called, model:', currentModel);

    // 1. Collect payload from the correct form
    const payload  = currentModel === 'scorecard'
        ? getScorecardBusinessPayload()
        : getXGBBusinessPayload();

    // 2. Client-side validation (fast feedback)
    const validationError = validatePayload(payload);
    if (validationError) {
        console.warn('[PREDICT] Validation failed:', validationError);
        displayMessage('danger', `❌ ${validationError}`);
        return;
    }

    // 3. Choose endpoint
    const endpoint = currentModel === 'scorecard'
        ? '/predict/business'
        : '/predict/business/xgb';

    console.log('[PREDICT] POSTing to:', `${API_BASE_URL}${endpoint}`);

    showLoading();

    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify(payload),
        });

        if (!response.ok) {
            // Try to extract a FastAPI detail message
            let detail = response.statusText;
            try {
                const errBody = await response.json();
                detail = errBody.detail || detail;
            } catch (_) { /* ignore parse error */ }
            throw new Error(detail);
        }

        const result = await response.json();
        console.log('[PREDICT] ✓ Prediction result:', result);

        hideLoading();

        // 4. Render results
        const html = currentModel === 'scorecard'
            ? renderScorecardResult(result)
            : renderXGBResult(result);

        document.getElementById('results-content').innerHTML = html;
        document.getElementById('results-section').classList.remove('hidden');
        displayMessage('success', '✓ Prediction completed successfully');

        // 5. Store globally so analytics pages can reuse
        window._lastPrediction = { model: currentModel, payload, result };
        console.log('[PREDICT] Stored window._lastPrediction for downstream reuse');

    } catch (error) {
        hideLoading();
        console.error('[PREDICT] ✗ Prediction error:', error.message);
        displayMessage('danger', `❌ ${error.message}`);
    }
}

async function runWhatIf() {
    console.log('[WHATIF] Running analysis');

    // 1. Get base payload (existing form)
    const basePayload = getScorecardBusinessPayload();

    // 2. Clone and modify
    const modifiedPayload = { ...basePayload };

    const feature = document.getElementById('whatif-feature').value;
    const value   = parseFloat(document.getElementById('whatif-value').value);

    // Apply change dynamically
    modifiedPayload[feature] = value;

    // Optional: adjust dependent fields (important for realism)
    if (feature === 'recent_delay') {
        modifiedPayload.num_delays = value > 0 ? 2 : 0;
    }

    // Optional: adjust num_delays automatically
    modifiedPayload.num_delays = modifiedPayload.recent_delay > 0 ? 2 : 0;

    console.log('[WHATIF] Base:', basePayload);
    console.log('[WHATIF] Modified:', modifiedPayload);

    try {
        const response = await fetch(`${API_BASE_URL}/whatif/scorecard`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                base_input: basePayload,
                modified_input: modifiedPayload
            }),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'What-if failed');
        }

        const result = await response.json();
        console.log('[WHATIF] Result:', result);

        // 3. Render result
        document.getElementById('whatif-result').innerHTML = `
            <div class="result-card">
                <div class="result-header">📊 What-if Result</div>

                <div class="result-row">
                    <span>Score:</span>
                    <span>
                        ${result.base_score} → ${result.new_score}
                        (${result.delta_score >= 0 ? '+' : ''}${result.delta_score})
                    </span>
                </div>

                <div class="result-row">
                    <span>PD:</span>
                    <span>
                        ${(result.base_pd * 100).toFixed(2)}% →
                        ${(result.new_pd * 100).toFixed(2)}%
                    </span>
                </div>
            </div>
        `;

    } catch (error) {
        console.error('[WHATIF] Error:', error.message);
        displayMessage('danger', `❌ ${error.message}`);
    }
}