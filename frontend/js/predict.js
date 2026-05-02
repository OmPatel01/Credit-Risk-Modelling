/**
 * PREDICTION PAGE — js/predict.js
 *
 * CHANGES FROM BACKEND UPDATE:
 *   1. normaliseRiskLevel() — backend now returns "High Risk", "Elevated", "Moderate",
 *      "Low", "Minimal" instead of simple low/medium/high. Map these to CSS classes.
 *   2. renderScorecardResult() — renders top_risk_drivers panel from response.
 *   3. renderScorecardResult() — shows policy override warning when decision was overridden.
 *   4. renderXGBResult() — same risk level normalisation applied.
 *   5. runWhatIf() — now renders base_decision → new_decision + decision_flipped flag.
 */

'use strict';

// const API_BASE_URL = 'http://localhost:8000';

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
// RISK LEVEL NORMALISATION
// NEW: backend returns "High Risk", "Elevated", "Moderate", "Low", "Minimal"
// We map these to the three CSS classes: risk-high, risk-medium, risk-low
// ─────────────────────────────────────────────────────────────────

/**
 * Map any backend risk_level string to a CSS-safe class suffix.
 * @param {string} riskLevel - e.g. "High Risk", "Elevated", "Moderate", "Low", "Minimal"
 * @returns {string} - "high" | "medium" | "low"
 */
function normaliseRiskClass(riskLevel) {
    const level = (riskLevel || '').toLowerCase();
    if (level.includes('high') || level === 'elevated') return 'high';
    if (level.includes('moderate'))                      return 'medium';
    return 'low'; // Low, Minimal, Unknown
}

// ─────────────────────────────────────────────────────────────────
// MODEL TAB SWITCHING
// ─────────────────────────────────────────────────────────────────

function switchModel(model, clickedBtn) {
    console.log('[PREDICT] switchModel():', { from: currentModel, to: model });
    currentModel = model;

    document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
    clickedBtn.classList.add('active');

    document.querySelectorAll('.form-section').forEach(s => s.classList.remove('active'));
    const sectionId = model === 'scorecard' ? 'scorecard-section' : 'xgboost-section';
    document.getElementById(sectionId).classList.add('active');

    document.getElementById('results-section').classList.add('hidden');
    document.getElementById('message-container').innerHTML = '';

    console.log('[PREDICT] Active model:', model);
}

// ─────────────────────────────────────────────────────────────────
// FORM DATA COLLECTION
// ─────────────────────────────────────────────────────────────────

function getScorecardBusinessPayload() {
    const payload = {
        LIMIT_BAL:          parseFloat(document.getElementById('sc-limit-bal').value),
        AGE:                parseInt(document.getElementById('sc-age').value, 10),
        EDUCATION:          parseInt(document.getElementById('sc-education').value, 10),
        recent_delay:       parseFloat(document.getElementById('sc-recent-delay').value),
        avg_past_delay:     parseFloat(document.getElementById('sc-avg-past-delay').value),
        num_delays:         parseInt(document.getElementById('sc-num-delays').value, 10),
        avg_bill_amount:    parseFloat(document.getElementById('sc-avg-bill').value),
        bill_growth_rate:   parseFloat(document.getElementById('sc-bill-growth').value),
        payment_amount:     parseFloat(document.getElementById('sc-payment-amt').value),
        zero_payment_count: parseInt(document.getElementById('sc-zero-payments').value, 10),
    };
    console.log('[PREDICT] Scorecard business payload:', payload);
    return payload;
}

function getXGBBusinessPayload() {
    const payload = {
        LIMIT_BAL:          parseFloat(document.getElementById('xgb-limit-bal').value),
        AGE:                parseInt(document.getElementById('xgb-age').value, 10),
        EDUCATION:          parseInt(document.getElementById('xgb-education').value, 10),
        recent_delay:       parseFloat(document.getElementById('xgb-recent-delay').value),
        avg_past_delay:     parseFloat(document.getElementById('xgb-avg-past-delay').value),
        num_delays:         parseInt(document.getElementById('xgb-num-delays').value, 10),
        avg_bill_amount:    parseFloat(document.getElementById('xgb-avg-bill').value),
        bill_growth_rate:   parseFloat(document.getElementById('xgb-bill-growth').value),
        payment_amount:     parseFloat(document.getElementById('xgb-payment-amt').value),
        zero_payment_count: parseInt(document.getElementById('xgb-zero-payments').value, 10),
    };
    console.log('[PREDICT] XGBoost business payload:', payload);
    return payload;
}

// ─────────────────────────────────────────────────────────────────
// CLIENT-SIDE VALIDATION
// ─────────────────────────────────────────────────────────────────

function validatePayload(payload) {
    if (!payload.LIMIT_BAL || payload.LIMIT_BAL <= 0)
        return 'Credit Limit must be a positive number.';
    if (isNaN(payload.AGE) || payload.AGE < 18 || payload.AGE > 100)
        return 'Age must be between 18 and 100.';
    if (isNaN(payload.EDUCATION) || payload.EDUCATION < 1 || payload.EDUCATION > 4)
        return 'Education level must be 1–4.';
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
    return null;
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

// ─────────────────────────────────────────────────────────────────
// RESULT RENDERING
// CHANGED: normaliseRiskClass() replaces direct .toLowerCase() on risk_level.
//          Added top_risk_drivers panel.
//          Added policy override advisory.
// ─────────────────────────────────────────────────────────────────

/**
 * Render top risk drivers returned by the backend.
 * NEW: backend now always includes top_risk_drivers in ScorecardResponse.
 * @param {Array} drivers - [{feature, contribution}, ...]
 * @returns {string} HTML string, or empty string if no drivers
 */
function renderTopRiskDrivers(drivers) {
    if (!drivers || drivers.length === 0) return '';

    const rows = drivers.map((d, i) => `
        <div class="result-row">
            <span class="result-label">#${i + 1} ${d.feature.replace(/_/g, ' ')}</span>
            <span class="result-value" style="color: var(--amber); font-size: 0.85rem;">
                contribution: ${d.contribution > 0 ? '+' : ''}${d.contribution.toFixed(4)}
            </span>
        </div>
    `).join('');

    return `
        <div class="result-card" style="margin-top: 1rem;">
            <div class="result-header">⚠️ Top Risk Drivers</div>
            ${rows}
            <div style="padding: 0.6rem 1.25rem; font-size: 0.78rem; color: var(--text-muted); font-family: var(--font-mono);">
                Positive contribution = feature is increasing default risk
            </div>
        </div>`;
}

/**
 * Render a policy override advisory if the model decision was overridden.
 * NEW: backend now applies policy rules; decision can be silently changed.
 * Without this, users see "Decline" with no explanation why.
 * @param {Object} result - full API response
 * @returns {string} HTML string, or empty string
 */
function renderPolicyNote(result) {
    // The backend doesn't currently expose policy_overridden on the ScorecardResponse
    // directly, but decision values of "Decline" from policy rules will have
    // a risk_level of Low/Moderate — that mismatch is the signal.
    // We check for it conservatively: only show the note when decision is Decline
    // but risk_level suggests it's not obviously high risk.
    const riskClass = normaliseRiskClass(result.risk_level);
    if (result.decision === 'Decline' && riskClass !== 'high') {
        return `
            <div class="info-box warning" style="margin-top: 1rem;">
                <strong>Policy Override Applied:</strong>
                This application was automatically declined due to a business rule
                (e.g. current payment overdue, chronic non-payment, or extreme utilisation)
                regardless of the model score.
            </div>`;
    }
    return '';
}

function renderScorecardResult(result) {
    const pd         = result.default_probability;
    const riskClass  = normaliseRiskClass(result.risk_level); // CHANGED: use normaliser
    const riskLabel  = result.risk_level;
    const decision   = result.decision;
    const decisionClass = decision.toLowerCase(); // Approve → approve, Decline → decline, Review → review

    return `
        <div class="result-card">
            <div class="result-header">🔵 Scorecard Model Results</div>

            <div class="result-row">
                <span class="result-label">Credit Score</span>
                <span class="result-value" style="font-size: 1.4rem; font-weight: 700; color: var(--acid);">
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
                    <span class="risk-indicator risk-${riskClass}">● ${riskLabel}</span>
                </span>
            </div>

            <div class="result-row">
                <span class="result-label">Lending Decision</span>
                <span class="result-value">
                    <span class="decision-badge decision-${decisionClass}">${decision}</span>
                </span>
            </div>
        </div>
        ${renderPolicyNote(result)}
        ${renderTopRiskDrivers(result.top_risk_drivers)}`;
}

function renderXGBResult(result) {
    const pd        = result.default_probability;
    const riskClass = normaliseRiskClass(result.risk_level); // CHANGED: use normaliser
    const riskLabel = result.risk_level;
    const decision  = result.decision;
    const decisionClass = decision.toLowerCase();

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
                    <span class="risk-indicator risk-${riskClass}">● ${riskLabel}</span>
                </span>
            </div>

            <div class="result-row">
                <span class="result-label">Lending Decision</span>
                <span class="result-value">
                    <span class="decision-badge decision-${decisionClass}">${decision}</span>
                </span>
            </div>
        </div>`;
}

// ─────────────────────────────────────────────────────────────────
// MAIN PREDICT FUNCTION
// ─────────────────────────────────────────────────────────────────

async function predictRisk() {
    console.log('[PREDICT] predictRisk() called, model:', currentModel);

    const payload = currentModel === 'scorecard'
        ? getScorecardBusinessPayload()
        : getXGBBusinessPayload();

    const validationError = validatePayload(payload);
    if (validationError) {
        console.warn('[PREDICT] Validation failed:', validationError);
        displayMessage('danger', `❌ ${validationError}`);
        return;
    }

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
            let detail = response.statusText;
            try {
                const errBody = await response.json();
                detail = errBody.detail || detail;
            } catch (_) {}
            throw new Error(detail);
        }

        const result = await response.json();
        console.log('[PREDICT] ✓ Prediction result:', result);

        hideLoading();

        const html = currentModel === 'scorecard'
            ? renderScorecardResult(result)
            : renderXGBResult(result);

        document.getElementById('results-content').innerHTML = html;
        document.getElementById('results-section').classList.remove('hidden');
        displayMessage('success', '✓ Prediction completed successfully');

        window._lastPrediction = { model: currentModel, payload, result };
        console.log('[PREDICT] Stored window._lastPrediction');

    } catch (error) {
        hideLoading();
        console.error('[PREDICT] ✗ Prediction error:', error.message);
        displayMessage('danger', `❌ ${error.message}`);
    }
}

// ─────────────────────────────────────────────────────────────────
// WHAT-IF ANALYSIS
// CHANGED: now renders base_decision → new_decision and decision_flipped flag.
// Backend returns these fields since WhatIfResponse was extended.
// ─────────────────────────────────────────────────────────────────

async function runWhatIf() {
    console.log('[WHATIF] Running analysis');

    const basePayload     = getScorecardBusinessPayload();
    const modifiedPayload = { ...basePayload };

    const feature = document.getElementById('whatif-feature').value;
    const value   = parseFloat(document.getElementById('whatif-value').value);

    modifiedPayload[feature] = value;

    if (feature === 'recent_delay') {
        modifiedPayload.num_delays = value > 0 ? 2 : 0;
    }
    modifiedPayload.num_delays = modifiedPayload.recent_delay > 0 ? 2 : 0;

    console.log('[WHATIF] Base:', basePayload);
    console.log('[WHATIF] Modified:', modifiedPayload);

    const whatifBtn = document.querySelector('[onclick="runWhatIf()"]');
    if (whatifBtn) {
        whatifBtn.disabled = true;
        whatifBtn.textContent = '⏳ Analyzing...';
    }

    try {
        const response = await fetch(`${API_BASE_URL}/whatif/scorecard`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                base_input:     basePayload,
                modified_input: modifiedPayload
            }),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'What-if failed');
        }

        const result = await response.json();
        console.log('[WHATIF] Result:', result);

        // CHANGED: render decision flip indicator — new field from backend
        const decisionFlipHTML = result.decision_flipped
            ? `<div class="result-row" style="background: rgba(0,229,200,0.05);">
                   <span class="result-label">Decision Changed</span>
                   <span class="result-value" style="color: var(--acid);">
                       <span class="decision-badge decision-${(result.base_decision||'').toLowerCase()}">${result.base_decision}</span>
                       → 
                       <span class="decision-badge decision-${(result.new_decision||'').toLowerCase()}">${result.new_decision}</span>
                   </span>
               </div>`
            : `<div class="result-row">
                   <span class="result-label">Decision</span>
                   <span class="result-value" style="color: var(--text-muted); font-size: 0.85rem;">
                       No change (${result.base_decision || '—'})
                   </span>
               </div>`;

        const scoreDelta = result.delta_score;
        const pdDelta    = result.delta_pd;

        document.getElementById('whatif-result').innerHTML = `
            <div class="result-card">
                <div class="result-header">📊 What-if Result</div>

                <div class="result-row">
                    <span class="result-label">Credit Score</span>
                    <span class="result-value">
                        ${result.base_score} → ${result.new_score}
                        <span style="color: ${scoreDelta >= 0 ? 'var(--green)' : 'var(--red)'}; margin-left: 0.5rem;">
                            (${scoreDelta >= 0 ? '+' : ''}${scoreDelta})
                        </span>
                    </span>
                </div>

                <div class="result-row">
                    <span class="result-label">Probability of Default</span>
                    <span class="result-value">
                        ${(result.base_pd * 100).toFixed(2)}% → ${(result.new_pd * 100).toFixed(2)}%
                        <span style="color: ${pdDelta <= 0 ? 'var(--green)' : 'var(--red)'}; margin-left: 0.5rem;">
                            (${pdDelta > 0 ? '+' : ''}${(pdDelta * 100).toFixed(2)}%)
                        </span>
                    </span>
                </div>

                ${decisionFlipHTML}
            </div>
        `;

    } catch (error) {
        console.error('[WHATIF] Error:', error.message);
        displayMessage('danger', `❌ ${error.message}`);
    } finally {
        if (whatifBtn) {
            whatifBtn.disabled = false;
            whatifBtn.textContent = '🔄 Run What-if Analysis';
        }
    }
}