/**
 * PREDICTION PAGE — js/predict.js
 *
 * SENIOR CREDIT RISK MODELLER REVIEW — CHANGES IN THIS VERSION:
 *
 *  1. predictRisk() now calls /explain and /recommend sequentially after
 *     the initial /predict/business call. Results are rendered in two
 *     new sections: "Why This Decision" and "What To Do About It".
 *
 *  2. renderExplainSection() — renders per-feature WOE contribution table
 *     with human-readable labels, direction badges, and score point bars.
 *     Uses FEATURE_LABELS from the backend response (label field on each row).
 *
 *  3. renderRecommendSection() — renders prioritised, actionable cards.
 *     Each card shows: current value, specific action, expected impact tier.
 *     This replaces the invisible /recommend endpoint the backend already provides.
 *
 *  4. renderScorecardResult() — now shows "X points to next threshold" so a
 *     loan officer knows how close the borrower is to a decision flip.
 *
 *  5. renderPolicyNote() — now uses policy_overridden and hard_rule_triggered
 *     from /explain response directly instead of the fragile heuristic inference.
 *
 *  6. renderTopRiskDrivers() — now uses the label field from /explain
 *     feature_explanations instead of raw underscore-separated feature names.
 *
 *  7. What-if feature dropdown is pre-populated from the top risk driver
 *     of the last explain result so the officer doesn't have to guess.
 *
 *  8. All error paths are handled per-section: if /explain fails, the base
 *     prediction still renders. If /recommend fails, explain still renders.
 */

'use strict';

let currentModel = 'scorecard';

// Stores the last explain result so what-if can pre-fill the top driver
let _lastExplainResult = null;

console.log('[PREDICT] Page script loaded');

// ─────────────────────────────────────────────────────────────────
// NAVIGATION
// ─────────────────────────────────────────────────────────────────

function navigateTo(page) {
    window.location.href = page;
}

// ─────────────────────────────────────────────────────────────────
// RISK LEVEL NORMALISATION
// ─────────────────────────────────────────────────────────────────

function normaliseRiskClass(riskLevel) {
    const level = (riskLevel || '').toLowerCase();
    if (level.includes('high') || level === 'elevated') return 'high';
    if (level.includes('moderate'))                      return 'medium';
    return 'low';
}

// ─────────────────────────────────────────────────────────────────
// MODEL TAB SWITCHING
// ─────────────────────────────────────────────────────────────────

function switchModel(model, clickedBtn) {
    currentModel = model;
    document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
    clickedBtn.classList.add('active');
    document.querySelectorAll('.form-section').forEach(s => s.classList.remove('active'));
    const sectionId = model === 'scorecard' ? 'scorecard-section' : 'xgboost-section';
    document.getElementById(sectionId).classList.add('active');
    document.getElementById('results-section').classList.add('hidden');
    document.getElementById('message-container').innerHTML = '';
}

// ─────────────────────────────────────────────────────────────────
// FORM DATA COLLECTION
// ─────────────────────────────────────────────────────────────────

function getScorecardBusinessPayload() {
    return {
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
}

function getXGBBusinessPayload() {
    return {
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
    container.innerHTML = `<div class="alert alert-${type}"><span>${icon}</span><span>${message}</span></div>`;
    if (type !== 'danger') setTimeout(() => { container.innerHTML = ''; }, 5000);
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
// SCORE THRESHOLD HELPER
// Returns how many points the borrower is from the next band boundary.
// SCORE_BANDS mirrors core/config.py SCORE_BANDS.
// ─────────────────────────────────────────────────────────────────

const SCORE_BANDS = [
    { low: 0,   high: 650, risk: 'High Risk',  decision: 'Decline' },
    { low: 650, high: 700, risk: 'Elevated',   decision: 'Decline' },
    { low: 700, high: 750, risk: 'Moderate',   decision: 'Review'  },
    { low: 750, high: 800, risk: 'Low',        decision: 'Approve' },
    { low: 800, high: 850, risk: 'Lower',      decision: 'Approve' },
    { low: 850, high: 9999,risk: 'Minimal',    decision: 'Approve' },
];

function getScoreContext(score) {
    for (let i = 0; i < SCORE_BANDS.length; i++) {
        const band = SCORE_BANDS[i];
        if (score >= band.low && score < band.high) {
            const nextBand = SCORE_BANDS[i + 1];
            if (!nextBand) return { pointsToNext: null, nextDecision: null };
            const pointsToNext = band.high - score;
            return {
                pointsToNext,
                nextDecision: nextBand.decision,
                nextBandName: nextBand.risk,
                willFlip: nextBand.decision !== band.decision,
            };
        }
    }
    return { pointsToNext: null, nextDecision: null };
}

// ─────────────────────────────────────────────────────────────────
// RESULT RENDERING — BASE PREDICTION
// ─────────────────────────────────────────────────────────────────

function renderScorecardResult(result) {
    const pd          = result.default_probability;
    const riskClass   = normaliseRiskClass(result.risk_level);
    const riskLabel   = result.risk_level;
    const decision    = result.decision;
    const decisionCls = decision.toLowerCase();
    const ctx         = getScoreContext(result.credit_score);

    // "Points to next threshold" banner — critical for loan officers
    let thresholdBanner = '';
    if (ctx.pointsToNext !== null) {
        const flipColour = ctx.willFlip ? 'var(--acid)' : 'var(--text-muted)';
        const flipLabel  = ctx.willFlip
            ? `→ would become <strong>${ctx.nextDecision}</strong>`
            : `(stays ${decision})`;
        thresholdBanner = `
            <div style="
                background: rgba(0,229,200,0.05);
                border: 1px solid rgba(0,229,200,0.2);
                border-radius: 8px;
                padding: 0.75rem 1.25rem;
                margin: 0.75rem 0 0;
                font-family: var(--font-mono);
                font-size: 0.8rem;
                color: var(--text-secondary);
            ">
                <span style="color:${flipColour}; font-weight:600;">
                    ${ctx.pointsToNext} score point${ctx.pointsToNext === 1 ? '' : 's'}
                </span>
                to next band (${ctx.nextBandName}) ${flipLabel}
            </div>`;
    }

    return `
        <div class="result-card">
            <div class="result-header">🔵 Scorecard Model Results</div>

            <div class="result-row">
                <span class="result-label">Credit Score</span>
                <span class="result-value" style="font-size:1.4rem; font-weight:700; color:var(--acid);">
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
                    <span class="decision-badge decision-${decisionCls}">${decision}</span>
                </span>
            </div>

            ${thresholdBanner}
        </div>`;
}

function renderXGBResult(result) {
    const pd        = result.default_probability;
    const riskClass = normaliseRiskClass(result.risk_level);
    const decision  = result.decision;

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
                    <span class="risk-indicator risk-${riskClass}">● ${result.risk_level}</span>
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
// RENDER POLICY OVERRIDE NOTE
// Uses hard_rule_triggered from /explain response directly.
// No more heuristic inference.
// ─────────────────────────────────────────────────────────────────

function renderPolicyNote(explainResult) {
    if (!explainResult || !explainResult.policy_overridden) return '';

    const rule = explainResult.hard_rule_triggered || 'A hard business rule was triggered.';
    // Strip the rule code prefix (e.g. "R001: ") for cleaner display
    const cleanReason = rule.replace(/^R\d+:\s*/i, '');

    return `
        <div class="info-box warning" style="margin-top:1rem;">
            <strong>⚠️ Policy Override Applied:</strong><br>
            ${cleanReason}
        </div>`;
}

// ─────────────────────────────────────────────────────────────────
// RENDER SOFT FLAGS (advisory — not overrides)
// ─────────────────────────────────────────────────────────────────

function renderSoftFlags(explainResult) {
    if (!explainResult || !explainResult.soft_flags || explainResult.soft_flags.length === 0) return '';

    const items = explainResult.soft_flags.map(flag => {
        const clean = flag.replace(/^S\d+:\s*/i, '');
        return `<li style="margin-bottom:0.3rem;">${clean}</li>`;
    }).join('');

    return `
        <div class="info-box" style="margin-top:1rem;">
            <strong>📋 Advisory Flags (no override):</strong>
            <ul style="margin:0.5rem 0 0 1rem;">${items}</ul>
        </div>`;
}

// ─────────────────────────────────────────────────────────────────
// RENDER EXPLAIN SECTION
// "Why This Decision" — per-feature WOE contribution breakdown.
// ─────────────────────────────────────────────────────────────────

function renderExplainSection(explainResult) {
    if (!explainResult || !explainResult.feature_explanations) return '';

    const features = explainResult.feature_explanations;

    // Separate risk-increasing from risk-decreasing for visual grouping
    const increasing = features.filter(f => f.risk_direction === 'increases_risk');
    const decreasing = features.filter(f => f.risk_direction === 'decreases_risk');

    function featureRow(f) {
        const dirColour = f.risk_direction === 'increases_risk'
            ? 'var(--red)'
            : f.risk_direction === 'decreases_risk' ? 'var(--green)' : 'var(--text-muted)';
        const dirLabel = f.risk_direction === 'increases_risk'
            ? '▲ Risk ↑'
            : f.risk_direction === 'decreases_risk' ? '▼ Risk ↓' : '— Neutral';
        const sign = f.contribution >= 0 ? '+' : '';
        // Normalise bar width: max contribution magnitude = 100%
        const barPct = Math.min(Math.abs(f.contribution) * 300, 100).toFixed(1);
        const barColour = f.risk_direction === 'increases_risk'
            ? 'rgba(255,74,94,0.6)' : 'rgba(0,200,150,0.5)';

        return `
            <tr>
                <td style="color:var(--text-primary); font-weight:500;">${f.label}</td>
                <td style="color:${dirColour}; font-size:0.75rem;">${dirLabel}</td>
                <td>
                    <div style="display:flex; align-items:center; gap:0.5rem;">
                        <div style="
                            width:${barPct}%;
                            min-width:2px;
                            height:6px;
                            background:${barColour};
                            border-radius:3px;
                            max-width:80px;
                        "></div>
                        <span style="font-family:var(--font-mono); font-size:0.78rem; color:var(--text-secondary);">
                            ${sign}${f.contribution.toFixed(4)}
                        </span>
                    </div>
                </td>
                <td style="font-family:var(--font-mono); font-size:0.78rem; color:var(--text-secondary); text-align:right;">
                    ${f.score_points >= 0 ? '+' : ''}${f.score_points} pts
                </td>
            </tr>`;
    }

    const basePoints = explainResult.base_points || 0;

    return `
        <div class="result-card" style="margin-top:1.5rem;">
            <div class="result-header">🔬 Why This Decision — Feature Contribution Breakdown</div>

            <div style="padding:0.75rem 1.25rem; background:var(--surface-2); border-bottom:1px solid var(--border);">
                <span style="font-family:var(--font-mono); font-size:0.78rem; color:var(--text-muted);">
                    Base Points (Intercept): ${basePoints}
                </span>
                <span style="font-family:var(--font-mono); font-size:0.78rem; color:var(--text-muted); margin-left:1.5rem;">
                    Final Score: ${explainResult.credit_score}
                </span>
            </div>

            ${increasing.length > 0 ? `
            <div style="padding:0.6rem 1.25rem 0.2rem; font-family:var(--font-mono); font-size:0.68rem; color:var(--red); text-transform:uppercase; letter-spacing:0.08em;">
                Risk-Increasing Factors
            </div>
            <div class="table-wrapper" style="border:none; border-radius:0;">
                <table>
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>Direction</th>
                            <th>Contribution</th>
                            <th style="text-align:right;">Score Points</th>
                        </tr>
                    </thead>
                    <tbody>${increasing.map(featureRow).join('')}</tbody>
                </table>
            </div>` : ''}

            ${decreasing.length > 0 ? `
            <div style="padding:0.6rem 1.25rem 0.2rem; font-family:var(--font-mono); font-size:0.68rem; color:var(--green); text-transform:uppercase; letter-spacing:0.08em; border-top:1px solid var(--border);">
                Risk-Decreasing Factors (Working In Your Favour)
            </div>
            <div class="table-wrapper" style="border:none; border-radius:0;">
                <table>
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>Direction</th>
                            <th>Contribution</th>
                            <th style="text-align:right;">Score Points</th>
                        </tr>
                    </thead>
                    <tbody>${decreasing.map(featureRow).join('')}</tbody>
                </table>
            </div>` : ''}

            <div style="padding:0.75rem 1.25rem; background:var(--surface-2); font-size:0.78rem; color:var(--text-muted); font-family:var(--font-mono);">
                Contribution = Model Coefficient × WOE Value. Positive = pushes toward default.
            </div>
        </div>`;
}

// ─────────────────────────────────────────────────────────────────
// RENDER RECOMMEND SECTION
// "What To Do About It" — actionable improvement cards.
// This is the critical missing piece for decision-making value.
// ─────────────────────────────────────────────────────────────────

function renderRecommendSection(recommendResult) {
    if (!recommendResult || !recommendResult.recommendations || recommendResult.recommendations.length === 0) {
        // No risky features — borrower is in good shape
        return `
            <div class="result-card" style="margin-top:1.5rem;">
                <div class="result-header">✅ Improvement Recommendations</div>
                <div style="padding:1.5rem 1.25rem; color:var(--green); font-size:0.9rem;">
                    No high-risk factors identified. This borrower's profile is in good standing.
                    Maintain current payment behaviour and credit utilisation.
                </div>
            </div>`;
    }

    const recs = recommendResult.recommendations;

    const impactColour = {
        'High':   'var(--red)',
        'Medium': 'var(--amber)',
        'Low':    'var(--blue)',
    };
    const impactBg = {
        'High':   'var(--red-bg)',
        'Medium': 'var(--amber-bg)',
        'Low':    'var(--blue-bg)',
    };

    const cards = recs.map(rec => `
        <div style="
            background: var(--surface-2);
            border: 1px solid var(--border);
            border-left: 3px solid ${impactColour[rec.expected_impact] || 'var(--text-muted)'};
            border-radius: 0 8px 8px 0;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
        ">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:0.5rem;">
                <div>
                    <span style="
                        font-family:var(--font-mono);
                        font-size:0.68rem;
                        color:var(--text-muted);
                        text-transform:uppercase;
                        letter-spacing:0.08em;
                    ">Priority ${rec.priority}</span>
                    <div style="font-weight:600; color:var(--text-primary); margin-top:0.15rem;">
                        ${rec.label}
                    </div>
                </div>
                <span style="
                    background:${impactBg[rec.expected_impact] || 'transparent'};
                    color:${impactColour[rec.expected_impact] || 'var(--text-muted)'};
                    font-family:var(--font-mono);
                    font-size:0.68rem;
                    font-weight:600;
                    padding:0.2rem 0.6rem;
                    border-radius:4px;
                    white-space:nowrap;
                ">
                    ${rec.expected_impact} Impact
                </span>
            </div>

            <div style="
                display:flex;
                gap:0.5rem;
                align-items:center;
                margin-bottom:0.6rem;
                font-family:var(--font-mono);
                font-size:0.78rem;
            ">
                <span style="color:var(--text-muted);">Current:</span>
                <span style="
                    background:var(--surface-1);
                    border:1px solid var(--border);
                    padding:0.15rem 0.5rem;
                    border-radius:4px;
                    color:var(--text-secondary);
                ">${rec.current_value}</span>
            </div>

            <div style="color:var(--text-secondary); font-size:0.875rem; line-height:1.6;">
                ${rec.action}
            </div>
        </div>
    `).join('');

    return `
        <div class="result-card" style="margin-top:1.5rem;">
            <div class="result-header">💡 What To Do — Actionable Improvement Plan</div>
            <div style="
                padding:0.75rem 1.25rem;
                background:var(--surface-2);
                border-bottom:1px solid var(--border);
                font-size:0.8rem;
                color:var(--text-secondary);
                line-height:1.5;
            ">
                Recommendations are ranked by risk impact. Addressing Priority 1 first will have
                the largest effect on the credit score. The expected impact tier (High/Medium/Low)
                reflects how much improvement each action typically produces within 3–6 months.
            </div>
            <div style="padding:1.25rem;">
                ${cards}
            </div>
            <div style="
                padding:0.75rem 1.25rem;
                background:var(--surface-2);
                border-top:1px solid var(--border);
                font-size:0.78rem;
                color:var(--text-muted);
                font-family:var(--font-mono);
            ">
                These recommendations are specific to the scorecard model's risk drivers.
                Individual circumstances may vary. Credit improvement typically takes 3–12 months
                of consistent behaviour change.
            </div>
        </div>`;
}

// ─────────────────────────────────────────────────────────────────
// MAIN PREDICT FUNCTION
// Now orchestrates: predict → explain → recommend in sequence.
// Each step degrades gracefully if the API call fails.
// ─────────────────────────────────────────────────────────────────

async function predictRisk() {
    const payload = currentModel === 'scorecard'
        ? getScorecardBusinessPayload()
        : getXGBBusinessPayload();

    const validationError = validatePayload(payload);
    if (validationError) {
        displayMessage('danger', `❌ ${validationError}`);
        return;
    }

    const endpoint = currentModel === 'scorecard'
        ? '/predict/business'
        : '/predict/business/xgb';

    showLoading();

    try {
        // ── Step 1: Base prediction
        const predResp = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        if (!predResp.ok) {
            let detail = predResp.statusText;
            try { detail = (await predResp.json()).detail || detail; } catch (_) {}
            throw new Error(detail);
        }
        const predResult = await predResp.json();
        console.log('[PREDICT] Prediction result:', predResult);

        // ── Build base HTML
        let html = currentModel === 'scorecard'
            ? renderScorecardResult(predResult)
            : renderXGBResult(predResult);

        // ── Step 2 & 3: Explain + Recommend (scorecard only — XGBoost has no WOE layer)
        let explainResult   = null;
        let recommendResult = null;

        if (currentModel === 'scorecard') {
            // Explain
            try {
                const expResp = await fetch(`${API_BASE_URL}/explain`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                if (expResp.ok) {
                    explainResult   = await expResp.json();
                    _lastExplainResult = explainResult;
                    console.log('[PREDICT] Explain result:', explainResult);
                }
            } catch (e) {
                console.warn('[PREDICT] /explain call failed (non-fatal):', e.message);
            }

            // Recommend
            try {
                const recResp = await fetch(`${API_BASE_URL}/recommend`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                if (recResp.ok) {
                    recommendResult = await recResp.json();
                    console.log('[PREDICT] Recommend result:', recommendResult);
                }
            } catch (e) {
                console.warn('[PREDICT] /recommend call failed (non-fatal):', e.message);
            }

            // Append policy note and soft flags from explain result
            html += renderPolicyNote(explainResult);
            html += renderSoftFlags(explainResult);

            // Append explain breakdown
            html += renderExplainSection(explainResult);

            // Append recommendation cards
            html += renderRecommendSection(recommendResult);

            // Pre-fill what-if dropdown with the top risk driver
            prefillWhatIfFromExplain(explainResult);
        }

        hideLoading();
        document.getElementById('results-content').innerHTML = html;
        document.getElementById('results-section').classList.remove('hidden');
        displayMessage('success', '✓ Prediction and explanation completed');

        window._lastPrediction = { model: currentModel, payload, predResult, explainResult, recommendResult };

    } catch (error) {
        hideLoading();
        console.error('[PREDICT] Error:', error.message);
        displayMessage('danger', `❌ ${error.message}`);
    }
}

// ─────────────────────────────────────────────────────────────────
// PRE-FILL WHAT-IF DROPDOWN
// Sets the feature dropdown to the top risk driver from explain.
// Stops the officer having to guess which feature to simulate.
// ─────────────────────────────────────────────────────────────────

// Maps explain feature names to the what-if dropdown option values
const FEATURE_TO_WHATIF = {
    'PAY_0':             'recent_delay',
    'MAX_DELAY':         'avg_past_delay',
    'PAST_DELAY_AVG':    'avg_past_delay',
    'NUM_DELAYS':        'num_delays',
    'NUM_ZERO_PAYMENTS': 'zero_payment_count',
    'AVG_PAY_BILL_RATIO':'payment_amount',
    'PAY_AMT1':          'payment_amount',
    'UTILIZATION':       'avg_bill_amount',
    'BILL_GROWTH':       'bill_growth_rate',
    'LIMIT_BAL':         'LIMIT_BAL',
    'AGE':               'LIMIT_BAL', // age not changeable; fall back to limit
};

function prefillWhatIfFromExplain(explainResult) {
    if (!explainResult || !explainResult.feature_explanations) return;

    const dropdown = document.getElementById('whatif-feature');
    if (!dropdown) return;

    // Find the top risk-increasing feature that maps to a what-if field
    const topDriver = explainResult.feature_explanations.find(f => {
        return f.risk_direction === 'increases_risk' && FEATURE_TO_WHATIF[f.feature];
    });

    if (topDriver) {
        const targetValue = FEATURE_TO_WHATIF[topDriver.feature];
        dropdown.value = targetValue;
        console.log(`[WHATIF] Pre-filled dropdown with top driver: ${topDriver.feature} → ${targetValue}`);

        // Also add a hint label below the dropdown
        const hint = document.getElementById('whatif-feature-hint');
        if (hint) {
            hint.textContent = `Suggested from top risk driver: ${topDriver.label}`;
            hint.style.color = 'var(--amber)';
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// WHAT-IF ANALYSIS
// ─────────────────────────────────────────────────────────────────

async function runWhatIf() {
    const basePayload     = getScorecardBusinessPayload();
    const modifiedPayload = { ...basePayload };

    const feature = document.getElementById('whatif-feature').value;
    const value   = parseFloat(document.getElementById('whatif-value').value);

    modifiedPayload[feature] = value;
    modifiedPayload.num_delays = modifiedPayload.recent_delay > 0 ? 2 : 0;

    const whatifBtn = document.querySelector('[onclick="runWhatIf()"]');
    if (whatifBtn) { whatifBtn.disabled = true; whatifBtn.textContent = '⏳ Analyzing...'; }

    try {
        const response = await fetch(`${API_BASE_URL}/whatif/scorecard`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ base_input: basePayload, modified_input: modifiedPayload }),
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'What-if failed');
        }

        const result = await response.json();

        const decisionFlipHTML = result.decision_flipped
            ? `<div class="result-row" style="background:rgba(0,229,200,0.05);">
                   <span class="result-label">⚡ Decision Changed</span>
                   <span class="result-value" style="color:var(--acid);">
                       <span class="decision-badge decision-${(result.base_decision||'').toLowerCase()}">${result.base_decision}</span>
                       →
                       <span class="decision-badge decision-${(result.new_decision||'').toLowerCase()}">${result.new_decision}</span>
                   </span>
               </div>`
            : `<div class="result-row">
                   <span class="result-label">Decision</span>
                   <span class="result-value" style="color:var(--text-muted); font-size:0.85rem;">
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
                        <span style="color:${scoreDelta >= 0 ? 'var(--green)' : 'var(--red)'}; margin-left:0.5rem;">
                            (${scoreDelta >= 0 ? '+' : ''}${scoreDelta})
                        </span>
                    </span>
                </div>

                <div class="result-row">
                    <span class="result-label">Probability of Default</span>
                    <span class="result-value">
                        ${(result.base_pd * 100).toFixed(2)}% → ${(result.new_pd * 100).toFixed(2)}%
                        <span style="color:${pdDelta <= 0 ? 'var(--green)' : 'var(--red)'}; margin-left:0.5rem;">
                            (${pdDelta > 0 ? '+' : ''}${(pdDelta * 100).toFixed(2)}%)
                        </span>
                    </span>
                </div>

                ${decisionFlipHTML}

                ${result.decision_flipped ? `
                <div style="padding:0.75rem 1.25rem; background:rgba(0,229,200,0.05); font-size:0.82rem; color:var(--acid); font-family:var(--font-mono);">
                    ✓ This change would be sufficient to flip the lending decision.
                </div>` : ''}
            </div>`;

    } catch (error) {
        displayMessage('danger', `❌ ${error.message}`);
    } finally {
        if (whatifBtn) { whatifBtn.disabled = false; whatifBtn.textContent = '🔄 Run What-if Analysis'; }
    }
}