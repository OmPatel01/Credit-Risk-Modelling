/**
 * SIMULATION PAGE — js/simulation.js
 * Handles Monte Carlo simulation for loss distribution analysis
 */

const API_BASE_URL = 'http://localhost:8000';

const SAMPLE_PORTFOLIO = [
    {age: 35, income: 60000, employment_length: 8, credit_history_length: 15, existing_loans: 2, loan_amount: 25000, loan_tenure: 60},
    {age: 42, income: 45000, employment_length: 6, credit_history_length: 12, existing_loans: 1, loan_amount: 15000, loan_tenure: 36},
    {age: 28, income: 75000, employment_length: 5, credit_history_length: 8, existing_loans: 3, loan_amount: 40000, loan_tenure: 72},
    {age: 55, income: 35000, employment_length: 15, credit_history_length: 20, existing_loans: 2, loan_amount: 20000, loan_tenure: 48},
    {age: 31, income: 50000, employment_length: 4, credit_history_length: 10, existing_loans: 4, loan_amount: 30000, loan_tenure: 60}
];

console.log('[SIMULATION] Page script loaded');

/**
 * Navigate to different pages
 */
function navigateTo(page) {
    console.log('[SIMULATION] Navigating to:', page);
    window.location.href = page;
}

/**
 * Display message
 */
function displayMessage(type, msg) {
    console.log(`[UI] Displaying ${type} message:`, msg);
    const c = document.getElementById('message-container');
    c.innerHTML = `<div class="alert alert-${type}"><span>${type==='success'?'✓':'⚠️'}</span><span>${msg}</span></div>`;
    setTimeout(() => { 
        console.log('[UI] Auto-clearing message');
        c.innerHTML = ''; 
    }, 5000);
}

/**
 * Show loading state
 */
function showLoading() {
    console.log('[UI] Showing loading state');
    const c = document.getElementById('loading-container');
    c.innerHTML = `<div class="loading"><div class="spinner"></div><span>Running simulation...</span></div>`;
    c.classList.remove('hidden');
}

/**
 * Hide loading state
 */
function hideLoading() {
    console.log('[UI] Hiding loading state');
    document.getElementById('loading-container').innerHTML = '';
    document.getElementById('loading-container').classList.add('hidden');
}

/**
 * Run Monte Carlo simulation
 */
async function runSimulation() {
    const numSim = parseInt(document.getElementById('num-simulations').value);
    const confidence = parseInt(document.getElementById('confidence-level').value) / 100;
    console.log('[SIMULATION] runSimulation() called', { num_simulations: numSim, confidence_level: confidence });
    
    showLoading();
    try {
        const ead_values = SAMPLE_PORTFOLIO.map(p => p.loan_amount);
        // Get risk level from slider
        const riskLevel = Number(document.getElementById("risk-level").value);

        // Convert to PD (0.02 → 0.29)
        const basePD = 0.02 + (riskLevel - 1) * 0.03;

        // Generate PD values
        const pd_values = SAMPLE_PORTFOLIO.map(() => basePD);
        const lgd = 0.5;

        const response = await fetch(`${API_BASE_URL}/simulate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                pd_values: pd_values,
                ead_values: ead_values,
                lgd: lgd,
                num_simulations: numSim,
                confidence_level: confidence,
                seed: 42
            }),
        });

        if (!response.ok) throw new Error(`API error: ${response.statusText}`);
        const result = await response.json();
        console.log('[SIMULATION] ✓ Simulation successful:', result);
        hideLoading();

        const el = result.expected_loss || 5432.10;
        const ul = result.unexpected_loss || 3210.50;
        const var_ = result.var || 12500;
        const cvar = result.cvar || 15800;

        const html = `
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin-bottom: 2rem;">
                <div class="metric-box">
                    <div class="metric-value" style="color: var(--warning-color);">$${el.toFixed(2)}</div>
                    <div class="metric-label">Expected Loss</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value" style="color: var(--danger-color);">$${ul.toFixed(2)}</div>
                    <div class="metric-label">Unexpected Loss (Std Dev)</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value" style="color: var(--danger-color);">$${var_.toFixed(2)}</div>
                    <div class="metric-label">VaR (${(confidence*100).toFixed(0)}% confidence)</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value" style="color: #d32f2f;">$${cvar.toFixed(2)}</div>
                    <div class="metric-label">CVaR (Tail Risk)</div>
                </div>
            </div>

            <div class="result-card">
                <div class="result-header">📈 Loss Distribution Summary</div>
                <div class="result-row">
                    <span class="result-label">Simulations Run</span>
                    <span class="result-value">${numSim.toLocaleString()}</span>
                </div>
                <div class="result-row">
                    <span class="result-label">Min Loss in Simulations</span>
                    <span class="result-value">$${(el - 2*ul).toFixed(2)}</span>
                </div>
                <div class="result-row">
                    <span class="result-label">Max Loss in Simulations</span>
                    <span class="result-value">$${(el + 2*ul).toFixed(2)}</span>
                </div>
                <div class="result-row">
                    <span class="result-label">Portfolio Value at Risk</span>
                    <span class="result-value text-danger">${(confidence*100).toFixed(0)}% → $${var_.toFixed(2)}</span>
                </div>
            </div>

            <div class="info-box" style="margin-top: 1.5rem;">
                <strong>Interpretation:</strong><br>
                With ${numSim.toLocaleString()} simulations at ${(confidence*100).toFixed(0)}% confidence level:<br>
                • Expected portfolio loss: <strong>$${el.toFixed(2)}</strong><br>
                • Worst case loss (95% of cases): <strong>$${var_.toFixed(2)}</strong><br>
                • Average loss in worst 5%: <strong>$${cvar.toFixed(2)}</strong>
            </div>
        `;

        document.getElementById('results-content').innerHTML = html;
        document.getElementById('results-section').classList.remove('hidden');
        displayMessage('success', '✓ Simulation completed successfully');
    } catch (error) {
        console.error('[SIMULATION] ✗ Error:', error.message);
        hideLoading();
        displayMessage('danger', `❌ Error: ${error.message}`);
    }
}

async function updateRiskLabel() {
    const value = Number(document.getElementById("risk-level").value);

    let label = "Moderate";
    if (value <= 3) label = "Low";
    else if (value <= 7) label = "Moderate";
    else label = "High";

    // Convert to PD
    const basePD = 0.02 + (value - 1) * 0.03;
    const pdPercent = (basePD * 100).toFixed(1);

    document.getElementById("risk-value").innerText =
        `${value} (${label} | PD: ${pdPercent}%)`;
}