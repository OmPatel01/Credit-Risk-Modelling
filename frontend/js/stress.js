/**
 * STRESS TESTING PAGE — js/stress.js
 * Handles stress testing scenario analysis
 */

// const API_BASE_URL = 'http://localhost:8000';

const SAMPLE_PORTFOLIO = [
    {age: 35, income: 60000, employment_length: 8, credit_history_length: 15, existing_loans: 2, loan_amount: 25000, loan_tenure: 60},
    {age: 42, income: 45000, employment_length: 6, credit_history_length: 12, existing_loans: 1, loan_amount: 15000, loan_tenure: 36},
    {age: 28, income: 75000, employment_length: 5, credit_history_length: 8, existing_loans: 3, loan_amount: 40000, loan_tenure: 72},
    {age: 55, income: 35000, employment_length: 15, credit_history_length: 20, existing_loans: 2, loan_amount: 20000, loan_tenure: 48},
    {age: 31, income: 50000, employment_length: 4, credit_history_length: 10, existing_loans: 4, loan_amount: 30000, loan_tenure: 60}
];

console.log('[STRESS] Page script loaded');

/**
 * Navigate to different pages
 */
function navigateTo(page) {
    console.log('[STRESS] Navigating to:', page);
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
    c.innerHTML = `<div class="loading"><div class="spinner"></div><span>Running stress test...</span></div>`;
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
 * Run stress testing across multiple scenarios
 */
async function runStressTest() {
    console.log('[STRESS] runStressTest() called with portfolio_size:', SAMPLE_PORTFOLIO.length);
    showLoading();
    try {
        const ead_values = SAMPLE_PORTFOLIO.map(p => p.loan_amount);
        const pd_values = SAMPLE_PORTFOLIO.map(() => 0.1);
        const lgd = 0.5;

        const response = await fetch(`${API_BASE_URL}/stress-test`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                pd_values: pd_values,
                ead_values: ead_values,
                lgd: Number(lgd),
                run_simulation: true,
                num_simulations: Number(1000),
                seed: Number(42)
            }),
        });

        if (!response.ok) throw new Error(`API error: ${response.statusText}`);
        const result = await response.json();
        console.log('[STRESS] ✓ Stress test successful:', { scenarios: ['base_case', 'mild_stress', 'severe_stress'] });
        hideLoading();

        const scenarios = result.scenarios;

        const base = scenarios.find(s => s.scenario.toLowerCase().includes("base"));
        const mild = scenarios.find(s => s.scenario.toLowerCase().includes("mild"));
        const severe = scenarios.find(s => s.scenario.toLowerCase().includes("severe"));
        // const base = result.base_case || {total_pd: 0.108, total_ecl: 5432.10};
        // const mild = result.mild_stress || {total_pd: 0.162, total_ecl: 8148.15};
        // const severe = result.severe_stress || {total_pd: 0.324, total_ecl: 16296.30};

        if (!base || !mild || !severe) {
            throw new Error("Missing scenario data from backend");
        }
        const html = `
            <div class="comparison-container">
                <div class="comparison-card">
                    <div class="comparison-title">🟢 Base Case</div>
                    <div class="comparison-metric">
                        <div class="comparison-metric-label">Avg Portfolio PD</div>
                        <div class="comparison-metric-value">${(base.avg_pd * 100).toFixed(2)}%</div>
                    </div>
                    <div class="comparison-metric">
                        <div class="comparison-metric-label">Total ECL</div>
                        <div class="comparison-metric-value" style="color: var(--success-color);">$${base.total_ecl.toFixed(2)}</div>
                    </div>
                </div>

                <div class="comparison-card">
                    <div class="comparison-title">🟡 Mild Stress</div>
                    <div class="comparison-metric">
                        <div class="comparison-metric-label">Avg Portfolio PD</div>
                        <div class="comparison-metric-value">${(mild.avg_pd * 100).toFixed(2)}%</div>
                    </div>
                    <div class="comparison-metric">
                        <div class="comparison-metric-label">Total ECL</div>
                        <div class="comparison-metric-value" style="color: var(--warning-color);">$${mild.total_ecl.toFixed(2)}</div>
                    </div>
                    <div style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.5rem;">
                        +${((mild.total_ecl / base.total_ecl - 1) * 100).toFixed(1)}% vs Base
                    </div>
                </div>

                <div class="comparison-card">
                    <div class="comparison-title">🔴 Severe Stress</div>
                    <div class="comparison-metric">
                        <div class="comparison-metric-label">Avg Portfolio PD</div>
                        <div class="comparison-metric-value">${(severe.avg_pd * 100).toFixed(2)}%</div>
                    </div>
                    <div class="comparison-metric">
                        <div class="comparison-metric-label">Total ECL</div>
                        <div class="comparison-metric-value" style="color: var(--danger-color);">$${severe.total_ecl.toFixed(2)}</div>
                    </div>
                    <div style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.5rem;">
                        +${((severe.total_ecl / base.total_ecl - 1) * 100).toFixed(1)}% vs Base
                    </div>
                </div>
            </div>

            <div class="table-wrapper" style="margin-top: 2rem;">
                <table>
                    <thead>
                        <tr><th>Scenario</th><th>Avg PD</th><th>Total ECL</th><th>vs Base</th></tr>
                    </thead>
                    <tbody>
                        <tr><td>🟢 Base Case</td><td>${(base.avg_pd * 100).toFixed(2)}%</td><td>$${base.total_ecl.toFixed(2)}</td><td>Baseline</td></tr>
                        <tr><td>🟡 Mild Stress</td><td>${(mild.avg_pd * 100).toFixed(2)}%</td><td>$${mild.total_ecl.toFixed(2)}</td><td>+${((mild.total_ecl / base.total_ecl - 1) * 100).toFixed(1)}%</td></tr>
                        <tr><td>🔴 Severe Stress</td><td>${(severe.avg_pd * 100).toFixed(2)}%</td><td>$${severe.total_ecl.toFixed(2)}</td><td>+${((severe.total_ecl / base.total_ecl - 1) * 100).toFixed(1)}%</td></tr>
                    </tbody>
                </table>
            </div>

            <div class="info-box" style="margin-top: 1.5rem;">
                <strong>Key Findings:</strong><br>
                In a severe stress scenario, portfolio losses could increase by <strong>${((severe.total_ecl / base.total_ecl - 1) * 100).toFixed(1)}%</strong>.
                This highlights the importance of maintaining adequate capital buffers.
            </div>
        `;

        document.getElementById('results-content').innerHTML = html;
        document.getElementById('results-section').classList.remove('hidden');
        renderStressChart(base, mild, severe);
        displayMessage('success', '✓ Stress testing completed');
    } catch (error) {
        console.error('[STRESS] ✗ Error:', error.message);
        hideLoading();
        displayMessage('danger', `❌ Error: ${error.message}`);
    }
}

function renderStressChart(base, mild, severe) {
    const canvas = document.getElementById('stressChart');

    if (!canvas) {
        console.error("Chart canvas not found!");
        return;
    }

    const ctx = canvas.getContext('2d');

    if (window.stressChartInstance) {
        window.stressChartInstance.destroy();
    }

    window.stressChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Base Case', 'Mild Stress', 'Severe Stress'],
            datasets: [{
                label: 'Total ECL ($)',
                data: [
                    base.total_ecl,
                    mild.total_ecl,
                    severe.total_ecl
                ],
                borderWidth: 2
            }]
        }
    });
}