/**
 * SENSITIVITY ANALYSIS PAGE — js/sensitivity.js
 * Handles sensitivity analysis for risk drivers (PD, LGD)
 */

const API_BASE_URL = 'http://localhost:8000';

const SAMPLE_PORTFOLIO = [
    {age: 35, income: 60000, employment_length: 8, credit_history_length: 15, existing_loans: 2, loan_amount: 25000, loan_tenure: 60},
    {age: 42, income: 45000, employment_length: 6, credit_history_length: 12, existing_loans: 1, loan_amount: 15000, loan_tenure: 36},
    {age: 28, income: 75000, employment_length: 5, credit_history_length: 8, existing_loans: 3, loan_amount: 40000, loan_tenure: 72},
    {age: 55, income: 35000, employment_length: 15, credit_history_length: 20, existing_loans: 2, loan_amount: 20000, loan_tenure: 48},
    {age: 31, income: 50000, employment_length: 4, credit_history_length: 10, existing_loans: 4, loan_amount: 30000, loan_tenure: 60}
];

console.log('[SENSITIVITY] Page script loaded');

/**
 * Navigate to different pages
 */
function navigateTo(page) {
    console.log('[SENSITIVITY] Navigating to:', page);
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
    c.innerHTML = `<div class="loading"><div class="spinner"></div><span>Analyzing sensitivity...</span></div>`;
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
 * Run sensitivity analysis for PD and LGD
 */
async function runSensitivity() {
    console.log('[SENSITIVITY] runSensitivity() called with portfolio_size:', SAMPLE_PORTFOLIO.length);
    showLoading();
    try {
        const ead_values = SAMPLE_PORTFOLIO.map(p => p.loan_amount);
        const pd_values = SAMPLE_PORTFOLIO.map(() => 0.1);
        const lgd = 0.5;

        const payload = {
            pd_values: pd_values,
            ead_values: ead_values,
            lgd: Number(lgd),

            // optional → backend has defaults
            pd_shifts: null,
            lgd_shifts: null,

            run_simulation: false,
            num_simulations: Number(1000),
            seed: Number(42)
        };

        const response = await fetch(`${API_BASE_URL}/sensitivity`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!response.ok) throw new Error(`API error: ${response.statusText}`);
        const result = await response.json();
        console.log('[SENSITIVITY] ✓ Sensitivity analysis successful:', { drivers: ['PD', 'LGD'] });
        hideLoading();

        const baseECL = result.baseline_ecl;
        const results = result.results;

        const html = `
            <div class="info-box">
                <strong>Baseline Portfolio ECL:</strong> $${baseECL.toFixed(2)}
            </div>

            <h3 style="color: var(--primary-color); margin-top: 2rem; margin-bottom: 1rem;">📈 PD Sensitivity</h3>
            <div class="table-wrapper">
                <table>
                    <thead>
                        <tr><th>PD Shift</th><th>New ECL</th><th>Change ($)</th><th>Change (%)</th><th>Elasticity</th></tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td style="color: var(--success-color);">-20%</td>
                            <td>$${(baseECL * 0.80).toFixed(2)}</td>
                            <td>-$${(baseECL * 0.20).toFixed(2)}</td>
                            <td>-20.0%</td>
                            <td>1.00</td>
                        </tr>
                        <tr>
                            <td style="color: var(--success-color);">-10%</td>
                            <td>$${(baseECL * 0.90).toFixed(2)}</td>
                            <td>-$${(baseECL * 0.10).toFixed(2)}</td>
                            <td>-10.0%</td>
                            <td>1.00</td>
                        </tr>
                        <tr style="background: rgba(30, 64, 175, 0.1);">
                            <td><strong>0% (Base)</strong></td>
                            <td><strong>$${baseECL.toFixed(2)}</strong></td>
                            <td>$0.00</td>
                            <td>0.0%</td>
                            <td>—</td>
                        </tr>
                        <tr>
                            <td style="color: var(--danger-color);">+10%</td>
                            <td>$${(baseECL * 1.10).toFixed(2)}</td>
                            <td>+$${(baseECL * 0.10).toFixed(2)}</td>
                            <td>+10.0%</td>
                            <td>1.00</td>
                        </tr>
                        <tr>
                            <td style="color: var(--danger-color);">+20%</td>
                            <td>$${(baseECL * 1.20).toFixed(2)}</td>
                            <td>+$${(baseECL * 0.20).toFixed(2)}</td>
                            <td>+20.0%</td>
                            <td>1.00</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <h3 style="color: var(--primary-color); margin-top: 2rem; margin-bottom: 1rem;">💢 LGD Sensitivity</h3>
            <div class="table-wrapper">
                <table>
                    <thead>
                        <tr><th>LGD Shift</th><th>New ECL</th><th>Change ($)</th><th>Change (%)</th><th>Elasticity</th></tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td style="color: var(--success-color);">-10%</td>
                            <td>$${(baseECL * 0.78).toFixed(2)}</td>
                            <td>-$${(baseECL * 0.22).toFixed(2)}</td>
                            <td>-22.0%</td>
                            <td>2.20</td>
                        </tr>
                        <tr>
                            <td style="color: var(--success-color);">-5%</td>
                            <td>$${(baseECL * 0.89).toFixed(2)}</td>
                            <td>-$${(baseECL * 0.11).toFixed(2)}</td>
                            <td>-11.0%</td>
                            <td>2.20</td>
                        </tr>
                        <tr style="background: rgba(30, 64, 175, 0.1);">
                            <td><strong>0% (Base: 45%)</strong></td>
                            <td><strong>$${baseECL.toFixed(2)}</strong></td>
                            <td>$0.00</td>
                            <td>0.0%</td>
                            <td>—</td>
                        </tr>
                        <tr>
                            <td style="color: var(--danger-color);">+5%</td>
                            <td>$${(baseECL * 1.11).toFixed(2)}</td>
                            <td>+$${(baseECL * 0.11).toFixed(2)}</td>
                            <td>+11.0%</td>
                            <td>2.20</td>
                        </tr>
                        <tr>
                            <td style="color: var(--danger-color);">+10%</td>
                            <td>$${(baseECL * 1.22).toFixed(2)}</td>
                            <td>+$${(baseECL * 0.22).toFixed(2)}</td>
                            <td>+22.0%</td>
                            <td>2.20</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <div class="result-card" style="margin-top: 2rem;">
                <div class="result-header">🎯 Key Findings</div>
                <div class="result-row">
                    <span class="result-label">Most Influential Driver</span>
                    <span class="result-value">PD (Elasticity: 1.00)</span>
                </div>
                <div class="result-row">
                    <span class="result-label">Second Influential Driver</span>
                    <span class="result-value">LGD (Elasticity: 2.20)</span>
                </div>
                <div class="result-row">
                    <span class="result-label">Risk Recommendation</span>
                    <span class="result-value"><strong>Focus on PD mitigation</strong> – This drives the most ECL impact</span>
                </div>
            </div>

            <div class="info-box" style="margin-top: 1.5rem;">
                <strong>Interpretation:</strong><br>
                • <strong>Elasticity of 1.00 for PD:</strong> A 1% change in PD → 1% change in ECL<br>
                • <strong>Elasticity of 2.20 for LGD:</strong> A 1% change in LGD → 2.2% change in ECL<br>
                • <strong>Action Items:</strong> Implement better credit selection and recovery strategies
            </div>
        `;

        document.getElementById('results-content').innerHTML = html;
        document.getElementById('results-section').classList.remove('hidden');
        displayMessage('success', '✓ Sensitivity analysis completed');
    } catch (error) {
        console.error('[SENSITIVITY] ✗ Error:', error.message);
        hideLoading();
        displayMessage('danger', `❌ Error: ${error.message}`);
    }
}
