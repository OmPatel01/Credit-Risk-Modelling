/**
 * ANALYTICS PAGE — js/analytics.js
 * Handles risk segmentation and ECL calculation
 */

const API_BASE_URL = 'http://localhost:8000';

// Sample portfolio
const SAMPLE_PORTFOLIO = [
    {age: 35, income: 60000, employment_length: 8, credit_history_length: 15, existing_loans: 2, loan_amount: 25000, loan_tenure: 60},
    {age: 42, income: 45000, employment_length: 6, credit_history_length: 12, existing_loans: 1, loan_amount: 15000, loan_tenure: 36},
    {age: 28, income: 75000, employment_length: 5, credit_history_length: 8, existing_loans: 3, loan_amount: 40000, loan_tenure: 72},
    {age: 55, income: 35000, employment_length: 15, credit_history_length: 20, existing_loans: 2, loan_amount: 20000, loan_tenure: 48},
    {age: 31, income: 50000, employment_length: 4, credit_history_length: 10, existing_loans: 4, loan_amount: 30000, loan_tenure: 60}
];

console.log('[ANALYTICS] Page script loaded');

/**
 * Navigate to different pages
 */
function navigateTo(page) {
    console.log('[ANALYTICS] Navigating to:', page);
    window.location.href = page;
}

/**
 * Display message
 */
function displayMessage(type, message) {
    console.log(`[UI] Displaying ${type} message:`, message);
    const container = document.getElementById('message-container');
    const alertClass = `alert-${type}`;
    const icon = type === 'success' ? '✓' : '⚠️';
    container.innerHTML = `<div class="alert ${alertClass}"><span>${icon}</span><span>${message}</span></div>`;
    setTimeout(() => { 
        console.log('[UI] Auto-clearing message');
        container.innerHTML = ''; 
    }, 5000);
}

/**
 * Show loading state
 */
function showLoading() {
    console.log('[UI] Showing loading state');
    document.getElementById('loading-container').innerHTML = `
        <div class="loading"><div class="spinner"></div><span>Processing...</span></div>
    `;
    document.getElementById('loading-container').classList.remove('hidden');
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
 * Get risk badge HTML
 */
function getRiskBadgeHTML(riskLevel) {
    const classes = {
        'Low': 'low',
        'Medium': 'medium',
        'High': 'high'
    };
    const cls = classes[riskLevel] || 'low';
    console.log('[FORMAT] Risk badge:', { riskLevel, className: cls });
    return `<span class="risk-indicator risk-${cls}">● ${riskLevel}</span>`;
}

/**
 * Analyze risk segmentation
 */
function analyzeSegmentation() {
    const rows = document.querySelectorAll('#portfolio-table-body tr');

    let low = 0, medium = 0, high = 0;

    rows.forEach(row => {
        const cells = row.querySelectorAll('td');
        const pd = parseFloat(cells[4].innerText);

        let segment = '';
        if (pd < 0.05) { segment = 'Low'; low++; }
        else if (pd <= 0.20) { segment = 'Medium'; medium++; }
        else { segment = 'High'; high++; }

        cells[5].innerText = segment;
    });

    document.getElementById('segmentation-content').innerHTML = `
        <div class="info-box">
            Low: ${low} | Medium: ${medium} | High: ${high}
        </div>
    `;

    document.getElementById('segmentation-section').classList.remove('hidden');
}

/**
 * Calculate ECL (Expected Credit Loss)
 */
async function calculateECL() {
    console.log('[ECL] Running calculation');

    const rows = document.querySelectorAll('#portfolio-table-body tr');

    const pd_values = [];
    const ead_values = [];
    const segment_labels = [];

    rows.forEach(row => {
        const cells = row.querySelectorAll('td');

        const ead = parseFloat(cells[3].innerText);
        const pd  = parseFloat(cells[4].innerText);

        pd_values.push(pd);
        ead_values.push(ead);

        // segmentation logic
        let segment = '';
        if (pd < 0.05) segment = 'Low';
        else if (pd <= 0.20) segment = 'Medium';
        else segment = 'High';

        segment_labels.push(segment);

        // update segment column
        cells[5].innerText = segment;
    });

    const lgd = parseFloat(document.getElementById('lgd-input').value);

    console.log('[ECL] Payload:', { pd_values, ead_values, lgd });

    try {
        const response = await fetch(`${API_BASE_URL}/ecl`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                pd_values,
                lgd,
                ead_values,
                segment_labels
            })
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail);
        }

        const result = await response.json();

        // update ECL in table
        rows.forEach((row, i) => {
            const cells = row.querySelectorAll('td');
            cells[5].innerText += ` | ECL: ${result.individual_ecl[i]}`;
        });

        document.getElementById('ecl-content').innerHTML = `
            <div class="metric-box">
                <div class="metric-value">$${result.total_ecl}</div>
                <div class="metric-label">Total ECL</div>
            </div>
        `;

        document.getElementById('ecl-section').classList.remove('hidden');

    } catch (error) {
        console.error('[ECL] Error:', error.message);
        alert(error.message);
    }
}