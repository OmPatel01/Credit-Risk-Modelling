/**
 * Dashboard Page Script
 *
 * CHANGES FROM BACKEND UPDATE:
 *   1. loadPortfolioKPIs() — new function. Fetches real metrics from
 *      GET /portfolio/summary and populates the KPI cards if they exist.
 *   2. DOMContentLoaded handler now calls loadPortfolioKPIs().
 *   3. Graceful degradation: if the endpoint returns 503 (training not run)
 *      or fails, KPI cards show a "—" placeholder and log a warning.
 *      No error is surfaced to the user — the page still works fully.
 */

// const API_BASE_URL = 'http://localhost:8000';

console.log('[INDEX] Dashboard page script loaded');

function navigateTo(page) {
    console.log('[INDEX] Navigation triggered:', page);
    window.location.href = page;
}

/**
 * Fetch portfolio KPIs and render them into dashboard metric cards.
 * Cards are identified by data-kpi attributes so this is safe even
 * if the HTML doesn't have the metric cards yet.
 */
async function loadPortfolioKPIs() {
    console.log('[INDEX] Loading portfolio KPIs...');
    try {
        const result = await getPortfolioSummary();

        if (!result.success) {
            console.warn('[INDEX] Portfolio KPIs unavailable:', result.error);
            return; // silently degrade — don't block the page
        }

        const kpi = result.data;
        console.log('[INDEX] Portfolio KPIs loaded:', kpi);

        // Populate any metric elements tagged with data-kpi attributes.
        // This is non-destructive: if elements don't exist, nothing happens.
        const mapping = {
            'kpi-auc':           kpi.auc    != null ? kpi.auc.toFixed(4)                       : null,
            'kpi-ks':            kpi.ks     != null ? kpi.ks.toFixed(4)                        : null,
            'kpi-gini':          kpi.gini   != null ? kpi.gini.toFixed(4)                      : null,
            'kpi-avg-pd':        kpi.avg_pd != null ? (kpi.avg_pd * 100).toFixed(1) + '%'      : null,
            'kpi-total-ecl':     kpi.total_ecl != null
                                    ? 'NT$' + kpi.total_ecl.toLocaleString('en-US', { maximumFractionDigits: 0 })
                                    : null,
            'kpi-high-risk':     kpi.pct_high_risk   != null ? (kpi.pct_high_risk * 100).toFixed(1) + '%'   : null,
            'kpi-medium-risk':   kpi.pct_medium_risk != null ? (kpi.pct_medium_risk * 100).toFixed(1) + '%' : null,
            'kpi-low-risk':      kpi.pct_low_risk    != null ? (kpi.pct_low_risk * 100).toFixed(1) + '%'    : null,
            'kpi-borrowers':     kpi.total_borrowers != null ? kpi.total_borrowers.toLocaleString()         : null,
        };

        for (const [id, value] of Object.entries(mapping)) {
            if (value === null) continue;
            const el = document.getElementById(id);
            if (el) {
                el.textContent = value;
                console.log(`[INDEX] KPI populated: #${id} = ${value}`);
            }
        }

    } catch (err) {
        console.warn('[INDEX] Failed to load portfolio KPIs:', err.message);
        // No user-facing error — this is supplementary data
    }
}

document.addEventListener('DOMContentLoaded', () => {
    console.log('[INDEX] DOM content loaded, dashboard ready');

    const features = [
        'Prediction Engine (Scorecard & XGBoost)',
        'Risk Segmentation Analysis',
        'Expected Credit Loss Calculation',
        'Monte Carlo Simulation',
        'Stress Testing',
        'Sensitivity Analysis',
    ];
    console.log('[INDEX] Available features:', features);

    // Load real KPIs if the endpoint is available
    loadPortfolioKPIs();
});