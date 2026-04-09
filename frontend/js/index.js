/**
 * Dashboard Page Script
 * Handles navigation and page initialization for the credit risk analytics dashboard
 */

// const API_BASE_URL = 'http://localhost:8000';

console.log('[INDEX] Dashboard page script loaded');

/**
 * Navigate to a different page
 * @param {string} page - The page filename to navigate to
 */
function navigateTo(page) {
    console.log('[INDEX] Navigation triggered:', page);
    window.location.href = page;
}

/**
 * Initialize dashboard on page load
 */
document.addEventListener('DOMContentLoaded', () => {
    console.log('[INDEX] DOM content loaded, dashboard ready');
    
    // Log available features
    const features = [
        'Prediction Engine (Scorecard & XGBoost)',
        'Risk Segmentation Analysis',
        'Expected Credit Loss Calculation',
        'Monte Carlo Simulation',
        'Stress Testing',
        'Sensitivity Analysis'
    ];
    
    console.log('[INDEX] Available features:', features);
});
