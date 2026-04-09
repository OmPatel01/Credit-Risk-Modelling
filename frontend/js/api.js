/**
 * ================================================================
 * api.js - Centralized API Integration Layer
 * ================================================================
 * 
 * RESPONSIBILITY:
 * - All API calls to FastAPI backend are made through this file
 * - Handles request/response formatting
 * - Centralized error handling
 * - Provides clean interface for UI pages
 * 
 * API Base URL Configuration
 */

// const API_BASE_URL = 'http://localhost:8000';

/**
 * ================================================================
 * PREDICTION APIs
 * ================================================================
 */

/**
 * Predict default probability using Scorecard (Champion Model)
 * @param {Object} clientInput - Borrower features object
 * @returns {Promise<Object>} - {credit_score, default_probability, risk_level, decision}
 */
async function predictScorecard(clientInput) {
  console.log('[SCORECARD] Starting prediction...', clientInput);
  try {
    console.log('[SCORECARD] Sending request to:', `${API_BASE_URL}/predict`);
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(clientInput),
    });

    if (!response.ok) {
      throw new Error(`Scorecard prediction failed: ${response.statusText}`);
    }

    const data = await response.json();
    console.log('[SCORECARD] ✓ Prediction successful:', data);
    return {
      success: true,
      data: data,
    };
  } catch (error) {
    console.error('[SCORECARD] ✗ Prediction error:', error.message);
    return {
      success: false,
      error: error.message,
    };
  }
}

/**
 * Predict default probability using XGBoost (Challenger Model)
 * @param {Object} clientInput - Borrower features object
 * @returns {Promise<Object>} - {default_probability, risk_level, decision}
 */
async function predictXGB(clientInput) {
  console.log('[XGB] Starting prediction...', clientInput);
  try {
    console.log('[XGB] Sending request to:', `${API_BASE_URL}/predict/xgb`);
    const response = await fetch(`${API_BASE_URL}/predict/xgb`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(clientInput),
    });

    if (!response.ok) {
      throw new Error(`XGBoost prediction failed: ${response.statusText}`);
    }

    const data = await response.json();
    console.log('[XGB] ✓ Prediction successful:', data);
    return {
      success: true,
      data: data,
    };
  } catch (error) {
    console.error('[XGB] ✗ Prediction error:', error.message);
    return {
      success: false,
      error: error.message,
    };
  }
}

/**
 * Predict using both models (Champion + Challenger)
 * @param {Object} clientInput - Borrower features object
 * @returns {Promise<Object>} - {scorecard, xgboost}
 */
async function predictBoth(clientInput) {
  console.log('[BOTH] Starting dual prediction...', clientInput);
  try {
    console.log('[BOTH] Sending request to:', `${API_BASE_URL}/predict/both`);
    const response = await fetch(`${API_BASE_URL}/predict/both`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(clientInput),
    });

    if (!response.ok) {
      throw new Error(`Combined prediction failed: ${response.statusText}`);
    }

    const data = await response.json();
    console.log('[BOTH] ✓ Dual prediction successful:', data);
    return {
      success: true,
      data: data,
    };
  } catch (error) {
    console.error('[BOTH] ✗ Prediction error:', error.message);
    return {
      success: false,
      error: error.message,
    };
  }
}

/**
 * ================================================================
 * SEGMENTATION APIs
 * ================================================================
 */

/**
 * Get risk segmentation for a portfolio
 * @param {Array} portfolio - List of client objects
 * @returns {Promise<Object>} - Segmentation results with buckets
 */
async function getSegmentation(portfolio) {
  console.log('[SEGMENTATION] Analyzing portfolio...', { portfolio_size: portfolio.length });
  try {
    console.log('[SEGMENTATION] Sending request to:', `${API_BASE_URL}/segmentation`);
    const response = await fetch(`${API_BASE_URL}/segmentation`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ portfolio: portfolio }),
    });

    if (!response.ok) {
      throw new Error(`Segmentation API failed: ${response.statusText}`);
    }

    const data = await response.json();
    console.log('[SEGMENTATION] ✓ Segmentation complete:', data);
    return {
      success: true,
      data: data,
    };
  } catch (error) {
    console.error('[SEGMENTATION] ✗ Error:', error.message);
    return {
      success: false,
      error: error.message,
    };
  }
}

/**
 * ================================================================
 * EXPECTED CREDIT LOSS (ECL) APIs
 * ================================================================
 */

/**
 * Calculate ECL for a portfolio
 * @param {Array} portfolio - List of client objects with PD, LGD, EAD
 * @returns {Promise<Object>} - {total_ecl, bucket_ecl, portfolio_summary}
 */
async function getECL(portfolio) {
  console.log('[ECL] Calculating expected credit loss...', { portfolio_size: portfolio.length });
  try {
    console.log('[ECL] Sending request to:', `${API_BASE_URL}/ecl`);
    const response = await fetch(`${API_BASE_URL}/ecl`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ portfolio: portfolio }),
    });

    if (!response.ok) {
      throw new Error(`ECL calculation failed: ${response.statusText}`);
    }

    const data = await response.json();
    console.log('[ECL] ✓ ECL calculation complete:', { total_ecl: data.total_ecl });
    return {
      success: true,
      data: data,
    };
  } catch (error) {
    console.error('[ECL] ✗ Error:', error.message);
    return {
      success: false,
      error: error.message,
    };
  }
}

/**
 * ================================================================
 * MONTE CARLO SIMULATION APIs
 * ================================================================
 */

/**
 * Run Monte Carlo simulation on portfolio
 * @param {Array} portfolio - List of clients
 * @param {Number} num_simulations - Number of simulations (default: 10000)
 * @param {Number} confidence_level - Confidence level for VaR (default: 0.95)
 * @returns {Promise<Object>} - {expected_loss, unexpected_loss, var, cvar, loss_distribution}
 */
async function runSimulation(portfolio, num_simulations = 10000, confidence_level = 0.95) {
  console.log('[SIMULATION] Starting Monte Carlo...', { num_simulations, confidence_level, portfolio_size: portfolio.length });
  try {
    console.log('[SIMULATION] Sending request to:', `${API_BASE_URL}/simulation`);
    const response = await fetch(`${API_BASE_URL}/simulation`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        portfolio: portfolio,
        num_simulations: num_simulations,
        confidence_level: confidence_level,
      }),
    });

    if (!response.ok) {
      throw new Error(`Simulation failed: ${response.statusText}`);
    }

    const data = await response.json();
    console.log('[SIMULATION] ✓ Simulation complete:', { expected_loss: data.expected_loss, var: data.var });
    return {
      success: true,
      data: data,
    };
  } catch (error) {
    console.error('[SIMULATION] ✗ Error:', error.message);
    return {
      success: false,
      error: error.message,
    };
  }
}

/**
 * ================================================================
 * STRESS TESTING APIs
 * ================================================================
 */

/**
 * Run stress testing on portfolio under different scenarios
 * @param {Array} portfolio - List of clients
 * @returns {Promise<Object>} - {base_case, mild_stress, severe_stress, comparison}
 */
async function runStressTesting(portfolio) {
  console.log('[STRESS] Running stress testing...', { portfolio_size: portfolio.length });
  try {
    console.log('[STRESS] Sending request to:', `${API_BASE_URL}/stress`);
    const response = await fetch(`${API_BASE_URL}/stress`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ portfolio: portfolio }),
    });

    if (!response.ok) {
      throw new Error(`Stress testing failed: ${response.statusText}`);
    }

    const data = await response.json();
    console.log('[STRESS] ✓ Stress testing complete:', { scenarios: ['base_case', 'mild_stress', 'severe_stress'] });
    return {
      success: true,
      data: data,
    };
  } catch (error) {
    console.error('[STRESS] ✗ Error:', error.message);
    return {
      success: false,
      error: error.message,
    };
  }
}

/**
 * ================================================================
 * SENSITIVITY ANALYSIS APIs
 * ================================================================
 */

/**
 * Run sensitivity analysis on key risk drivers
 * @param {Array} portfolio - List of clients
 * @returns {Promise<Object>} - {sensitivity_table, primary_drivers}
 */
async function runSensitivityAnalysis(portfolio) {
  console.log('[SENSITIVITY] Running sensitivity analysis...', { portfolio_size: portfolio.length });
  try {
    console.log('[SENSITIVITY] Sending request to:', `${API_BASE_URL}/sensitivity`);
    const response = await fetch(`${API_BASE_URL}/sensitivity`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ portfolio: portfolio }),
    });

    if (!response.ok) {
      throw new Error(`Sensitivity analysis failed: ${response.statusText}`);
    }

    const data = await response.json();
    console.log('[SENSITIVITY] ✓ Sensitivity analysis complete:', { drivers: ['PD', 'LGD'] });
    return {
      success: true,
      data: data,
    };
  } catch (error) {
    console.error('[SENSITIVITY] ✗ Error:', error.message);
    return {
      success: false,
      error: error.message,
    };
  }
}

/**
 * ================================================================
 * HEALTH & METADATA APIs
 * ================================================================
 */

/**
 * Check if API is healthy and models are loaded
 * @returns {Promise<Object>} - {status, models, version}
 */
async function healthCheck() {
  console.log('[HEALTH] Checking API health...');
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
    });

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }

    const data = await response.json();
    console.log('[HEALTH] ✓ API is healthy:', data);
    return {
      success: true,
      data: data,
    };
  } catch (error) {
    console.error('[HEALTH] ✗ API error:', error.message);
    return {
      success: false,
      error: error.message,
    };
  }
}

/**
 * Get model metadata and information
 * @returns {Promise<Object>} - {model_type, target, features_used, metrics}
 */
async function getModelInfo() {
  console.log('[MODEL_INFO] Fetching model metadata...');
  try {
    const response = await fetch(`${API_BASE_URL}/model-info`, {
      method: 'GET',
    });

    if (!response.ok) {
      throw new Error(`Model info fetch failed: ${response.statusText}`);
    }

    const data = await response.json();
    console.log('[MODEL_INFO] ✓ Model info retrieved:', data);
    return {
      success: true,
      data: data,
    };
  } catch (error) {
    console.error('[MODEL_INFO] ✗ Error:', error.message);
    return {
      success: false,
      error: error.message,
    };
  }
}

/**
 * ================================================================
 * UTILITY FUNCTIONS
 * ================================================================
 */

/**
 * Display error message to user
 * @param {String} message - Error message
 * @param {String} elementId - ID of element to display error in
 */
function displayError(message, elementId = 'error-container') {
  console.warn('[UI] Displaying error message:', message, { elementId });
  const errorContainer = document.getElementById(elementId);
  if (errorContainer) {
    errorContainer.innerHTML = `
      <div class="alert alert-danger">
        <span>⚠️</span>
        <span>${message}</span>
      </div>
    `;
    errorContainer.classList.remove('hidden');
  }
}

/**
 * Clear error messages
 * @param {String} elementId - ID of element to clear
 */
function clearError(elementId = 'error-container') {
  console.log('[UI] Clearing error message', { elementId });
  const errorContainer = document.getElementById(elementId);
  if (errorContainer) {
    errorContainer.innerHTML = '';
    errorContainer.classList.add('hidden');
  }
}

/**
 * Display success message to user
 * @param {String} message - Success message
 * @param {String} elementId - ID of element to display message in
 */
function displaySuccess(message, elementId = 'success-container') {
  console.log('[UI] Displaying success message:', message, { elementId });
  const successContainer = document.getElementById(elementId);
  if (successContainer) {
    successContainer.innerHTML = `
      <div class="alert alert-success">
        <span>✓</span>
        <span>${message}</span>
      </div>
    `;
    successContainer.classList.remove('hidden');
  }
}

/**
 * Clear success messages
 * @param {String} elementId - ID of element to clear
 */
function clearSuccess(elementId = 'success-container') {
  console.log('[UI] Clearing success message', { elementId });
  const successContainer = document.getElementById(elementId);
  if (successContainer) {
    successContainer.innerHTML = '';
    successContainer.classList.add('hidden');
  }
}

/**
 * Show loading spinner
 * @param {String} elementId - ID of element to show spinner in
 */
function showLoading(elementId = 'loading-container') {
  console.log('[UI] Showing loading spinner', { elementId });
  const loadingContainer = document.getElementById(elementId);
  if (loadingContainer) {
    loadingContainer.innerHTML = `
      <div class="loading">
        <div class="spinner"></div>
        <span>Loading...</span>
      </div>
    `;
    loadingContainer.classList.remove('hidden');
  }
}

/**
 * Hide loading spinner
 * @param {String} elementId - ID of element to hide spinner from
 */
function hideLoading(elementId = 'loading-container') {
  console.log('[UI] Hiding loading spinner', { elementId });
  const loadingContainer = document.getElementById(elementId);
  if (loadingContainer) {
    loadingContainer.innerHTML = '';
    loadingContainer.classList.add('hidden');
  }
}

/**
 * Format number as percentage with decimals
 * @param {Number} value - Value to format
 * @param {Number} decimals - Number of decimal places (default: 2)
 * @returns {String} - Formatted percentage string
 */
function formatPercentage(value, decimals = 2) {
  if (value === null || value === undefined) return 'N/A';
  return (value * 100).toFixed(decimals) + '%';
}

/**
 * Format number as currency
 * @param {Number} value - Value to format
 * @param {String} currency - Currency code (default: 'USD')
 * @returns {String} - Formatted currency string
 */
function formatCurrency(value, currency = 'USD') {
  if (value === null || value === undefined) return 'N/A';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: currency,
  }).format(value);
}

/**
 * Format number with commas
 * @param {Number} value - Value to format
 * @param {Number} decimals - Number of decimal places
 * @returns {String} - Formatted number string
 */
function formatNumber(value, decimals = 0) {
  if (value === null || value === undefined) return 'N/A';
  return parseFloat(value).toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

/**
 * Get risk level classification based on PD
 * @param {Number} pd - Probability of default (0-1)
 * @returns {String} - Risk level: 'low', 'medium', or 'high'
 */
function getRiskLevel(pd) {
  if (pd < 0.05) return 'low';
  if (pd < 0.20) return 'medium';
  return 'high';
}

/**
 * Get risk label with styling
 * @param {String} riskLevel - Risk level: 'low', 'medium', or 'high'
 * @returns {String} - HTML string with risk badge
 */
function getRiskBadge(riskLevel) {
  const badges = {
    low: '<span class="risk-indicator risk-low">● Low Risk</span>',
    medium: '<span class="risk-indicator risk-medium">● Medium Risk</span>',
    high: '<span class="risk-indicator risk-high">● High Risk</span>',
  };
  return badges[riskLevel] || '';
}

/**
 * Get decision badge with styling
 * @param {String} decision - Decision: 'APPROVE', 'REJECT', or 'REVIEW'
 * @returns {String} - HTML string with decision badge
 */
function getDecisionBadge(decision) {
  const badges = {
    'APPROVE': '<span class="decision-badge decision-approve">✓ APPROVE</span>',
    'REJECT': '<span class="decision-badge decision-reject">✗ REJECT</span>',
    'REVIEW': '<span class="decision-badge decision-review">⚠ REVIEW</span>',
  };
  return badges[decision] || '';
}

/**
 * Validate borrower input form
 * @param {Object} formData - Form data object
 * @returns {Object} - {isValid: boolean, errors: array}
 */
function validateBorrowerInput(formData) {
  const errors = [];
  
  // Check required fields
  const requiredFields = [
    'age', 'income', 'employment_length', 'credit_history_length',
    'existing_loans', 'loan_amount', 'loan_tenure'
  ];
  
  for (const field of requiredFields) {
    if (formData[field] === null || formData[field] === undefined || formData[field] === '') {
      errors.push(`${field.replace(/_/g, ' ')} is required`);
    }
  }
  
  // Validate numeric fields
  if (formData.age && (isNaN(formData.age) || formData.age < 18 || formData.age > 100)) {
    errors.push('Age must be between 18 and 100');
  }
  
  if (formData.income && (isNaN(formData.income) || formData.income < 0)) {
    errors.push('Income must be a positive number');
  }
  
  if (formData.loan_amount && (isNaN(formData.loan_amount) || formData.loan_amount < 0)) {
    errors.push('Loan amount must be a positive number');
  }
  
  return {
    isValid: errors.length === 0,
    errors: errors,
  };
}

/**
 * ================================================================
 * EXPORT FOR USE IN OTHER MODULES
 * ================================================================
 */

// Prediction functions
export {
  predictScorecard,
  predictXGB,
  predictBoth,
};

// Segmentation and ECL
export {
  getSegmentation,
  getECL,
};

// Simulation and Stress
export {
  runSimulation,
  runStressTesting,
};

// Sensitivity
export {
  runSensitivityAnalysis,
};

// Health and Metadata
export {
  healthCheck,
  getModelInfo,
};

// Utility functions
export {
  displayError,
  clearError,
  displaySuccess,
  clearSuccess,
  showLoading,
  hideLoading,
  formatPercentage,
  formatCurrency,
  formatNumber,
  getRiskLevel,
  getRiskBadge,
  getDecisionBadge,
  validateBorrowerInput,
};
