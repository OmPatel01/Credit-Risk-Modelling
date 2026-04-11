"""
app/schemas/risk_schemas.py
----------------------------
Pydantic request and response models for all five risk analytics endpoints.

Each endpoint pair is grouped as (Request, Response) in order of feature complexity:
    1. Segmentation   — simplest; only PD values required
    2. ECL            — adds LGD and EAD; optional segment labels for breakdown
    3. Simulation     — adds simulation count, confidence level, random seed
    4. Stress Test    — uses predefined scenarios; optional Monte Carlo overlay
    5. Sensitivity    — caller-supplied shift arrays; optional Monte Carlo overlay

Shared design rules across all schemas:
    - pd_values always validated to [0, 1] range via a field_validator
    - ead_values always validated to be non-negative (losses cannot be negative)
    - Optional simulation-related fields always default to sensible values
      so callers can omit them for simple use cases
    - Response models use Optional fields where a value may legitimately be absent
      (e.g. segment_ecl is None when no segment labels were provided)
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict


# ═══════════════════════════════════════════════════════════════════
# 1. RISK SEGMENTATION
# ═══════════════════════════════════════════════════════════════════

class SegmentationRequest(BaseModel):
    """Input for assigning borrowers to risk buckets based on their PD values."""

    pd_values: List[float] = Field(
        ...,
        description="Array of PD values, each in [0, 1].",
        min_length=1,
    )
    method: str = Field(
        default="quantile",
        description="'quantile' = equal-sized buckets; 'fixed' = user-defined PD thresholds.",
    )
    num_quantiles: Optional[int] = Field(
        default=5,
        description="Number of quantile buckets (used when method='quantile').",
        ge=2, le=20,
    )
    thresholds: Optional[List[float]] = Field(
        default=None,
        description=(
            "PD cut-points for fixed segmentation. "
            "[0.05, 0.15, 0.30, 0.50] creates 5 buckets: <5%, 5–15%, 15–30%, 30–50%, >50%."
        ),
    )
    labels: Optional[List[str]] = Field(
        default=None,
        description="Custom bucket names. Must have len(thresholds)+1 or num_quantiles items.",
    )

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Normalise to lowercase and reject any method string other than the two supported values."""
        v = v.lower().strip()
        if v not in ("quantile", "fixed"):
            raise ValueError("method must be 'quantile' or 'fixed'")
        return v

    @field_validator("pd_values")
    @classmethod
    def validate_pd_range(cls, v: List[float]) -> List[float]:
        """Reject any PD outside [0, 1] — probabilities must be valid."""
        for val in v:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"PD values must be in [0, 1]. Got {val}")
        return v


class BucketSummary(BaseModel):
    """Aggregate statistics for a single risk bucket in the segmentation response."""
    bucket: str    # bucket label (e.g. "A", "B", or custom name)
    count:  int    # number of borrowers in this bucket
    avg_pd: float  # mean PD within the bucket — characterises the typical borrower risk level


class SegmentationResponse(BaseModel):
    """Output from risk segmentation."""
    bucket_assignments: List[str]       # parallel to input pd_values; index i = borrower i's bucket
    summary: List[BucketSummary]        # one entry per distinct bucket, sorted by label


# ═══════════════════════════════════════════════════════════════════
# 2. EXPECTED CREDIT LOSS (ECL)
# ═══════════════════════════════════════════════════════════════════

class ECLRequest(BaseModel):
    """Input for ECL computation: ECL = PD × LGD × EAD per borrower."""

    pd_values: List[float] = Field(
        ..., description="Array of PD values, each in [0, 1].", min_length=1,
    )
    lgd: float = Field(
        ..., description="Loss Given Default — fraction of EAD lost if default occurs.", ge=0, le=1,
    )
    ead_values: List[float] = Field(
        ..., description="Exposure at Default — outstanding balance per borrower.", min_length=1,
    )
    segment_labels: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional segment label per borrower for grouped ECL breakdown "
            "(e.g. risk bucket or product type). Must be same length as pd_values."
        ),
    )

    @field_validator("pd_values")
    @classmethod
    def validate_pd_range(cls, v: List[float]) -> List[float]:
        for val in v:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"PD values must be in [0, 1]. Got {val}")
        return v

    @field_validator("ead_values")
    @classmethod
    def validate_ead_positive(cls, v: List[float]) -> List[float]:
        """EAD represents an outstanding balance — it cannot be negative."""
        for val in v:
            if val < 0:
                raise ValueError(f"EAD values must be non-negative. Got {val}")
        return v


class SegmentECL(BaseModel):
    """ECL breakdown for one segment — returned only when segment_labels were provided."""
    segment:   str
    count:     int
    total_ecl: float
    avg_ecl:   float


class ECLResponse(BaseModel):
    """Output from ECL computation."""
    individual_ecl: List[float]              # parallel to input; ECL per borrower
    total_ecl: float                         # sum of all individual ECLs — the portfolio provision amount
    mean_ecl:  float                         # average per borrower — useful for portfolio sizing
    segment_ecl: Optional[List[SegmentECL]] = None  # None when no segment_labels were provided


# ═══════════════════════════════════════════════════════════════════
# 3. MONTE CARLO SIMULATION
# ═══════════════════════════════════════════════════════════════════

class SimulationRequest(BaseModel):
    """Input for Monte Carlo portfolio loss simulation."""

    pd_values: List[float] = Field(
        ..., description="Array of PD values, each in [0, 1].", min_length=1,
    )
    lgd: float = Field(
        ..., description="Loss Given Default (scalar applied to all borrowers).", ge=0, le=1,
    )
    ead_values: List[float] = Field(
        ..., description="Exposure at Default per borrower.", min_length=1,
    )
    num_simulations: int = Field(
        default=10_000,
        description="Number of Monte Carlo scenarios. More = more accurate tail estimates but slower.",
        ge=100, le=500_000,
    )
    confidence_level: float = Field(
        default=0.95,
        description="Confidence level for VaR and CVaR (e.g. 0.95 = 95th percentile).",
        gt=0, lt=1,
    )
    seed: Optional[int] = Field(
        default=42,
        description="Random seed for reproducibility. Set to None for a fresh run each time.",
    )

    @field_validator("pd_values")
    @classmethod
    def validate_pd_range(cls, v: List[float]) -> List[float]:
        for val in v:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"PD values must be in [0, 1]. Got {val}")
        return v


class SimulationResponse(BaseModel):
    """Output from Monte Carlo simulation."""
    num_simulations:  int
    expected_loss:    float = Field(..., description="Mean simulated portfolio loss — approximates deterministic ECL")
    unexpected_loss:  float = Field(..., description="Standard deviation of simulated losses — measures volatility")
    var:              float = Field(..., description="Value at Risk — loss not exceeded in confidence_level% of simulations")
    cvar:             float = Field(..., description="Conditional VaR (Expected Shortfall) — mean loss in the worst tail")
    confidence_level: float
    min_loss:         float
    max_loss:         float
    percentile_5:     float  # 5th percentile — optimistic scenario loss
    percentile_95:    float  # 95th percentile — pessimistic scenario loss (same as VaR at 0.95)


# ═══════════════════════════════════════════════════════════════════
# 4. SCENARIO ANALYSIS (STRESS TESTING)
# ═══════════════════════════════════════════════════════════════════

class StressTestRequest(BaseModel):
    """
    Input for stress testing.

    The available scenarios (Base / Mild / Severe) and their PD multipliers
    are defined in services/risk_config.py. Callers cannot define custom
    scenarios via this API — only supply the portfolio data to be stressed.
    """

    pd_values: List[float] = Field(
        ..., description="Array of PD values, each in [0, 1].", min_length=1,
    )
    lgd: float = Field(
        ..., description="Baseline LGD — may be overridden per scenario by the config.", ge=0, le=1,
    )
    ead_values: List[float] = Field(
        ..., description="Array of EAD values.", min_length=1,
    )
    run_simulation: bool = Field(
        default=False,
        description="If true, also runs Monte Carlo under each scenario to expose tail risk.",
    )
    num_simulations: int = Field(
        default=10_000,
        description="Number of simulations per scenario if run_simulation is true.",
        ge=100, le=500_000,
    )
    seed: Optional[int] = Field(default=42)

    @field_validator("pd_values")
    @classmethod
    def validate_pd_range(cls, v: List[float]) -> List[float]:
        for val in v:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"PD values must be in [0, 1]. Got {val}")
        return v


class ScenarioResult(BaseModel):
    """Results for a single stress scenario."""
    scenario:        str
    pd_multiplier:   float
    lgd_used:        float
    total_ecl:       float
    mean_ecl:        float
    avg_pd:          float
    ecl_change_pct:  Optional[float] = Field(
        None, description="% ECL change vs Base scenario. None for the Base scenario itself."
    )
    simulation: Optional[SimulationResponse] = None  # None when run_simulation=False


class StressTestResponse(BaseModel):
    """Output from stress testing — one ScenarioResult per scenario in risk_config.STRESS_SCENARIOS."""
    scenarios: List[ScenarioResult]


# ═══════════════════════════════════════════════════════════════════
# 5. SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════

class SensitivityRequest(BaseModel):
    """
    Input for sensitivity analysis.

    Unlike stress testing (fixed scenarios), sensitivity accepts custom shift arrays
    so callers can explore any range of PD/LGD perturbations.
    When pd_shifts or lgd_shifts are omitted, defaults from risk_config.py are used.
    """

    pd_values: List[float] = Field(
        ..., description="Array of PD values, each in [0, 1].", min_length=1,
    )
    lgd: float = Field(
        ..., description="Baseline LGD.", ge=0, le=1,
    )
    ead_values: List[float] = Field(
        ..., description="Array of EAD values.", min_length=1,
    )
    pd_shifts: Optional[List[float]] = Field(
        default=None,
        description=(
            "Relative PD shifts as fractions. "
            "[-0.20, -0.10, 0.10, 0.20] tests ±10% and ±20% PD perturbations."
        ),
    )
    lgd_shifts: Optional[List[float]] = Field(
        default=None,
        description=(
            "Absolute LGD shifts in percentage points. "
            "[-0.10, 0.10] tests LGD dropping or rising by 10pp."
        ),
    )
    run_simulation: bool = Field(
        default=False,
        description="If true, also runs Monte Carlo for each shift scenario.",
    )
    num_simulations: int = Field(default=10_000, ge=100, le=500_000)
    seed: Optional[int] = Field(default=42)

    @field_validator("pd_values")
    @classmethod
    def validate_pd_range(cls, v: List[float]) -> List[float]:
        for val in v:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"PD values must be in [0, 1]. Got {val}")
        return v


class SensitivityResult(BaseModel):
    """ECL impact for a single PD or LGD shift."""
    driver:          str    # "PD" or "LGD"
    shift:           float  # raw shift value (e.g. 0.10)
    shift_label:     str    # human-readable label (e.g. "10%") for frontend display
    total_ecl:       float
    ecl_change_pct:  float  # % change vs unshifted baseline ECL
    simulation: Optional[SimulationResponse] = None  # None when run_simulation=False


class SensitivityResponse(BaseModel):
    """Output from sensitivity analysis."""
    baseline_ecl: float              # ECL with unmodified PD and LGD — all results are relative to this
    results: List[SensitivityResult] # PD shift results first, then LGD shift results