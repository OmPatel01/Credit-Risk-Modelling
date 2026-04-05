"""
risk_schemas.py
---------------
Pydantic models for credit risk analytics endpoints.

Covers:
  - Risk Segmentation
  - Expected Credit Loss (ECL)
  - Monte Carlo Simulation
  - Scenario Analysis (Stress Testing)
  - Sensitivity Analysis
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict


# ═══════════════════════════════════════════════════════════════════
# 1. Risk Segmentation
# ═══════════════════════════════════════════════════════════════════

class SegmentationRequest(BaseModel):
    """Input for risk segmentation."""
    pd_values: List[float] = Field(
        ...,
        description="Array of PD (probability of default) values, each in [0, 1].",
        min_length=1,
    )
    method: str = Field(
        default="quantile",
        description="Segmentation method: 'quantile' or 'fixed'.",
    )
    num_quantiles: Optional[int] = Field(
        default=5,
        description="Number of quantile buckets (used when method='quantile').",
        ge=2,
        le=20,
    )
    thresholds: Optional[List[float]] = Field(
        default=None,
        description=(
            "Custom PD thresholds for fixed segmentation. "
            "Example: [0.05, 0.15, 0.30, 0.50] creates 5 buckets."
        ),
    )
    labels: Optional[List[str]] = Field(
        default=None,
        description="Custom bucket labels (must have len(thresholds)+1 or num_quantiles items).",
    )

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in ("quantile", "fixed"):
            raise ValueError("method must be 'quantile' or 'fixed'")
        return v

    @field_validator("pd_values")
    @classmethod
    def validate_pd_range(cls, v: List[float]) -> List[float]:
        for val in v:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"PD values must be in [0, 1]. Got {val}")
        return v


class BucketSummary(BaseModel):
    """Summary statistics for a single risk bucket."""
    bucket: str
    count: int
    avg_pd: float


class SegmentationResponse(BaseModel):
    """Output from risk segmentation."""
    bucket_assignments: List[str]
    summary: List[BucketSummary]


# ═══════════════════════════════════════════════════════════════════
# 2. Expected Credit Loss (ECL)
# ═══════════════════════════════════════════════════════════════════

class ECLRequest(BaseModel):
    """Input for ECL calculation: ECL = PD × LGD × EAD."""
    pd_values: List[float] = Field(
        ..., description="Array of PD values, each in [0, 1].", min_length=1,
    )
    lgd: float = Field(
        ..., description="Loss Given Default (scalar, 0–1).", ge=0, le=1,
    )
    ead_values: List[float] = Field(
        ..., description="Array of Exposure at Default values.", min_length=1,
    )
    segment_labels: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional segment labels per borrower for grouped ECL breakdown. "
            "Must be same length as pd_values."
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
        for val in v:
            if val < 0:
                raise ValueError(f"EAD values must be non-negative. Got {val}")
        return v


class SegmentECL(BaseModel):
    """ECL breakdown for a single segment."""
    segment: str
    count: int
    total_ecl: float
    avg_ecl: float


class ECLResponse(BaseModel):
    """Output from ECL calculation."""
    individual_ecl: List[float]
    total_ecl: float
    mean_ecl: float
    segment_ecl: Optional[List[SegmentECL]] = None


# ═══════════════════════════════════════════════════════════════════
# 3. Monte Carlo Simulation
# ═══════════════════════════════════════════════════════════════════

class SimulationRequest(BaseModel):
    """Input for Monte Carlo loss simulation."""
    pd_values: List[float] = Field(
        ..., description="Array of PD values, each in [0, 1].", min_length=1,
    )
    lgd: float = Field(
        ..., description="Loss Given Default (scalar, 0–1).", ge=0, le=1,
    )
    ead_values: List[float] = Field(
        ..., description="Array of Exposure at Default values.", min_length=1,
    )
    num_simulations: int = Field(
        default=10_000,
        description="Number of Monte Carlo simulations.",
        ge=100,
        le=500_000,
    )
    confidence_level: float = Field(
        default=0.95,
        description="Confidence level for VaR and CVaR.",
        gt=0,
        lt=1,
    )
    seed: Optional[int] = Field(
        default=42, description="Random seed for reproducibility.",
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
    num_simulations: int
    expected_loss: float = Field(..., description="Mean simulated portfolio loss")
    unexpected_loss: float = Field(..., description="Std dev of simulated losses")
    var: float = Field(..., description="Value at Risk at given confidence level")
    cvar: float = Field(..., description="Conditional VaR (Expected Shortfall)")
    confidence_level: float
    min_loss: float
    max_loss: float
    percentile_5: float
    percentile_95: float


# ═══════════════════════════════════════════════════════════════════
# 4. Scenario Analysis (Stress Testing)
# ═══════════════════════════════════════════════════════════════════

class StressTestRequest(BaseModel):
    """Input for stress testing."""
    pd_values: List[float] = Field(
        ..., description="Array of PD values, each in [0, 1].", min_length=1,
    )
    lgd: float = Field(
        ..., description="Baseline Loss Given Default.", ge=0, le=1,
    )
    ead_values: List[float] = Field(
        ..., description="Array of Exposure at Default values.", min_length=1,
    )
    run_simulation: bool = Field(
        default=False,
        description="If true, also runs Monte Carlo under each scenario.",
    )
    num_simulations: int = Field(
        default=10_000,
        description="Number of simulations if run_simulation is true.",
        ge=100,
        le=500_000,
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
    scenario: str
    pd_multiplier: float
    lgd_used: float
    total_ecl: float
    mean_ecl: float
    ecl_change_pct: Optional[float] = Field(
        None, description="% change vs base scenario"
    )
    simulation: Optional[SimulationResponse] = None


class StressTestResponse(BaseModel):
    """Output from stress testing."""
    scenarios: List[ScenarioResult]


# ═══════════════════════════════════════════════════════════════════
# 5. Sensitivity Analysis
# ═══════════════════════════════════════════════════════════════════

class SensitivityRequest(BaseModel):
    """Input for sensitivity analysis."""
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
        description="PD multiplier shifts (e.g. [-0.20, -0.10, 0.10, 0.20]).",
    )
    lgd_shifts: Optional[List[float]] = Field(
        default=None,
        description="LGD absolute shifts (e.g. [-0.10, 0.10]).",
    )
    run_simulation: bool = Field(
        default=False,
        description="If true, also recomputes Monte Carlo metrics for each scenario.",
    )
    num_simulations: int = Field(
        default=10_000, ge=100, le=500_000,
    )
    seed: Optional[int] = Field(default=42)

    @field_validator("pd_values")
    @classmethod
    def validate_pd_range(cls, v: List[float]) -> List[float]:
        for val in v:
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"PD values must be in [0, 1]. Got {val}")
        return v


class SensitivityResult(BaseModel):
    """Result for a single sensitivity scenario."""
    driver: str
    shift: float
    shift_label: str
    total_ecl: float
    ecl_change_pct: float
    simulation: Optional[SimulationResponse] = None


class SensitivityResponse(BaseModel):
    """Output from sensitivity analysis."""
    baseline_ecl: float
    results: List[SensitivityResult]
