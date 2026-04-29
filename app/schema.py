"""
app/schema.py
-------------
Pydantic models for all prediction endpoint request and response validation.

Pydantic enforces types, runs field validators, and auto-generates the JSON schema
that appears in the /docs Swagger UI — no manual validation code needed in route handlers.

Two input formats are defined here:
    ClientInput    — raw 21-column dataset format; maps directly to the original training data.
                     Used by /predict/* routes. Intended for technical integrations.
    BusinessInput  — 10-field business-friendly format; maps to ClientInput via input_mapper.
                     Used by /predict/business/* routes. Intended for frontend and non-technical users.

The separation exists because exposing raw PAY_X / BILL_AMTX fields to business users is
error-prone — the field semantics (e.g. PAY scale of -2 to 8) require dataset knowledge.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any

class ClientInput(BaseModel):
    """
    Raw 21-column client record in the original dataset format.

    Field values must match the dataset encoding exactly:
        PAY_X scale: -2 = no consumption, -1 = paid in full, 0 = revolving credit,
                      1 = payment delay 1 month, ..., 8 = payment delay 8+ months
        EDUCATION:   1=graduate school, 2=university, 3=high school, 4=others
        Amounts:     in New Taiwan Dollars (NT$)
    """

    LIMIT_BAL: float = Field(..., description="Credit limit amount (NT dollars)", example=50000)

    AGE:       int   = Field(..., description="Client age in years", example=35)
    EDUCATION: int   = Field(..., description="Education level (1-6)", example=2)

    # PAY_0 = most recent month (September); PAY_6 = oldest month (April)
    # Gaps in numbering (PAY_1 is missing) match the original dataset convention
    PAY_0: int = Field(..., description="Repayment status September 2005", example=0)
    PAY_2: int = Field(..., description="Repayment status August 2005",    example=0)
    PAY_3: int = Field(..., description="Repayment status July 2005",      example=0)
    PAY_4: int = Field(..., description="Repayment status June 2005",      example=0)
    PAY_5: int = Field(..., description="Repayment status May 2005",       example=0)
    PAY_6: int = Field(..., description="Repayment status April 2005",     example=0)

    # BILL_AMT1 = most recent; BILL_AMT6 = oldest (mirrors PAY ordering)
    BILL_AMT1: float = Field(..., description="Bill amount September 2005", example=20000)
    BILL_AMT2: float = Field(..., description="Bill amount August 2005",    example=19000)
    BILL_AMT3: float = Field(..., description="Bill amount July 2005",      example=18000)
    BILL_AMT4: float = Field(..., description="Bill amount June 2005",      example=17000)
    BILL_AMT5: float = Field(..., description="Bill amount May 2005",       example=16000)
    BILL_AMT6: float = Field(..., description="Bill amount April 2005",     example=15000)

    PAY_AMT1: float = Field(..., description="Payment amount September 2005", example=5000)
    PAY_AMT2: float = Field(..., description="Payment amount August 2005",    example=5000)
    PAY_AMT3: float = Field(..., description="Payment amount July 2005",      example=4000)
    PAY_AMT4: float = Field(..., description="Payment amount June 2005",      example=4000)
    PAY_AMT5: float = Field(..., description="Payment amount May 2005",       example=3000)
    PAY_AMT6: float = Field(..., description="Payment amount April 2005",     example=3000)

    class Config:
        json_schema_extra = {
            "example": {
                "LIMIT_BAL": 50000, "AGE": 35, "EDUCATION": 2,
                "PAY_0": 0, "PAY_2": 0, "PAY_3": 0,
                "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
                "BILL_AMT1": 20000, "BILL_AMT2": 19000, "BILL_AMT3": 18000,
                "BILL_AMT4": 17000, "BILL_AMT5": 16000, "BILL_AMT6": 15000,
                "PAY_AMT1": 5000, "PAY_AMT2": 5000, "PAY_AMT3": 4000,
                "PAY_AMT4": 4000, "PAY_AMT5": 3000, "PAY_AMT6": 3000,
            }
        }


# ── Nested models used inside ScorecardResponse ──────────────────────────────

class RiskDriver(BaseModel):
    """A single ranked risk driver returned in the scorecard response."""
    feature:      str    # raw feature name (e.g. "MAX_DELAY", "UTILIZATION")
    contribution: float  # signed coef × WOE value; positive = pushes toward default


# ── Response models ───────────────────────────────────────────────────────────

class ScorecardResponse(BaseModel):
    """
    Champion model response including interpretable credit score, risk band,
    per-feature WOE×coefficient contributions, per-feature score points,
    and the top 3 ranked risk drivers.
    """
    model:               str
    credit_score:        int    # additive points-based score (higher = lower risk, range ~576–906)
    default_probability: float  # P(default) from the LR model, rounded to 4 decimal places
    risk_level:          str    # human-readable risk band (e.g. "Moderate")
    decision:            str    # lending recommendation: Approve / Review / Decline

    # Per-feature breakdown — both fields are dicts keyed by raw feature name
    feature_contributions: Optional[Dict[str, float]] = None
    # coef × WOE per feature; positive value = feature is increasing default risk

    point_contributions: Optional[Dict[str, Any]] = None
    # integer score points assigned to each feature bin (from the scorecard table)
    # Also contains "__basepoints__" key for the intercept contribution

    top_risk_drivers: Optional[List[RiskDriver]] = None
    # Ranked list of the top 3 features most responsible for the borrower's risk


class XGBResponse(BaseModel):
    """Challenger model response — no credit score since XGBoost doesn't produce a points table."""
    model:               str
    default_probability: float
    risk_level:          str
    decision:            str


class CombinedResponse(BaseModel):
    """Both champion and challenger results in a single response for A/B comparison."""
    scorecard: ScorecardResponse
    xgboost:   XGBResponse


class BusinessInput(BaseModel):
    """
    User-friendly input format for the /predict/business/* routes.

    Uses business terminology rather than raw dataset column names. The input_mapper
    module converts these 10 fields deterministically into the 21 raw columns
    that the preprocessing pipeline expects.

    Repayment status encoding (same scale as raw PAY_X):
        -2 = no consumption
        -1 = paid in full
         0 = paid minimum / revolving credit
         1 = 1 month overdue
         ... up to 8
    """

    # ── Credit profile
    LIMIT_BAL: float = Field(..., description="Credit limit (NT dollars)", example=50000, ge=0)
    AGE:       int   = Field(..., description="Age in years", example=35, ge=18, le=100)
    EDUCATION: int   = Field(
        ...,
        description="Education level (1=graduate school, 2=university, 3=high school, 4=others)",
        example=2, ge=1, le=6
    )

    # ── Repayment behaviour
    recent_delay:   int   = Field(
        ...,
        description="Most recent payment status (-2 to 8)",
        example=0, ge=-2, le=8
    )
    avg_past_delay: float = Field(
        ...,
        description="Average payment status over previous 5 months (PAY_2..PAY_6)",
        example=0.0, ge=-2, le=8
    )
    num_delays:     int   = Field(
        ...,
        description="Number of months (out of PAY_2..PAY_6) with payment delays",
        example=0, ge=0, le=6
    )

    # ── Billing history
    avg_bill_amount:  float = Field(..., description="Average monthly bill (NT dollars)", example=15000, ge=0)
    bill_growth_rate: float = Field(
        default=0.0,
        description="Rate of change in bills from oldest to newest month (-1 to 1); negative = shrinking debt",
        example=0.0, ge=-1, le=1
    )

    # ── Payment amounts
    payment_amount:    float = Field(..., description="Typical monthly payment (NT dollars)", example=5000, ge=0)
    zero_payment_count: int  = Field(
        default=0,
        description="Number of months with zero payment (applied to oldest months first)",
        example=0, ge=0, le=6
    )

    class Config:
        json_schema_extra = {
            "example": {
                "LIMIT_BAL": 50000, "AGE": 35, "EDUCATION": 2,
                "recent_delay": 0, "avg_past_delay": 0.0, "num_delays": 0,
                "avg_bill_amount": 15000, "bill_growth_rate": 0.0,
                "payment_amount": 5000, "zero_payment_count": 0,
            }
        }


class WhatIfRequest(BaseModel):
    """Two BusinessInput scenarios for side-by-side scorecard comparison."""
    base_input:     BusinessInput  # current / actual applicant profile
    modified_input: BusinessInput  # hypothetical / improved profile to compare against


class WhatIfResponse(BaseModel):
    """
    Score and PD delta between two scenarios.

    Positive delta_score = score improved (lower risk).
    Negative delta_pd    = risk decreased (good).
    decision_flipped     = True when the lending decision changed between scenarios
                           (e.g. Decline → Approve, or Review → Approve).
    """
    base_score:  int
    new_score:   int
    delta_score: int    # new_score − base_score; positive = improvement

    base_pd:  float
    new_pd:   float
    delta_pd: float    # new_pd − base_pd; negative = risk decreased (good)

    base_decision:    str   # lending decision for the base scenario
    new_decision:     str   # lending decision for the modified scenario
    decision_flipped: bool  # True if the decision category changed


# ── Explainability request/response models ────────────────────────────────────

class ExplainRequest(BaseModel):
    """Input for the /explain endpoint — accepts a BusinessInput payload."""
    input: BusinessInput


class FeatureExplanation(BaseModel):
    """Explanation for a single feature."""
    feature:       str    # raw feature name
    woe_value:     float  # Weight of Evidence value for the borrower's bin
    coefficient:   float  # LR coefficient for this feature
    contribution:  float  # coef × WOE — signed contribution to log-odds
    score_points:  int    # integer points this feature contributes to the credit score
    risk_direction: str   # "increases_risk" | "decreases_risk" | "neutral"


class ExplainResponse(BaseModel):
    """Full feature-level explanation of a scorecard prediction."""
    credit_score:        int
    default_probability: float
    risk_level:          str
    decision:            str
    base_points:         int                    # scorecard intercept points
    feature_explanations: List[FeatureExplanation]  # one entry per scorecard feature, sorted by contribution desc
    top_risk_drivers:    List[RiskDriver]            # top 3 features pushing toward default


# ── Recommendation request/response models ────────────────────────────────────

class RecommendRequest(BaseModel):
    """Input for the /recommend endpoint."""
    input: BusinessInput


class Recommendation(BaseModel):
    """A single actionable recommendation for one risk driver."""
    feature:        str   # raw feature name
    current_value:  str   # human-readable description of the borrower's current value
    action:         str   # what the borrower should do (e.g. "Reduce credit utilisation below 50%")
    expected_impact: str  # qualitative impact (e.g. "High", "Medium", "Low")
    priority:       int   # 1 = most impactful, ascending


class RecommendResponse(BaseModel):
    """Actionable improvement recommendations based on the top risk drivers."""
    credit_score:        int
    default_probability: float
    risk_level:          str
    decision:            str
    recommendations:     List[Recommendation]


# ── Portfolio KPI response model ──────────────────────────────────────────────

class PortfolioSummaryResponse(BaseModel):
    """
    Aggregate portfolio KPIs read from stored model metadata.
    Returned by /portfolio/summary.
    """
    avg_pd:           float   # average probability of default across training/test set
    pct_high_risk:    float   # % of borrowers classified as high risk (PD > 0.20)
    pct_medium_risk:  float   # % of borrowers classified as medium risk (0.05 <= PD <= 0.20)
    pct_low_risk:     float   # % of borrowers classified as low risk (PD < 0.05)
    total_ecl:        float   # total expected credit loss (using stored PDs and default LGD)
    auc:              float   # model AUC from training evaluation
    ks:               float   # KS statistic from training evaluation
    gini:             float   # Gini coefficient (2×AUC − 1)
    total_borrowers:  int     # total number of records in the evaluation set