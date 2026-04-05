"""
schema.py
---------
Pydantic models for request and response validation.

Pydantic automatically:
- Validates data types
- Returns clear error messages for missing/wrong fields
- Generates API documentation (visible at /docs)
"""

from pydantic import BaseModel, Field
from typing import Optional


class ClientInput(BaseModel):
    """
    Raw input fields for a credit card client.
    These match the original dataset columns before feature engineering.
    FastAPI validates this automatically before it reaches your predict function.
    """

    # ── Credit limit ─────────────────────────────────────────────
    LIMIT_BAL: float = Field(..., description="Credit limit amount (NT dollars)", example=50000)

    # ── Demographics ─────────────────────────────────────────────
    AGE:       int   = Field(..., description="Client age in years", example=35)

    # ── Repayment status (PAY scale: -2 to 8) ────────────────────
    PAY_0: int = Field(..., description="Repayment status September 2005", example=0)
    PAY_2: int = Field(..., description="Repayment status August 2005",    example=0)
    PAY_3: int = Field(..., description="Repayment status July 2005",      example=0)
    PAY_4: int = Field(..., description="Repayment status June 2005",      example=0)
    PAY_5: int = Field(..., description="Repayment status May 2005",       example=0)
    PAY_6: int = Field(..., description="Repayment status April 2005",     example=0)

    # ── Bill amounts ──────────────────────────────────────────────
    BILL_AMT1: float = Field(..., description="Bill amount September 2005", example=20000)
    BILL_AMT2: float = Field(..., description="Bill amount August 2005",    example=19000)
    BILL_AMT3: float = Field(..., description="Bill amount July 2005",      example=18000)
    BILL_AMT4: float = Field(..., description="Bill amount June 2005",      example=17000)
    BILL_AMT5: float = Field(..., description="Bill amount May 2005",       example=16000)
    BILL_AMT6: float = Field(..., description="Bill amount April 2005",     example=15000)

    # ── Payment amounts ───────────────────────────────────────────
    PAY_AMT1: float = Field(..., description="Payment amount September 2005", example=5000)
    PAY_AMT2: float = Field(..., description="Payment amount August 2005",    example=5000)
    PAY_AMT3: float = Field(..., description="Payment amount July 2005",      example=4000)
    PAY_AMT4: float = Field(..., description="Payment amount June 2005",      example=4000)
    PAY_AMT5: float = Field(..., description="Payment amount May 2005",       example=3000)
    PAY_AMT6: float = Field(..., description="Payment amount April 2005",     example=3000)

    class Config:
        json_schema_extra = {
            "example": {
                "LIMIT_BAL": 50000,
                "AGE": 35,
                "PAY_0": 0, "PAY_2": 0, "PAY_3": 0,
                "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
                "BILL_AMT1": 20000, "BILL_AMT2": 19000, "BILL_AMT3": 18000,
                "BILL_AMT4": 17000, "BILL_AMT5": 16000, "BILL_AMT6": 15000,
                "PAY_AMT1": 5000, "PAY_AMT2": 5000, "PAY_AMT3": 4000,
                "PAY_AMT4": 4000, "PAY_AMT5": 3000, "PAY_AMT6": 3000,
            }
        }


class ScorecardResponse(BaseModel):
    """Response from the scorecard (LR + WOE) model."""
    model:               str
    credit_score:        int
    default_probability: float
    risk_level:          str
    decision:            str


class XGBResponse(BaseModel):
    """Response from the XGBoost challenger model."""
    model:               str
    default_probability: float
    risk_level:          str
    decision:            str


class CombinedResponse(BaseModel):
    """Response combining both models — returned by /predict/both."""
    scorecard: ScorecardResponse
    xgboost:   XGBResponse
