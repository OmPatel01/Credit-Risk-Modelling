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
from typing import Optional, Dict

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
    EDUCATION: int   = Field(..., description="Education level (1-6)", example=2)

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
                "EDUCATION": 2,
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


class BusinessInput(BaseModel):
    """
    User-friendly business input for credit risk prediction.
    
    This schema abstracts technical details and maps to the raw dataset format
    (ClientInput) via the input_mapper module.
    
    Fields are intentionally named using business terminology rather than
    the original dataset feature names.
    """

    # ── Credit Profile ───────────────────────────────────────────
    LIMIT_BAL: float = Field(
        ..., 
        description="Credit limit (NT dollars)", 
        example=50000,
        ge=0
    )
    AGE: int = Field(
        ..., 
        description="Age in years", 
        example=35,
        ge=18,
        le=100
    )
    EDUCATION: int = Field(
        ..., 
        description="Education level (1=graduate school, 2=university, 3=high school, 4=others)", 
        example=2,
        ge=1,
        le=6
    )

    # ── Repayment Behavior ───────────────────────────────────────
    recent_delay: int = Field(
        ...,
        description="Most recent payment status (-2 to 8, where 0=paid on time, >0=months overdue, -1/-2=revolving credit/paid in full)",
        example=0,
        ge=-2,
        le=8
    )
    avg_past_delay: float = Field(
        ...,
        description="Average payment status over previous 5 months (PAY_2 through PAY_6)",
        example=0.0,
        ge=-2,
        le=8
    )
    num_delays: int = Field(
        ...,
        description="Number of months with payment delays (0-6)",
        example=0,
        ge=0,
        le=6
    )

    # ── Billing History ──────────────────────────────────────────
    avg_bill_amount: float = Field(
        ...,
        description="Average monthly bill amount (NT dollars)",
        example=15000,
        ge=0
    )
    bill_growth_rate: float = Field(
        default=0.0,
        description="Rate of growth in bill amounts from oldest to newest month (-1 to 1)",
        example=0.0,
        ge=-1,
        le=1
    )

    # ── Payment Amount ───────────────────────────────────────────
    payment_amount: float = Field(
        ...,
        description="Typical monthly payment amount (NT dollars)",
        example=5000,
        ge=0
    )
    zero_payment_count: int = Field(
        default=0,
        description="Number of months with zero payment (0-6)",
        example=0,
        ge=0,
        le=6
    )

    class Config:
        json_schema_extra = {
            "example": {
                "LIMIT_BAL": 50000,
                "AGE": 35,
                "EDUCATION": 2,
                "recent_delay": 0,
                "avg_past_delay": 0.0,
                "num_delays": 0,
                "avg_bill_amount": 15000,
                "bill_growth_rate": 0.0,
                "payment_amount": 5000,
                "zero_payment_count": 0,
            }
        }


class WhatIfRequest(BaseModel):
    base_input: BusinessInput
    modified_input: BusinessInput


class WhatIfResponse(BaseModel):
    base_score: int
    new_score: int
    delta_score: int

    base_pd: float
    new_pd: float
    delta_pd: float
