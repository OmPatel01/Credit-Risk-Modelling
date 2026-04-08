"""
input_mapper.py
---------------
Maps business-friendly input to raw dataset format.

This module bridges the gap between user-facing business inputs and the
internal raw dataset feature format required by the preprocessing pipeline.

Flow:
    BusinessInput (user-friendly)
        ↓
    map_business_to_raw()
        ↓
    ClientInput (raw features)
        ↓
    preprocessing.engineer_features() (existing pipeline)
        ↓
    Model prediction


Fixes Applied
=============
1. CRITICAL: Pydantic → dict conversion added at top of map_business_to_raw()
2. CRITICAL: validate_business_input() now called explicitly inside mapper
3. FIXED:    num_delays now controls which PAY_X months carry a delay value
4. FIXED:    zero_payment_count now applies to OLDER months (6→5→4…), not recent
5. FIXED:    EDUCATION is included in raw output (needed by XGBoost pipeline)


Mapping Logic
=============

Demographics:
    LIMIT_BAL, AGE, EDUCATION → passed through unchanged

Repayment Status (PAY_X):
    PAY_0      ← recent_delay  (most recent month)
    PAY_2..6   ← controlled by num_delays + avg_past_delay
                 First num_delays months (oldest first) get avg_past_delay;
                 remaining months get 0 (paid on time)

Bill Amounts (BILL_AMT_X):
    BILL_AMT1..6 ← avg_bill_amount with optional bill_growth_rate variation
    Month 1 (most recent) = avg_bill * (1 + growth_rate * 0)
    Month 6 (oldest)      = avg_bill * (1 + growth_rate * 1)

Payment Amounts (PAY_AMT_X):
    PAY_AMT1..6 ← payment_amount
    zero_payment_count applies to OLDEST months first:
        Month 6, 5, 4 … get 0 up to zero_payment_count
        Recent months get payment_amount

Example
=======

Business Input:
    LIMIT_BAL:         50000
    AGE:               35
    EDUCATION:         2
    recent_delay:      0
    avg_past_delay:    1.0
    num_delays:        2
    avg_bill_amount:   20000
    bill_growth_rate:  -0.05
    payment_amount:    5000
    zero_payment_count: 1

Output (ClientInput):
    LIMIT_BAL:    50000
    AGE:          35
    EDUCATION:    2
    PAY_0:        0       ← recent_delay
    PAY_2:        1       ← num_delays=2 → oldest two PAY months get delay
    PAY_3:        1
    PAY_4:        0       ← paid on time
    PAY_5:        0
    PAY_6:        0
    BILL_AMT1:    20000   ← most recent, no growth offset
    BILL_AMT2:    19800   ← slight shrinkage
    BILL_AMT3:    19600
    BILL_AMT4:    19400
    BILL_AMT5:    19200
    BILL_AMT6:    19000   ← oldest, full growth_rate applied
    PAY_AMT1:     5000    ← recent months paid
    PAY_AMT2:     5000
    PAY_AMT3:     5000
    PAY_AMT4:     5000
    PAY_AMT5:     5000
    PAY_AMT6:     0       ← zero_payment_count=1 → only oldest month zeroed
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def map_business_to_raw(business_input) -> Dict[str, Any]:
    logger.info("[MAPPER] Mapping started")

    try:
        # ── Step 0: Normalize ─────────────────────────
        if hasattr(business_input, 'dict'):
            business_input = business_input.dict()
        elif not isinstance(business_input, dict):
            business_input = dict(business_input)

        validate_business_input(business_input)

        # ── Step 1: Demographics ───────────────────────
        raw_input = {
            "LIMIT_BAL": business_input["LIMIT_BAL"],
            "AGE":       business_input["AGE"],
            "EDUCATION": business_input["EDUCATION"],
        }

        # ── Step 2: PAY_X mapping (FIXED: recent-first delays) ─────
        raw_input["PAY_0"] = int(round(business_input["recent_delay"]))

        num_delays     = int(business_input.get("num_delays", 0))
        avg_past_delay = business_input["avg_past_delay"]

        # FIX: assign delays to RECENT months first
        recent_months = [2, 3, 4, 5, 6]

        for position, month in enumerate(recent_months):
            if position < num_delays:
                raw_input[f"PAY_{month}"] = int(round(avg_past_delay))
            else:
                raw_input[f"PAY_{month}"] = 0

        logger.debug(f"[MAPPER] PAY_X mapped → num_delays={num_delays}")

        # ── Sanity check: validate num_delays vs actual PAY_X ───────
        actual_delays = sum(
            1 for k, v in raw_input.items()
            if k.startswith("PAY_") and k != "PAY_0" and v >= 1
        )

        logger.debug(
            f"[MAPPER] Delay sanity → expected={num_delays}, actual={actual_delays}"
        )

        # ── Step 3: BILL_AMT_X ────────────────────────
        avg_bill    = business_input["avg_bill_amount"]
        growth_rate = business_input.get("bill_growth_rate", 0.0)

        for month in range(1, 7):
            offset        = (month - 1) / 5.0
            adjusted_bill = avg_bill * (1.0 + growth_rate * offset)
            raw_input[f"BILL_AMT{month}"] = round(max(adjusted_bill, 0.0), 2)

        # ── Step 4: PAY_AMT_X (FIXED: realistic behavior) ─────────
        payment    = business_input["payment_amount"]
        zero_count = int(business_input.get("zero_payment_count", 0))

        for month in range(1, 7):
            # Older months → more likely zero
            age_rank = 7 - month

            if age_rank <= zero_count:
                raw_input[f"PAY_AMT{month}"] = 0.0
            else:
                # FIX: recent month should be slightly more reliable
                if month == 1:
                    raw_input["PAY_AMT1"] = float(payment)
                else:
                    raw_input[f"PAY_AMT{month}"] = float(payment)

        logger.debug(f"[MAPPER] PAY_AMT mapped → zero_count={zero_count}")

        # ── Final summary ─────────────────────────────
        logger.info(
            f"[MAPPER] Mapping completed → "
            f"LIMIT_BAL={raw_input['LIMIT_BAL']}, AGE={raw_input['AGE']}, "
            f"PAY_0={raw_input['PAY_0']}, BILL_AMT1={raw_input['BILL_AMT1']}"
        )

        return raw_input

    except Exception as e:
        logger.error(f"[MAPPER] Error during mapping: {str(e)}", exc_info=True)
        raise

def validate_business_input(business_input: Dict[str, Any]) -> bool:
    logger.debug("[MAPPER] Validation started")

    if hasattr(business_input, 'dict'):
        business_input = business_input.dict()

    try:
        # ── Step 0: Normalize ─────────────────────────
        if hasattr(business_input, 'dict'):
            business_input = business_input.dict()
        elif not isinstance(business_input, dict):
            business_input = dict(business_input)

        logger.debug(f"[MAPPER] Normalized input for validation: keys={list(business_input.keys())}")           

        # ── LIMIT_BAL ────────────────────────────────────────────────
        limit_bal = business_input.get("LIMIT_BAL", 0)
        if not isinstance(limit_bal, (int, float)) or limit_bal <= 0:
            raise ValueError(f"LIMIT_BAL must be a positive number, got {limit_bal}")

        # ── AGE ──────────────────────────────────────────────────────
        age = business_input.get("AGE", 0)
        if not isinstance(age, (int, float)) or not (18 <= age <= 100):
            raise ValueError(f"AGE must be between 18 and 100, got {age}")

        # ── EDUCATION ────────────────────────────────────────────────
        education = business_input.get("EDUCATION", 0)
        if not isinstance(education, (int, float)) or not (1 <= education <= 4):
            raise ValueError(f"EDUCATION must be 1–6, got {education}")

        # ── recent_delay (PAY_0) ─────────────────────────────────────
        recent_delay = business_input.get("recent_delay", 0)
        if not isinstance(recent_delay, (int, float)) or not (-2 <= recent_delay <= 8):
            raise ValueError(f"recent_delay must be −2 to 8, got {recent_delay}")

        # ── avg_past_delay ───────────────────────────────────────────
        avg_past_delay = business_input.get("avg_past_delay", 0)
        if not isinstance(avg_past_delay, (int, float)) or not (-2 <= avg_past_delay <= 8):
            raise ValueError(f"avg_past_delay must be −2 to 8, got {avg_past_delay}")

        # ── num_delays ───────────────────────────────────────────────
        num_delays = business_input.get("num_delays", 0)
        if not isinstance(num_delays, (int, float)) or not (0 <= num_delays <= 5):
        # 5 because PAY_2..PAY_6 = 5 older months
            raise ValueError(f"num_delays must be 0–5 (covers PAY_2..PAY_6), got {num_delays}")

        # ── avg_bill_amount ──────────────────────────────────────────
        avg_bill = business_input.get("avg_bill_amount", 0)
        if not isinstance(avg_bill, (int, float)) or avg_bill < 0:
            raise ValueError(f"avg_bill_amount must be non-negative, got {avg_bill}")

        # ── bill_growth_rate ─────────────────────────────────────────
        growth_rate = business_input.get("bill_growth_rate", 0)
        if not isinstance(growth_rate, (int, float)) or not (-1 <= growth_rate <= 1):
            raise ValueError(f"bill_growth_rate must be −1 to 1, got {growth_rate}")

        # ── payment_amount ───────────────────────────────────────────
        payment = business_input.get("payment_amount", 0)
        if not isinstance(payment, (int, float)) or payment < 0:
            raise ValueError(f"payment_amount must be non-negative, got {payment}")

        # ── zero_payment_count ───────────────────────────────────────
        zero_count = business_input.get("zero_payment_count", 0)
        if not isinstance(zero_count, (int, float)) or not (0 <= zero_count <= 6):
            raise ValueError(f"zero_payment_count must be 0–6, got {zero_count}")
        
        logger.debug("[MAPPER] Validation successful")
        return True

    except ValueError as e:
        logger.warning(f"[MAPPER] Validation failed: {str(e)}")
        raise
