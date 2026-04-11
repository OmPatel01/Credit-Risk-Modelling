"""
core/input_mapper.py
--------------------
Translates user-facing BusinessInput fields into the raw dataset format (ClientInput)
that the preprocessing pipeline expects.

Why this exists:
    The model was trained on a dataset with 21 specific columns (PAY_0, BILL_AMT1..6, etc.).
    Asking API consumers to populate all 21 fields correctly is error-prone and exposes
    internal model implementation details. BusinessInput instead captures 10 intuitive
    business concepts and this module deterministically reconstructs the 21 raw columns.

Mapping contract (what each business field controls):
    LIMIT_BAL, AGE, EDUCATION     → passed through unchanged
    recent_delay                  → PAY_0 (most recent repayment status)
    avg_past_delay + num_delays   → PAY_2..PAY_6 (older months; first num_delays months
                                    get avg_past_delay, remainder get 0)
    avg_bill_amount + bill_growth_rate → BILL_AMT1..6 (most recent = avg_bill, oldest
                                    = avg_bill × (1 + growth_rate))
    payment_amount + zero_payment_count → PAY_AMT1..6 (oldest zero_payment_count months
                                    are set to 0; recent months get payment_amount)

See module docstring examples below for concrete input/output pairs.
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def map_business_to_raw(business_input) -> Dict[str, Any]:
    """
    Convert a BusinessInput (dict or Pydantic model) into the 21-column raw feature dict.

    Validates inputs first, then reconstructs each raw column group in order.
    Raises ValueError (from validate_business_input) on any out-of-range field.
    """
    logger.info("[MAPPER] Mapping started")

    try:
        # ── Normalise to plain dict so the function works with both Pydantic models
        # and raw dicts (useful when calling from tests or other services)
        if hasattr(business_input, 'dict'):
            business_input = business_input.dict()
        elif not isinstance(business_input, dict):
            business_input = dict(business_input)

        validate_business_input(business_input)

        # ── Demographics — passed through without transformation
        raw_input = {
            "LIMIT_BAL": business_input["LIMIT_BAL"],
            "AGE":       business_input["AGE"],
            "EDUCATION": business_input["EDUCATION"],
        }

        # ── PAY_X mapping — reconstructs 6 monthly repayment status columns
        # PAY_0 = most recent month, controlled directly by recent_delay
        raw_input["PAY_0"] = int(round(business_input["recent_delay"]))

        num_delays     = int(business_input.get("num_delays", 0))
        avg_past_delay = business_input["avg_past_delay"]

        # Assign delays to the MOST RECENT older months first (PAY_2, PAY_3, ...);
        # remaining months are set to 0 (paid on time).
        # Example: num_delays=2 → PAY_2=avg_past_delay, PAY_3=avg_past_delay, PAY_4..6=0
        recent_months = [2, 3, 4, 5, 6]
        for position, month in enumerate(recent_months):
            if position < num_delays:
                raw_input[f"PAY_{month}"] = int(round(avg_past_delay))
            else:
                raw_input[f"PAY_{month}"] = 0

        logger.debug(f"[MAPPER] PAY_X mapped → num_delays={num_delays}")

        # Sanity check: count how many PAY_X (excluding PAY_0) actually ended up >= 1
        actual_delays = sum(
            1 for k, v in raw_input.items()
            if k.startswith("PAY_") and k != "PAY_0" and v >= 1
        )
        logger.debug(
            f"[MAPPER] Delay sanity → expected={num_delays}, actual={actual_delays}"
        )

        # ── BILL_AMT_X mapping — interpolates bill amounts across 6 months
        # BILL_AMT1 = most recent (no growth offset), BILL_AMT6 = oldest (full offset).
        # A negative growth_rate means the balance was shrinking over time (healthy sign).
        avg_bill    = business_input["avg_bill_amount"]
        growth_rate = business_input.get("bill_growth_rate", 0.0)

        for month in range(1, 7):
            # offset goes from 0.0 (month 1, most recent) to 1.0 (month 6, oldest)
            offset        = (month - 1) / 5.0
            adjusted_bill = avg_bill * (1.0 + growth_rate * offset)
            raw_input[f"BILL_AMT{month}"] = round(max(adjusted_bill, 0.0), 2)

        # ── PAY_AMT_X mapping — zero payments are assigned to OLDEST months first
        # This reflects real behaviour: clients who stop paying do so progressively,
        # and their most recent months are the first to show zero payments when
        # looking backward in time. age_rank=1 = most recent, age_rank=6 = oldest.
        payment    = business_input["payment_amount"]
        zero_count = int(business_input.get("zero_payment_count", 0))

        for month in range(1, 7):
            age_rank = 7 - month  # month=1 → age_rank=6 (most recent), month=6 → age_rank=1 (oldest)
            if age_rank <= zero_count:
                raw_input[f"PAY_AMT{month}"] = 0.0
            else:
                raw_input[f"PAY_AMT{month}"] = float(payment)

        logger.debug(f"[MAPPER] PAY_AMT mapped → zero_count={zero_count}")

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
    """
    Enforce domain constraints on all BusinessInput fields before mapping.

    Raises ValueError with a descriptive message on the first failing field.
    These constraints mirror the Pydantic schema validators in app/schema.py
    but are re-checked here because map_business_to_raw can also be called
    with plain dicts that bypass Pydantic (e.g. in tests or internal tooling).
    """
    logger.debug("[MAPPER] Validation started")

    # Re-normalise in case this function is called directly with a Pydantic model
    if hasattr(business_input, 'dict'):
        business_input = business_input.dict()
    elif not isinstance(business_input, dict):
        business_input = dict(business_input)

    logger.debug(f"[MAPPER] Normalized input for validation: keys={list(business_input.keys())}")

    # ── Each block validates one field; error messages include the actual received value
    # so it's immediately clear which field failed and why

    limit_bal = business_input.get("LIMIT_BAL", 0)
    if not isinstance(limit_bal, (int, float)) or limit_bal <= 0:
        raise ValueError(f"LIMIT_BAL must be a positive number, got {limit_bal}")

    age = business_input.get("AGE", 0)
    if not isinstance(age, (int, float)) or not (18 <= age <= 100):
        raise ValueError(f"AGE must be between 18 and 100, got {age}")

    education = business_input.get("EDUCATION", 0)
    if not isinstance(education, (int, float)) or not (1 <= education <= 4):
        raise ValueError(f"EDUCATION must be 1–6, got {education}")

    recent_delay = business_input.get("recent_delay", 0)
    if not isinstance(recent_delay, (int, float)) or not (-2 <= recent_delay <= 8):
        raise ValueError(f"recent_delay must be −2 to 8, got {recent_delay}")

    avg_past_delay = business_input.get("avg_past_delay", 0)
    if not isinstance(avg_past_delay, (int, float)) or not (-2 <= avg_past_delay <= 8):
        raise ValueError(f"avg_past_delay must be −2 to 8, got {avg_past_delay}")

    # num_delays covers PAY_2..PAY_6 = 5 possible months, not 6 (PAY_0 is controlled separately)
    num_delays = business_input.get("num_delays", 0)
    if not isinstance(num_delays, (int, float)) or not (0 <= num_delays <= 5):
        raise ValueError(f"num_delays must be 0–5 (covers PAY_2..PAY_6), got {num_delays}")

    avg_bill = business_input.get("avg_bill_amount", 0)
    if not isinstance(avg_bill, (int, float)) or avg_bill < 0:
        raise ValueError(f"avg_bill_amount must be non-negative, got {avg_bill}")

    growth_rate = business_input.get("bill_growth_rate", 0)
    if not isinstance(growth_rate, (int, float)) or not (-1 <= growth_rate <= 1):
        raise ValueError(f"bill_growth_rate must be −1 to 1, got {growth_rate}")

    payment = business_input.get("payment_amount", 0)
    if not isinstance(payment, (int, float)) or payment < 0:
        raise ValueError(f"payment_amount must be non-negative, got {payment}")

    zero_count = business_input.get("zero_payment_count", 0)
    if not isinstance(zero_count, (int, float)) or not (0 <= zero_count <= 6):
        raise ValueError(f"zero_payment_count must be 0–6, got {zero_count}")

    logger.debug("[MAPPER] Validation successful")
    return True