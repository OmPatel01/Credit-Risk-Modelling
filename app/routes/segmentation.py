"""
app/routes/segmentation.py
---------------------------
HTTP route for the risk segmentation endpoint.

Delegates to segmentation_service.perform_segmentation.
No model inference happens here — segmentation is a post-processing step applied
to PD values that were already produced by the prediction endpoints.
"""

from fastapi import APIRouter, HTTPException

from app.schemas.risk_schemas import SegmentationRequest, SegmentationResponse
from services.segmentation_service import perform_segmentation

router = APIRouter()


@router.post(
    "/segmentation",
    response_model=SegmentationResponse,
    tags=["Risk Analytics"],
    summary="Risk Segmentation",
)
def segment_risk(request: SegmentationRequest):
    """Assign borrowers to risk buckets (quantile or fixed-threshold) and return per-bucket statistics."""
    try:
        result = perform_segmentation(
            pd_values=request.pd_values,
            method=request.method,
            num_quantiles=request.num_quantiles,
            thresholds=request.thresholds,
            labels=request.labels,
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))