"""Request and response models for the API.

Pydantic checks every request against these models and rejects bad input with
a 422 error before it reaches the model. tests/test_api.py checks that the
Session fields match the input columns in config.yaml, so the two cannot drift.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE_SESSION = {
    "Administrative": 1,
    "Administrative_Duration": 14.65,
    "Informational": 0,
    "Informational_Duration": 0.0,
    "ProductRelated": 19,
    "ProductRelated_Duration": 283.88,
    "BounceRates": 0.008,
    "ExitRates": 0.042,
    "SpecialDay": 0.0,
    "Month": "June",
    "OperatingSystems": 1,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 1,
    "VisitorType": "Returning_Visitor",
    "Weekend": False,
}


class Session(BaseModel):
    """One shopping session, as known mid-visit.

    PageValues is not needed: it is left out of the model to prevent leakage.
    Extra fields are ignored. Unknown categories (for example a new Browser ID)
    are accepted and handled by the model's encoder.
    """

    model_config = ConfigDict(extra="ignore", json_schema_extra={"example": EXAMPLE_SESSION})

    Administrative: int = Field(ge=0, description="Administrative pages visited")
    Administrative_Duration: float = Field(ge=0, description="Seconds on administrative pages")
    Informational: int = Field(ge=0, description="Informational pages visited")
    Informational_Duration: float = Field(ge=0, description="Seconds on informational pages")
    ProductRelated: int = Field(ge=0, description="Product pages visited")
    ProductRelated_Duration: float = Field(ge=0, description="Seconds on product pages")
    BounceRates: float = Field(ge=0, le=1, description="Average bounce rate of pages visited")
    ExitRates: float = Field(ge=0, le=1, description="Average exit rate of pages visited")
    SpecialDay: float = Field(ge=0, le=1, description="Closeness to a special shopping day")
    Month: str = Field(min_length=1, description="Month of the visit, e.g. Nov or June")
    OperatingSystems: int = Field(description="Operating system ID (a category)")
    Browser: int = Field(description="Browser ID (a category)")
    Region: int = Field(description="Region ID (a category)")
    TrafficType: int = Field(description="Traffic source ID (a category)")
    VisitorType: str = Field(min_length=1, description="Returning_Visitor, New_Visitor or Other")
    Weekend: bool = Field(description="Whether the visit is on a weekend")


class Contributor(BaseModel):
    """One behaviour and how much it moved the score (SHAP, in log-odds)."""

    feature: str
    value: bool | int | float | str
    contribution: float
    effect: Literal["increases", "decreases"]


class Prediction(BaseModel):
    model_version: str
    conversion_probability: float = Field(description="Chance the session ends in a purchase")
    will_convert: bool = Field(description="Whether the probability reaches the threshold")
    threshold: float = Field(description="Threshold used for will_convert")
    confidence: Literal["low", "medium", "high"] = Field(
        description="How far the probability is from the threshold. Not a calibrated certainty."
    )
    top_contributors: list[Contributor]


class Health(BaseModel):
    status: Literal["ok"]
    model_version: str
