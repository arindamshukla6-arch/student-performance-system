"""
predictor.py
------------
Loads the saved scikit-learn pipeline and turns validated form data into a
prediction, a basic explanation and recommended support actions.
"""

import json
from pathlib import Path

import joblib

from preprocessing import RISK_LABELS, to_feature_frame

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "student_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

_model = None


class ModelNotAvailable(Exception):
    """Raised when the trained model file is missing or cannot be loaded."""


def load_model():
    """Load the model once and keep it in memory."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise ModelNotAvailable(
                "Trained model not found. Run  python train_model.py  first."
            )
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception as exc:  # corrupted file, version mismatch, ...
            raise ModelNotAvailable(
                f"The model file could not be loaded ({exc}). "
                "Run  python train_model.py  again to rebuild it."
            ) from exc
    return _model


def reload_model():
    """Forget the cached model (used by tests)."""
    global _model
    _model = None


def model_available():
    try:
        load_model()
        return True
    except ModelNotAvailable:
        return False


def load_metrics():
    """Return the evaluation results saved by train_model.py, or None."""
    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text())
    except (OSError, ValueError):
        return None


def predict_risk(clean):
    """Predict the risk level for one validated student.

    Returns a dict with the label, the confidence (percentage, taken from the
    model's predicted probabilities) and the probability of every class.
    """
    pipeline = load_model()
    features = to_feature_frame(clean)
    label = str(pipeline.predict(features)[0])

    confidence = None
    probabilities = {}
    if hasattr(pipeline, "predict_proba"):
        scores = pipeline.predict_proba(features)[0]
        probabilities = {
            str(cls): round(float(p) * 100, 1) for cls, p in zip(pipeline.classes_, scores)
        }
        # keep the display order Low -> Moderate -> High
        probabilities = {k: probabilities[k] for k in RISK_LABELS if k in probabilities}
        confidence = probabilities.get(label)

    return {"label": label, "confidence": confidence, "probabilities": probabilities}


# ---------------------------------------------------------------------------
# Explanation and recommendations
# ---------------------------------------------------------------------------
# These thresholds are simple teaching rules used ONLY to describe the inputs in
# plain language. They are not the boundaries the model learned.
def explain_inputs(clean):
    """Describe each input as 'good', 'watch' or 'concern' in plain language."""
    factors = []

    att = clean["attendance"]
    if att < 65:
        factors.append(("concern", f"Attendance is {att:g}%, which is low."))
    elif att < 75:
        factors.append(("watch", f"Attendance is {att:g}%, close to the usual minimum requirement."))
    else:
        factors.append(("good", f"Attendance is {att:g}%, which is healthy."))

    marks = clean["internal_marks_pct"]
    if marks < 40:
        factors.append(("concern", f"Internal marks are {marks:g}% of the maximum, which is low."))
    elif marks < 55:
        factors.append(("watch", f"Internal marks are {marks:g}% of the maximum, which is below average."))
    else:
        factors.append(("good", f"Internal marks are {marks:g}% of the maximum, which is satisfactory."))

    status = clean["assignment_status"]
    if status == "Not Completed":
        factors.append(("concern", "Assignments are not completed."))
    elif status == "Partially Completed":
        factors.append(("watch", "Assignments are only partially completed."))
    else:
        factors.append(("good", "Assignments are completed."))

    prev = clean["previous_performance_pct"]
    if prev < 45:
        factors.append(("concern", f"Previous performance is {prev:g}%, which is low."))
    elif prev < 60:
        factors.append(("watch", f"Previous performance is {prev:g}%, which is moderate."))
    else:
        factors.append(("good", f"Previous performance is {prev:g}%, which is solid."))

    return [{"level": level, "text": text} for level, text in factors]


RECOMMENDATIONS = {
    "Low Risk": [
        "Maintain the current study routine.",
        "Continue regular attendance.",
        "Keep completing assignments on time.",
    ],
    "Moderate Risk": [
        "Improve attendance.",
        "Focus on weaker academic subjects.",
        "Complete pending assignments.",
        "Consult the subject teacher.",
    ],
    "High Risk": [
        "Arrange a teacher-mentor discussion.",
        "Identify attendance and learning difficulties.",
        "Prepare a structured improvement plan.",
        "Provide additional academic support.",
    ],
}

SUMMARY_TEXT = {
    "Low Risk": "The inputs look healthy overall, so the model sees little sign of academic difficulty.",
    "Moderate Risk": "Some inputs are below the level where students usually feel comfortable. Early support can prevent this from growing.",
    "High Risk": "Several inputs are weak at the same time. A conversation with the student soon is recommended.",
}
