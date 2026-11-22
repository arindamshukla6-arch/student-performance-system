"""
preprocessing.py
----------------
Everything that is shared by the training script and the Flask app:

* constants (feature names, risk labels, assignment options)
* input validation for the web forms
* the scikit-learn preprocessing step (missing values + encoding + scaling)

Keeping this in one file guarantees that the model sees data prepared in
exactly the same way during training and during prediction.
"""

import math
import re

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RISK_LABELS = ["Low Risk", "Moderate Risk", "High Risk"]
TARGET = "risk_level"

ASSIGNMENT_OPTIONS = ["Completed", "Partially Completed", "Not Completed"]

# Model input columns. Marks and previous performance are always converted to
# percentages (0-100) before they reach the model, so every number is on the
# same, clearly defined scale.
NUMERIC_FEATURES = ["attendance", "internal_marks_pct", "previous_performance_pct"]
CATEGORICAL_FEATURES = ["assignment_status"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Scale used for "previous academic performance"
#   percentage : 0 - 100  (used as is)
#   cgpa       : 0 - 10   (converted with percentage = CGPA x 10; this is a
#                          simple approximation, universities differ)
PREVIOUS_SCALES = {"percentage": 100.0, "cgpa": 10.0}

STUDENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{3,20}$")


# ---------------------------------------------------------------------------
# Preprocessing pipeline step
# ---------------------------------------------------------------------------
def build_preprocessor():
    """Return the ColumnTransformer used at the start of every model pipeline.

    numeric columns     -> fill missing with the median -> standardise
    categorical column  -> fill missing with the most common value -> one-hot
    """
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(categories=[ASSIGNMENT_OPTIONS], handle_unknown="ignore"),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def to_feature_frame(clean):
    """Turn one validated student dict into the one-row DataFrame the model expects."""
    return pd.DataFrame([{name: clean[name] for name in FEATURES}], columns=FEATURES)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _read_number(form, key, label, errors):
    """Read a finite float from the form. Records an error and returns None if invalid."""
    raw = (form.get(key) or "").strip()
    if raw == "":
        errors[key] = f"{label} is required."
        return None
    try:
        value = float(raw)
    except ValueError:
        errors[key] = f"{label} must be a number."
        return None
    if not math.isfinite(value):
        errors[key] = f"{label} must be a real number."
        return None
    return value


def validate_student_input(form):
    """Validate the academic fields of the add-student / predict forms.

    Returns (clean, errors). `clean` is only usable when `errors` is empty.
    """
    errors = {}
    clean = {}

    attendance = _read_number(form, "attendance", "Attendance", errors)
    if attendance is not None:
        if not 0 <= attendance <= 100:
            errors["attendance"] = "Attendance must be between 0 and 100."
        else:
            clean["attendance"] = round(attendance, 1)

    obtained = _read_number(form, "internal_obtained", "Internal marks obtained", errors)
    maximum = _read_number(form, "internal_max", "Maximum internal marks", errors)
    if maximum is not None and not 0 < maximum <= 1000:
        errors["internal_max"] = "Maximum marks must be greater than 0 and at most 1000."
        maximum = None
    if obtained is not None and obtained < 0:
        errors["internal_obtained"] = "Marks obtained cannot be negative."
        obtained = None
    if obtained is not None and maximum is not None:
        if obtained > maximum:
            errors["internal_obtained"] = "Marks obtained cannot be more than the maximum marks."
        else:
            clean["internal_obtained"] = round(obtained, 2)
            clean["internal_max"] = round(maximum, 2)
            clean["internal_marks_pct"] = round(obtained / maximum * 100, 1)

    status = (form.get("assignment_status") or "").strip()
    if status not in ASSIGNMENT_OPTIONS:
        errors["assignment_status"] = "Choose one of the assignment status options."
    else:
        clean["assignment_status"] = status

    scale = (form.get("previous_scale") or "percentage").strip().lower()
    if scale not in PREVIOUS_SCALES:
        errors["previous_scale"] = "Choose percentage or CGPA."
        scale = None
    previous = _read_number(form, "previous_value", "Previous performance", errors)
    if previous is not None and scale is not None:
        top = PREVIOUS_SCALES[scale]
        if not 0 <= previous <= top:
            errors["previous_value"] = f"Previous performance must be between 0 and {top:g} for this scale."
        else:
            clean["previous_scale"] = scale
            clean["previous_value"] = round(previous, 2)
            clean["previous_performance_pct"] = round(previous / top * 100, 1)

    return clean, errors


def validate_identity(form):
    """Validate the student ID and the (display-only) name."""
    errors = {}
    student_id = (form.get("student_id") or "").strip().upper()
    name = " ".join((form.get("student_name") or "").split())

    if not STUDENT_ID_PATTERN.match(student_id):
        errors["student_id"] = "Student ID must be 3-20 characters: letters, digits, - or _."
    if len(name) > 60:
        errors["student_name"] = "Name must be 60 characters or fewer."
    return {"student_id": student_id, "name": name}, errors
