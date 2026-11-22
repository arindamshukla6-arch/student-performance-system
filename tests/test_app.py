"""
Tests for the Student Performance Early Warning System.

Run from the project folder (after  python train_model.py):
    python -m pytest tests -v
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import predictor  # noqa: E402
from app import DEMO_PASSWORD, DEMO_USERNAME, create_app  # noqa: E402
from preprocessing import validate_identity, validate_student_input  # noqa: E402

GOOD_FORM = {
    "attendance": "82",
    "internal_obtained": "40",
    "internal_max": "50",
    "assignment_status": "Completed",
    "previous_value": "72",
    "previous_scale": "percentage",
}

pytestmark = pytest.mark.skipif(
    not predictor.MODEL_PATH.exists(), reason="Run python train_model.py first"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def client(tmp_path):
    app = create_app({"TESTING": True, "DATABASE": str(tmp_path / "test.db")})
    with app.test_client() as c:
        yield c


def token(client, url="/login"):
    html = client.get(url).get_data(as_text=True)
    return re.search(r'name="_csrf" value="([^"]+)"', html).group(1)


@pytest.fixture()
def logged_in(client):
    client.post("/login", data={"_csrf": token(client), "username": DEMO_USERNAME, "password": DEMO_PASSWORD})
    return client


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_valid_input_is_converted_to_percentages():
    clean, errors = validate_student_input(GOOD_FORM)
    assert errors == {}
    assert clean["internal_marks_pct"] == 80.0
    assert clean["previous_performance_pct"] == 72.0


def test_cgpa_is_converted_to_percentage():
    form = dict(GOOD_FORM, previous_value="7.5", previous_scale="cgpa")
    clean, errors = validate_student_input(form)
    assert errors == {}
    assert clean["previous_performance_pct"] == 75.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("attendance", "101"),
        ("attendance", "-1"),
        ("attendance", "abc"),
        ("attendance", "nan"),
        ("attendance", ""),
        ("internal_obtained", "60"),      # more than max of 50
        ("internal_max", "0"),
        ("assignment_status", "Maybe"),
        ("previous_value", "11"),          # only valid for percentage, see next test
    ],
)
def test_invalid_values_are_rejected(field, value):
    form = dict(GOOD_FORM, **{field: value})
    if field == "previous_value":
        form["previous_scale"] = "cgpa"
    _, errors = validate_student_input(form)
    assert errors, f"{field}={value!r} should be rejected"


def test_student_id_rules():
    _, errors = validate_identity({"student_id": "ab", "student_name": ""})
    assert "student_id" in errors
    ident, errors = validate_identity({"student_id": " stu-001 ", "student_name": " Test  Name "})
    assert errors == {} and ident["student_id"] == "STU-001" and ident["name"] == "Test Name"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def test_model_predicts_a_valid_label_with_probabilities():
    clean, _ = validate_student_input(GOOD_FORM)
    result = predictor.predict_risk(clean)
    assert result["label"] in ("Low Risk", "Moderate Risk", "High Risk")
    assert abs(sum(result["probabilities"].values()) - 100) < 0.5


def test_model_orders_extreme_students_sensibly():
    strong, _ = validate_student_input(GOOD_FORM)
    weak, _ = validate_student_input(
        dict(GOOD_FORM, attendance="35", internal_obtained="8", assignment_status="Not Completed", previous_value="30")
    )
    assert predictor.predict_risk(strong)["label"] == "Low Risk"
    assert predictor.predict_risk(weak)["label"] == "High Risk"


# ---------------------------------------------------------------------------
# Routes and security
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "path",
    ["/dashboard", "/students", "/students/add", "/predict", "/analytics", "/evaluation", "/report", "/about", "/export/csv"],
)
def test_pages_require_login(client, path):
    response = client.get(path)
    assert response.status_code == 302 and "/login" in response.headers["Location"]


def test_login_wrong_password_is_rejected(client):
    response = client.post("/login", data={"_csrf": token(client), "username": DEMO_USERNAME, "password": "wrong"})
    assert response.status_code == 401
    assert b"Incorrect email or password" in response.data


def test_post_without_csrf_token_is_rejected(client):
    response = client.post("/login", data={"username": DEMO_USERNAME, "password": DEMO_PASSWORD})
    assert response.status_code == 400


def test_login_does_not_redirect_to_other_sites(client):
    response = client.post(
        "/login?next=//evil.example.com",
        data={"_csrf": token(client), "username": DEMO_USERNAME, "password": DEMO_PASSWORD},
    )
    assert response.status_code == 302 and "evil.example.com" not in response.headers["Location"]


@pytest.mark.parametrize(
    "path",
    ["/", "/dashboard", "/students", "/students/add", "/predict", "/analytics", "/evaluation", "/report", "/about", "/export/csv"],
)
def test_every_page_loads_when_logged_in(logged_in, path):
    response = logged_in.get(path, follow_redirects=True)
    assert response.status_code == 200


def test_demo_data_is_seeded_from_the_real_model(logged_in):
    html = logged_in.get("/students").get_data(as_text=True)
    assert "DEMO001" in html and "Low Risk" in html and "High Risk" in html


def test_predict_page_shows_real_prediction(logged_in):
    data = dict(GOOD_FORM, _csrf=token(logged_in, "/predict"))
    response = logged_in.post("/predict", data=data)
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "Predicted risk category" in html and "Screening indication only" in html
    assert "Recommended support actions" in html


def test_predict_rejects_invalid_input(logged_in):
    data = dict(GOOD_FORM, attendance="150", _csrf=token(logged_in, "/predict"))
    response = logged_in.post("/predict", data=data)
    assert response.status_code == 400
    assert b"Attendance must be between 0 and 100" in response.data


def test_add_student_then_view_search_and_delete(logged_in):
    data = dict(GOOD_FORM, student_id="T001", student_name="Test Person", _csrf=token(logged_in, "/students/add"))
    response = logged_in.post("/students/add", data=data, follow_redirects=True)
    assert response.status_code == 200 and b"T001" in response.data

    # duplicate IDs are refused
    again = logged_in.post("/students/add", data=data)
    assert again.status_code == 400 and b"already exists" in again.data

    # search, filter and sort work
    assert b"T001" in logged_in.get("/students?q=T001").data
    assert b"T001" not in logged_in.get("/students?q=ZZZ").data
    assert logged_in.get("/students?sort=attendance&dir=asc&risk=High+Risk").status_code == 200
    assert logged_in.get("/students?sort=evil;DROP&dir=sideways").status_code == 200  # unknown values are ignored

    # delete
    csrf = token(logged_in, "/students/T001")
    deleted = logged_in.post("/students/T001/delete", data={"_csrf": csrf}, follow_redirects=True)
    assert b"deleted" in deleted.data
    assert logged_in.get("/students/T001").status_code == 404


def test_csv_export_downloads_records_and_neutralises_formulas(logged_in):
    data = dict(GOOD_FORM, student_id="T002", student_name="=cmd()", _csrf=token(logged_in, "/students/add"))
    logged_in.post("/students/add", data=data)
    response = logged_in.get("/export/csv")
    text = response.get_data(as_text=True)
    assert response.mimetype == "text/csv"
    assert text.startswith("Student ID,Student Name")
    assert "DEMO001" in text
    assert "'=cmd()" in text and ",=cmd()," not in text


def test_unknown_page_gives_friendly_404(logged_in):
    response = logged_in.get("/no-such-page")
    assert response.status_code == 404 and b"Page not found" in response.data


def test_missing_model_is_handled_gracefully(logged_in, monkeypatch):
    def broken():
        raise predictor.ModelNotAvailable("Trained model not found. Run  python train_model.py  first.")
    monkeypatch.setattr(predictor, "load_model", broken)
    data = dict(GOOD_FORM, _csrf=token(logged_in, "/predict"))
    response = logged_in.post("/predict", data=data)
    assert response.status_code == 503
    assert b"Trained model not found" in response.data
