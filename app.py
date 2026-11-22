"""
app.py
------
Flask web application for the AI-Based Student Performance Prediction and
Early Warning System (B.Tech mini project prototype).

Run:  python app.py      then open  http://127.0.0.1:5000

PROTOTYPE LIMITATIONS (also listed on the About page and in README.md):
* The login is a single DEMO account, not a real user management system.
* The data is synthetic; predictions are screening indications only.
* It has no audit log, no per-user roles and no production hardening.
"""

import csv
import io
import os
import secrets
import sqlite3
import time
from datetime import datetime
from functools import wraps

from flask import (
    Flask, Response, abort, flash, g, redirect, render_template, request,
    session, url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

import database as db
import predictor
from preprocessing import (
    ASSIGNMENT_OPTIONS, RISK_LABELS, validate_identity, validate_student_input,
)

# Demo account for the prototype. Override with environment variables if wanted.
DEMO_USERNAME = os.environ.get("DEMO_USERNAME", "teacher@demo.edu")
DEMO_PASSWORD = os.environ.get("DEMO_PASSWORD", "Teacher@123")

MAX_FAILED_LOGINS = 5
LOCKOUT_SECONDS = 60


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.update(
        # A new random key on each start is fine for a demo (you simply log in again).
        SECRET_KEY=os.environ.get("SECRET_KEY") or secrets.token_hex(32),
        DATABASE=str(db.DEFAULT_DB),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        MAX_CONTENT_LENGTH=64 * 1024,  # forms are tiny; reject large bodies
    )
    if test_config:
        app.config.update(test_config)
    app.json.sort_keys = False  # keep chart categories in their natural order
    app.jinja_env.policies["json.dumps_kwargs"] = {"sort_keys": False}  # same for the |tojson filter

    app.config["DEMO_PASSWORD_HASH"] = generate_password_hash(DEMO_PASSWORD)
    failed_logins = {}  # ip -> (count, locked_until)

    # ------------------------------------------------------------------
    # Database per request
    # ------------------------------------------------------------------
    def get_db():
        if "db" not in g:
            g.db = db.connect(app.config["DATABASE"])
        return g.db

    @app.teardown_appcontext
    def close_db(_exc):
        conn = g.pop("db", None)
        if conn is not None:
            conn.close()

    with app.app_context():
        os.makedirs(os.path.dirname(app.config["DATABASE"]), exist_ok=True)
        init_conn = db.connect(app.config["DATABASE"])
        db.init_db(init_conn)
        init_conn.close()

    seed_state = {"checked": False}

    def ensure_demo_data():
        """Seed the demo records (using the real model) once, when the database is empty.

        Runs at startup and again at login if the model was not ready earlier.
        The flag stops deleted records from coming back until the app restarts.
        To reset the demo data, stop the app and delete data/students.db.
        """
        if seed_state["checked"] or not predictor.model_available():
            return
        conn = get_db()
        if db.count_students(conn) == 0:
            db.seed_demo_students(conn, predictor.predict_risk)
        seed_state["checked"] = True

    with app.app_context():
        ensure_demo_data()

    # ------------------------------------------------------------------
    # Security helpers: login required + CSRF
    # ------------------------------------------------------------------
    def login_required(view):
        @wraps(view)
        def wrapper(*args, **kwargs):
            if not session.get("user"):
                return redirect(url_for("login", next=request.path))
            return view(*args, **kwargs)
        return wrapper

    def csrf_token():
        if "_csrf" not in session:
            session["_csrf"] = secrets.token_hex(16)
        return session["_csrf"]

    @app.before_request
    def protect_forms():
        if request.method == "POST":
            sent = request.form.get("_csrf", "")
            if not sent or not secrets.compare_digest(sent, session.get("_csrf", "")):
                abort(400, description="Your form session expired. Go back, reload the page and try again.")

    @app.context_processor
    def inject_globals():
        return {
            "csrf_token": csrf_token,
            "current_user": session.get("user"),
            "model_ok": predictor.model_available(),
            "RISK_LABELS": RISK_LABELS,
            "ASSIGNMENT_OPTIONS": ASSIGNMENT_OPTIONS,
            "now": datetime.now(),
        }

    @app.template_filter("risk_slug")
    def risk_slug(label):
        return {"Low Risk": "low", "Moderate Risk": "moderate", "High Risk": "high"}.get(label, "low")

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    @app.route("/")
    def home():
        return redirect(url_for("dashboard" if session.get("user") else "login"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if session.get("user"):
            return redirect(url_for("dashboard"))
        error = None
        if request.method == "POST":
            ip = request.remote_addr or "unknown"
            count, locked_until = failed_logins.get(ip, (0, 0))
            if locked_until > time.time():
                wait = int(locked_until - time.time()) + 1
                error = f"Too many failed attempts. Try again in {wait} seconds."
            else:
                username = (request.form.get("username") or "").strip().lower()
                password = request.form.get("password") or ""
                ok = username == DEMO_USERNAME.lower() and check_password_hash(
                    app.config["DEMO_PASSWORD_HASH"], password
                )
                if ok:
                    failed_logins.pop(ip, None)
                    ensure_demo_data()
                    session.clear()
                    session["user"] = DEMO_USERNAME
                    target = request.args.get("next", "")
                    if not target.startswith("/") or target.startswith("//"):
                        target = url_for("dashboard")
                    return redirect(target)
                count += 1
                locked = time.time() + LOCKOUT_SECONDS if count >= MAX_FAILED_LOGINS else 0
                failed_logins[ip] = (0 if locked else count, locked)
                error = "Incorrect email or password."
        return render_template(
            "login.html", error=error, demo_username=DEMO_USERNAME, demo_password=DEMO_PASSWORD
        ), (401 if error else 200)

    @app.route("/logout", methods=["POST"])
    def logout():
        session.clear()
        flash("You have been signed out.", "info")
        return redirect(url_for("login"))

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------
    @app.route("/dashboard")
    @login_required
    def dashboard():
        ensure_demo_data()
        conn = get_db()
        return render_template(
            "dashboard.html", stats=db.summary(conn), recent=db.recent_students(conn, 6)
        )

    # ------------------------------------------------------------------
    # Add student (saves a record and predicts)
    # ------------------------------------------------------------------
    @app.route("/students/add", methods=["GET", "POST"])
    @login_required
    def add_student():
        conn = get_db()
        if request.method == "GET":
            return render_template(
                "add_student.html",
                form={"student_id": db.next_student_id(conn), "internal_max": "50", "previous_scale": "percentage"},
                errors={},
            )

        identity, id_errors = validate_identity(request.form)
        clean, errors = validate_student_input(request.form)
        errors.update(id_errors)
        if not errors and db.student_exists(conn, identity["student_id"]):
            errors["student_id"] = "This Student ID already exists. Use a different ID."

        if not errors:
            try:
                prediction = predictor.predict_risk(clean)
            except predictor.ModelNotAvailable as exc:
                flash(str(exc), "error")
                return render_template("add_student.html", form=request.form, errors={}), 503
            try:
                db.insert_student(conn, identity, clean, prediction)
            except sqlite3.IntegrityError:
                errors["student_id"] = "This Student ID already exists. Use a different ID."
            else:
                flash(f"Student {identity['student_id']} saved. Risk level: {prediction['label']}.", "success")
                return redirect(url_for("student_detail", student_id=identity["student_id"]))

        flash("Please fix the highlighted fields.", "error")
        return render_template("add_student.html", form=request.form, errors=errors), 400

    # ------------------------------------------------------------------
    # Predict performance (quick check, nothing is saved)
    # ------------------------------------------------------------------
    @app.route("/predict", methods=["GET", "POST"])
    @login_required
    def predict():
        if request.method == "GET":
            return render_template(
                "predict.html", form={"internal_max": "50", "previous_scale": "percentage"}, errors={}
            )

        clean, errors = validate_student_input(request.form)
        if errors:
            flash("Please fix the highlighted fields.", "error")
            return render_template("predict.html", form=request.form, errors=errors), 400
        try:
            result = predictor.predict_risk(clean)
        except predictor.ModelNotAvailable as exc:
            flash(str(exc), "error")
            return render_template("predict.html", form=request.form, errors={}), 503
        return render_template("prediction.html", **_result_context(clean, result))

    def _result_context(clean, result, student=None):
        return {
            "inputs": clean,
            "result": result,
            "factors": predictor.explain_inputs(clean),
            "actions": predictor.RECOMMENDATIONS[result["label"]],
            "summary_text": predictor.SUMMARY_TEXT[result["label"]],
            "student": student,
        }

    # ------------------------------------------------------------------
    # Student records
    # ------------------------------------------------------------------
    @app.route("/students")
    @login_required
    def students():
        ensure_demo_data()
        search = (request.args.get("q") or "").strip()[:60]
        risk = request.args.get("risk", "")
        sort = request.args.get("sort", "added")
        direction = request.args.get("dir", "desc")
        rows = db.query_students(get_db(), search, risk, sort, direction)
        return render_template(
            "students.html", rows=rows, q=search, risk=risk, sort=sort, direction=direction,
            total=db.count_students(get_db()),
        )

    @app.route("/students/<student_id>")
    @login_required
    def student_detail(student_id):
        row = db.get_student(get_db(), student_id)
        if row is None:
            abort(404)
        clean = {
            "attendance": row["attendance"],
            "internal_obtained": row["internal_obtained"],
            "internal_max": row["internal_max"],
            "internal_marks_pct": row["internal_marks_pct"],
            "assignment_status": row["assignment_status"],
            "previous_scale": row["previous_scale"],
            "previous_value": row["previous_value"],
            "previous_performance_pct": row["previous_performance_pct"],
        }
        probabilities = {}
        for label, key in (("Low Risk", "prob_low"), ("Moderate Risk", "prob_moderate"), ("High Risk", "prob_high")):
            if row[key] is not None:
                probabilities[label] = row[key]
        result = {"label": row["risk_level"], "confidence": row["confidence"], "probabilities": probabilities}
        return render_template("prediction.html", **_result_context(clean, result, student=row))

    @app.route("/students/<student_id>/delete", methods=["POST"])
    @login_required
    def delete_student(student_id):
        if db.delete_student(get_db(), student_id):
            flash(f"Student {student_id} deleted.", "info")
        else:
            flash("That student record was not found.", "error")
        return redirect(url_for("students"))

    # ------------------------------------------------------------------
    # Analytics, evaluation, reports
    # ------------------------------------------------------------------
    @app.route("/analytics")
    @login_required
    def analytics():
        ensure_demo_data()
        return render_template(
            "analytics.html", stats=db.summary(get_db()), metrics=predictor.load_metrics()
        )

    @app.route("/evaluation")
    @login_required
    def evaluation():
        return render_template("evaluation.html", metrics=predictor.load_metrics())

    @app.route("/report")
    @login_required
    def report():
        ensure_demo_data()
        conn = get_db()
        by_level = {label: db.query_students(conn, risk=label, sort="attendance", direction="asc")
                    for label in RISK_LABELS}
        return render_template(
            "report.html", stats=db.summary(conn), by_level=by_level, metrics=predictor.load_metrics()
        )

    @app.route("/about")
    @login_required
    def about():
        return render_template("about.html")

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------
    def _safe_cell(value):
        """Stop spreadsheet programs from running text that starts like a formula."""
        text = "" if value is None else str(value)
        return "'" + text if text[:1] in ("=", "+", "-", "@") else text

    @app.route("/export/csv")
    @login_required
    def export_csv():
        rows = db.query_students(get_db(), sort="student_id", direction="asc")
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow([
            "Student ID", "Student Name", "Attendance (%)", "Internal Marks Obtained",
            "Internal Marks Maximum", "Internal Marks (%)", "Assignment Status",
            "Previous Performance (%)", "Predicted Risk", "Confidence (%)", "Record Type", "Added On",
        ])
        for r in rows:
            writer.writerow([
                _safe_cell(r["student_id"]), _safe_cell(r["name"]), r["attendance"],
                r["internal_obtained"], r["internal_max"], r["internal_marks_pct"],
                r["assignment_status"], r["previous_performance_pct"], r["risk_level"],
                "" if r["confidence"] is None else r["confidence"],
                "Synthetic demo" if r["is_demo"] else "Entered by user", r["created_at"],
            ])
        filename = f"student_records_{datetime.now():%Y%m%d_%H%M}.csv"
        return Response(
            buffer.getvalue(), mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    # ------------------------------------------------------------------
    # Error pages
    # ------------------------------------------------------------------
    @app.errorhandler(400)
    def bad_request(error):
        return render_template("error.html", code=400, title="Request could not be processed",
                               message=getattr(error, "description", "")), 400

    @app.errorhandler(404)
    def not_found(_error):
        return render_template("error.html", code=404, title="Page not found",
                               message="The page or record you asked for does not exist."), 404

    @app.errorhandler(413)
    def too_large(_error):
        return render_template("error.html", code=413, title="Request too large",
                               message="The submitted form was larger than expected."), 413

    @app.errorhandler(500)
    def server_error(_error):
        return render_template("error.html", code=500, title="Something went wrong",
                               message="An unexpected error occurred. Check the terminal for details."), 500

    return app


app = create_app()

if __name__ == "__main__":
    # Set FLASK_DEBUG=1 while developing to get auto-reload (never on a shared network).
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", host="127.0.0.1", port=5000)
