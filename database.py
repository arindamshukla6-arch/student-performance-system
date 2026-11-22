"""
database.py
-----------
Small SQLite helper layer. Every query uses placeholders (?) so user input is
never pasted into SQL text.
"""

import csv
import sqlite3
from datetime import datetime
from pathlib import Path

from preprocessing import RISK_LABELS

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB = BASE_DIR / "data" / "students.db"
DEMO_CSV = BASE_DIR / "data" / "demo_students.csv"

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL DEFAULT '',
    attendance REAL NOT NULL,
    internal_obtained REAL NOT NULL,
    internal_max REAL NOT NULL,
    internal_marks_pct REAL NOT NULL,
    assignment_status TEXT NOT NULL,
    previous_scale TEXT NOT NULL DEFAULT 'percentage',
    previous_value REAL NOT NULL,
    previous_performance_pct REAL NOT NULL,
    risk_level TEXT NOT NULL,
    confidence REAL,
    prob_low REAL,
    prob_moderate REAL,
    prob_high REAL,
    is_demo INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
"""

# Columns the records table may be sorted by (whitelist -> safe ORDER BY).
SORTABLE = {
    "student_id": "student_id",
    "name": "name",
    "attendance": "attendance",
    "internal": "internal_marks_pct",
    "assignment": "assignment_status",
    "previous": "previous_performance_pct",
    "risk": "CASE risk_level WHEN 'Low Risk' THEN 1 WHEN 'Moderate Risk' THEN 2 ELSE 3 END",
    "added": "id",
}


def connect(path=None):
    conn = sqlite3.connect(str(path or DEFAULT_DB))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn):
    conn.executescript(SCHEMA)
    conn.commit()


def count_students(conn):
    return conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]


def student_exists(conn, student_id):
    return conn.execute(
        "SELECT 1 FROM students WHERE student_id = ?", (student_id,)
    ).fetchone() is not None


def next_student_id(conn):
    """Suggest the next free ID of the form STU001, STU002, ..."""
    rows = conn.execute(
        "SELECT student_id FROM students WHERE student_id LIKE 'STU%'"
    ).fetchall()
    numbers = [int(r[0][3:]) for r in rows if r[0][3:].isdigit()]
    return f"STU{(max(numbers) + 1 if numbers else 1):03d}"


def insert_student(conn, identity, clean, prediction, is_demo=False):
    probs = prediction.get("probabilities", {})
    conn.execute(
        """INSERT INTO students (student_id, name, attendance, internal_obtained,
               internal_max, internal_marks_pct, assignment_status, previous_scale,
               previous_value, previous_performance_pct, risk_level, confidence,
               prob_low, prob_moderate, prob_high, is_demo, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            identity["student_id"],
            identity["name"],
            clean["attendance"],
            clean["internal_obtained"],
            clean["internal_max"],
            clean["internal_marks_pct"],
            clean["assignment_status"],
            clean["previous_scale"],
            clean["previous_value"],
            clean["previous_performance_pct"],
            prediction["label"],
            prediction.get("confidence"),
            probs.get("Low Risk"),
            probs.get("Moderate Risk"),
            probs.get("High Risk"),
            1 if is_demo else 0,
            datetime.now().strftime("%Y-%m-%d %H:%M"),
        ),
    )
    conn.commit()


def seed_demo_students(conn, predict_fn):
    """Fill an empty database with the SYNTHETIC demo records.

    `predict_fn` is the real model prediction function, so the risk shown for
    each demo student comes from the trained model, not from a hard-coded value.
    """
    if count_students(conn) > 0 or not DEMO_CSV.exists():
        return 0
    added = 0
    with open(DEMO_CSV, newline="") as handle:
        for row in csv.DictReader(handle):
            pct = float(row["previous_performance_pct"])
            max_marks = float(row["internal_max"])
            obtained = float(row["internal_obtained"])
            clean = {
                "attendance": float(row["attendance"]),
                "internal_obtained": obtained,
                "internal_max": max_marks,
                "internal_marks_pct": round(obtained / max_marks * 100, 1),
                "assignment_status": row["assignment_status"],
                "previous_scale": "percentage",
                "previous_value": pct,
                "previous_performance_pct": pct,
            }
            identity = {"student_id": row["student_id"], "name": row["name"]}
            insert_student(conn, identity, clean, predict_fn(clean), is_demo=True)
            added += 1
    return added


def query_students(conn, search="", risk="", sort="added", direction="desc"):
    """Return students filtered by search text / risk level and sorted."""
    sql = "SELECT * FROM students WHERE 1=1"
    params = []
    if search:
        escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        sql += " AND (student_id LIKE ? ESCAPE '\\' OR name LIKE ? ESCAPE '\\')"
        params += [f"%{escaped}%", f"%{escaped}%"]
    if risk in RISK_LABELS:
        sql += " AND risk_level = ?"
        params.append(risk)
    column = SORTABLE.get(sort, SORTABLE["added"])
    order = "ASC" if direction == "asc" else "DESC"
    sql += f" ORDER BY {column} {order}, id DESC"  # column comes from the whitelist only
    return conn.execute(sql, params).fetchall()


def get_student(conn, student_id):
    return conn.execute(
        "SELECT * FROM students WHERE student_id = ?", (student_id,)
    ).fetchone()


def delete_student(conn, student_id):
    cur = conn.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
    conn.commit()
    return cur.rowcount > 0


def recent_students(conn, limit=6):
    return conn.execute(
        "SELECT * FROM students ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()


# ---------------------------------------------------------------------------
# Aggregates for the dashboard, analytics and report pages
# ---------------------------------------------------------------------------
def _avg(values):
    values = list(values)
    return round(sum(values) / len(values), 1) if values else 0


def _band(value, edges, labels):
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


ATTENDANCE_BANDS = ["Below 65%", "65-74%", "75-84%", "85% and above"]
MARKS_BANDS = ["Below 40%", "40-54%", "55-69%", "70% and above"]


def summary(conn):
    rows = conn.execute("SELECT * FROM students").fetchall()
    counts = {label: 0 for label in RISK_LABELS}
    for r in rows:
        counts[r["risk_level"]] += 1

    by_risk = {}
    for label in RISK_LABELS:
        group = [r for r in rows if r["risk_level"] == label]
        by_risk[label] = {
            "attendance": _avg(r["attendance"] for r in group),
            "internal": _avg(r["internal_marks_pct"] for r in group),
            "previous": _avg(r["previous_performance_pct"] for r in group),
        }

    attendance_bands = {b: {l: 0 for l in RISK_LABELS} for b in ATTENDANCE_BANDS}
    marks_bands = {b: {l: 0 for l in RISK_LABELS} for b in MARKS_BANDS}
    for r in rows:
        a = _band(r["attendance"], [65, 75, 85], ATTENDANCE_BANDS)
        m = _band(r["internal_marks_pct"], [40, 55, 70], MARKS_BANDS)
        attendance_bands[a][r["risk_level"]] += 1
        marks_bands[m][r["risk_level"]] += 1

    total = len(rows)
    completed = sum(1 for r in rows if r["assignment_status"] == "Completed")
    return {
        "total": total,
        "counts": counts,
        "percent": {k: round(v / total * 100, 1) if total else 0 for k, v in counts.items()},
        "avg_attendance": _avg(r["attendance"] for r in rows),
        "avg_internal": _avg(r["internal_marks_pct"] for r in rows),
        "avg_previous": _avg(r["previous_performance_pct"] for r in rows),
        "assignment_completion": round(completed / total * 100, 1) if total else 0,
        "below_75": sum(1 for r in rows if r["attendance"] < 75),
        "by_risk": by_risk,
        "attendance_bands": attendance_bands,
        "marks_bands": marks_bands,
    }
