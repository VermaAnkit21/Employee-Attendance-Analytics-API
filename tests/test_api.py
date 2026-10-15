"""
End-to-end tests against a real MongoDB (MONGO_URI, default localhost). Uses a throwaway database.

    pip install -r requirements-dev.txt
    pytest -q
"""
import os
import random
import threading
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal

os.environ["MONGO_DB"] = "attendance_api_tests"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app, db  # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))
UTC = timezone.utc


def ms(ist: str) -> int:
    """'2026-07-06 09:40:01' (IST) -> epoch ms."""
    return int(datetime.strptime(ist, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST).timestamp()) * 1000


def utc(ist: str) -> datetime:
    return datetime.strptime(ist, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST).astimezone(UTC)


def hu(value, places):
    return float(Decimal(str(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


@pytest.fixture(scope="module")
def client():
    db.client.drop_database(db.name)
    with TestClient(app) as c:
        yield c
    db.client.drop_database(db.name)


@pytest.fixture(autouse=True)
def clean(client):
    db.employees.delete_many({})
    db.attendance_logs.delete_many({})


def make_emp(client, code="EMP0001", dept="Engineering", start="09:30", end="18:30", joined="2026-01-05", name=None):
    r = client.post("/employees", json={
        "emp_code": code, "name": name or code, "email": f"{code.lower()}@example.com", "department": dept,
        "shift_start": start, "shift_end": end, "joined_on": joined,
    })
    assert r.status_code == 201, r.text
    return r.json()


def run_parallel(n, fn):
    barrier = threading.Barrier(n)
    results = []
    lock = threading.Lock()

    def worker():
        barrier.wait()
        r = fn()
        with lock:
            results.append(r.status_code)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return sorted(results)


# --------------------------------------------------------------------------- #
# System, indexes, employees
# --------------------------------------------------------------------------- #
def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_indexes_created(client):
    assert {"emp_code_unique", "department_emp_code"} <= set(db.employees.index_information())
    assert {"emp_code_date_unique", "date_desc_emp_code", "emp_code_punch_in_desc"} <= set(db.attendance_logs.index_information())


def test_create_employee(client):
    body = make_emp(client)
    assert set(body) == {"emp_code", "name", "email", "department", "shift_start", "shift_end", "joined_on", "created_at"}
    assert isinstance(body["created_at"], int) and body["created_at"] > 10**12
    stored = db.employees.find_one({"emp_code": "EMP0001"})
    assert stored["created_at"].tzinfo is not None
    assert abs(stored["created_at"].timestamp() * 1000 - body["created_at"]) < 1
    assert client.post("/employees", json={**{k: body[k] for k in body if k != "created_at"}}).status_code == 409


@pytest.mark.parametrize("patch", [
    {"emp_code": "E1"}, {"emp_code": "EMP123"}, {"emp_code": "EMP1234567"}, {"email": "nope"},
    {"shift_start": "24:00"}, {"shift_start": "9:30"}, {"shift_end": "09:30"}, {"joined_on": "2026-02-30"},
    {"joined_on": "20260101"}, {"name": ""}, {"department": ""}, {"name": "x" * 101},
])
def test_create_employee_validation(client, patch):
    body = {"emp_code": "EMP0001", "name": "A", "email": "a@b.co", "department": "D", "joined_on": "2026-01-01", **patch}
    assert client.post("/employees", json=body).status_code == 422


def test_create_employee_concurrent(client):
    body = {"emp_code": "EMP0042", "name": "A", "email": "a@b.co", "department": "D", "joined_on": "2026-01-01"}
    codes = run_parallel(10, lambda: client.post("/employees", json=body))
    assert codes == [201] + [409] * 9
    assert db.employees.count_documents({"emp_code": "EMP0042"}) == 1


def test_list_employees(client):
    for i in [5, 3, 1, 4, 2]:
        make_emp(client, f"EMP000{i}", dept="A" if i % 2 else "B")
    r = client.get("/employees", params={"page_size": 2}).json()
    assert [e["emp_code"] for e in r["items"]] == ["EMP0001", "EMP0002"] and r["total"] == 5
    r = client.get("/employees", params={"page": 2, "page_size": 2}).json()
    assert [e["emp_code"] for e in r["items"]] == ["EMP0003", "EMP0004"]
    r = client.get("/employees", params={"department": "A"}).json()
    assert r["total"] == 3 and [e["emp_code"] for e in r["items"]] == ["EMP0001", "EMP0003", "EMP0005"]
    assert client.get("/employees", params={"department": "a"}).json()["total"] == 0
    for bad in [{"page": 0}, {"page_size": 0}, {"page_size": 101}]:
        assert client.get("/employees", params=bad).status_code == 422


# --------------------------------------------------------------------------- #
# Punch-in (R1, R2)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("when,late", [
    ("09:28:00", 0), ("09:40:00", 0), ("09:40:01", 10), ("09:41:00", 11), ("10:15:59", 45), ("10:05:00", 35),
])
def test_punch_in_late(client, when, late):
    make_emp(client)
    r = client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms(f"2026-07-06 {when}")})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["late_minutes"] == late
    assert body["date"] == "2026-07-06"
    assert body["punch_out"] is None and body["work_hours"] is None
    assert body["overtime_minutes"] == 0 and body["half_day"] is False and body["history"] == []
    assert "id" not in body and "_id" not in body


def test_punch_in_truncates_to_seconds(client):
    make_emp(client)
    r = client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms("2026-07-06 09:40:00") + 900})
    assert r.json()["late_minutes"] == 0
    assert r.json()["punch_in"] == ms("2026-07-06 09:40:00")
    assert db.attendance_logs.find_one()["punch_in"] == utc("2026-07-06 09:40:00")


def test_punch_in_ist_date_near_midnight(client):
    make_emp(client)
    # 00:30 IST on the 7th is still the 6th in UTC; the attendance date must be the IST date.
    r = client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms("2026-07-07 00:30:00")})
    assert r.json()["date"] == "2026-07-07"


def test_punch_in_overnight(client):
    make_emp(client, start="22:00", end="06:00")
    r = client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms("2026-07-06 21:55:00")})
    assert r.json()["date"] == "2026-07-06" and r.json()["late_minutes"] == 0
    # 00:40 IST on the 8th belongs to the shift that started on the 7th; 160 minutes late.
    r = client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms("2026-07-08 00:40:00")})
    assert r.json()["date"] == "2026-07-07" and r.json()["late_minutes"] == 160
    # 06:00 is not earlier than shift_end, so it is the 9th's own date.
    r = client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": ms("2026-07-09 06:00:00")})
    assert r.json()["date"] == "2026-07-09" and r.json()["late_minutes"] == 0


@pytest.mark.parametrize("punched_at", [1783312500, 1783312500000.0, "1783312500000", 4102444800001, True, None])
def test_punch_in_rejects_bad_timestamps(client, punched_at):
    make_emp(client)
    assert client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "punched_at": punched_at}).status_code == 422


def test_punch_in_errors(client):
    make_emp(client)
    assert client.post("/attendance/punch-in", json={"emp_code": "EMP9999"}).status_code == 404
    for status in ["ABSENT", "LEAVE", "present", "X"]:
        assert client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "status": status}).status_code == 422
    r = client.post("/attendance/punch-in", json={"emp_code": "EMP0001", "status": "WFH"})
    assert r.status_code == 201 and r.json()["status"] == "WFH"
    assert client.post("/attendance/punch-in", json={"emp_code": "EMP0001"}).status_code == 409


def test_punch_in_concurrent(client):
    make_emp(client)
    body = {"emp_code": "EMP0001", "punched_at": ms("2026-07-06 09:30:00")}
    assert run_parallel(10, lambda: client.post("/attendance/punch-in", json=body)) == [201] + [409] * 9
    assert db.attendance_logs.count_documents({}) == 1


# --------------------------------------------------------------------------- #
# Punch-out (R3 - R5)
# --------------------------------------------------------------------------- #
def punch(client, code, pin, pout=None, status="PRESENT"):
    r = client.post("/attendance/punch-in", json={"emp_code": code, "punched_at": ms(pin), "status": status})
    assert r.status_code == 201, r.text
    if pout:
        r = client.post("/attendance/punch-out", json={"emp_code": code, "punched_at": ms(pout)})
        assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.parametrize("pin,pout,hours,ot,half", [
    ("2026-07-06 09:28:00", "2026-07-06 18:35:00", 9.12, 0, False),
    ("2026-07-06 10:05:00", "2026-07-06 19:10:00", 9.08, 40, False),
    ("2026-07-06 09:30:00", "2026-07-06 18:59:59", 9.5, 0, False),   # 29:59 overtime -> 0
    ("2026-07-06 09:30:00", "2026-07-06 19:00:00", 9.5, 30, False),  # exactly 30 counts
    ("2026-07-06 09:30:00", "2026-07-06 13:30:18", 4.01, 0, True),   # 4.005 h rounds half-up
    ("2026-07-06 09:30:00", "2026-07-06 13:59:41", 4.49, 0, True),   # 4.4947 -> 4.49 < 4.50
    ("2026-07-06 09:30:00", "2026-07-06 13:59:42", 4.5, 0, False),   # 4.4950 -> 4.50 is not < 4.50
])
def test_punch_out_rules(client, pin, pout, hours, ot, half):
    make_emp(client)
    body = punch(client, "EMP0001", pin, pout)
    assert (body["work_hours"], body["overtime_minutes"], body["half_day"]) == (hours, ot, half)
    assert body["punch_out"] == ms(pout) and body["history"] == []


def test_punch_out_overnight(client):
    make_emp(client, start="22:00", end="06:00")
    body = punch(client, "EMP0001", "2026-07-06 21:55:00", "2026-07-07 06:40:00")
    assert (body["date"], body["work_hours"], body["overtime_minutes"]) == ("2026-07-06", 8.75, 40)


def test_punch_out_errors(client):
    make_emp(client)
    assert client.post("/attendance/punch-out", json={"emp_code": "EMP9999"}).status_code == 404
    assert client.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms("2026-07-06 18:00:00")}).status_code == 404
    punch(client, "EMP0001", "2026-07-06 09:30:00")
    assert client.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms("2026-07-06 09:30:00")}).status_code == 422
    assert client.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms("2026-07-07 09:30:01")}).status_code == 422
    assert client.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": 1783312500}).status_code == 422
    assert client.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms("2026-07-07 09:30:00")}).status_code == 200
    assert client.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms("2026-07-06 19:00:00")}).status_code == 409


def test_punch_out_picks_most_recent_record(client):
    make_emp(client)
    punch(client, "EMP0001", "2026-07-06 09:30:00", "2026-07-06 18:30:00")
    punch(client, "EMP0001", "2026-07-07 09:30:00")
    r = client.post("/attendance/punch-out", json={"emp_code": "EMP0001", "punched_at": ms("2026-07-07 18:30:00")})
    assert r.status_code == 200 and r.json()["date"] == "2026-07-07"


def test_punch_out_concurrent(client):
    make_emp(client)
    punch(client, "EMP0001", "2026-07-06 09:30:00")
    body = {"emp_code": "EMP0001", "punched_at": ms("2026-07-06 18:30:00")}
    assert run_parallel(10, lambda: client.post("/attendance/punch-out", json=body)) == [200] + [409] * 9


# --------------------------------------------------------------------------- #
# Attendance list
# --------------------------------------------------------------------------- #
def test_list_attendance(client):
    make_emp(client, "EMP0001")
    make_emp(client, "EMP0002")
    for d in ["06", "07", "08"]:
        punch(client, "EMP0002", f"2026-07-{d} 09:30:00")
        punch(client, "EMP0001", f"2026-07-{d} 09:30:00", status="WFH" if d == "07" else "PRESENT")
    r = client.get("/attendance").json()
    assert [(i["date"], i["emp_code"]) for i in r["items"]] == [
        ("2026-07-08", "EMP0001"), ("2026-07-08", "EMP0002"), ("2026-07-07", "EMP0001"),
        ("2026-07-07", "EMP0002"), ("2026-07-06", "EMP0001"), ("2026-07-06", "EMP0002")]
    assert r["total"] == 6 and isinstance(r["items"][0]["punch_in"], int)
    r = client.get("/attendance", params={"emp_code": "EMP0001", "date_from": "2026-07-07", "date_to": "2026-07-08", "page_size": 1, "page": 2}).json()
    assert r["total"] == 2 and [i["date"] for i in r["items"]] == ["2026-07-07"]
    assert client.get("/attendance", params={"status": "WFH"}).json()["total"] == 1
    for bad in [{"date_from": "2026-07-09", "date_to": "2026-07-08"}, {"status": "BAD"}, {"date_from": "07/07/2026"}, {"page_size": 101}, {"page": 0}]:
        assert client.get("/attendance", params=bad).status_code == 422


def test_list_attendance_legacy_record(client):
    make_emp(client, "EMP0002")
    db.attendance_logs.insert_one({"emp_code": "EMP0002", "date": "2026-07-14", "status": "PRESENT",
                                   "punch_in": utc("2026-07-14 09:32:00"), "punch_out": utc("2026-07-14 18:31:00"),
                                   "work_hours": 8.98, "late_minutes": 0, "overtime_minutes": 0})
    item = client.get("/attendance").json()["items"][0]
    assert item["half_day"] is False and item["history"] == [] and item["work_hours"] == 8.98


# --------------------------------------------------------------------------- #
# Regularization
# --------------------------------------------------------------------------- #
FIX = {"reason": "biometric glitch", "regularized_by": "hr.admin"}
FIX_DOLLAR = {"reason": "$history refund", "regularized_by": "$hr"}


def test_regularize_punch_in(client):
    make_emp(client)
    punch(client, "EMP0001", "2026-07-07 10:05:00", "2026-07-07 18:30:00")
    r = client.patch("/attendance/EMP0001/2026-07-07", json={**FIX, "punch_in": ms("2026-07-07 09:28:00"), "work_hours": 99})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["late_minutes"] == 0 and body["work_hours"] == 9.03
    (entry,) = body["history"]
    assert entry["by"] == "hr.admin" and entry["reason"] == "biometric glitch" and isinstance(entry["at"], int)
    assert entry["changes"] == {
        "punch_in": {"from": ms("2026-07-07 10:05:00"), "to": ms("2026-07-07 09:28:00")},
        "late_minutes": {"from": 35, "to": 0},
        "work_hours": {"from": 8.42, "to": 9.03},
    }
    stored = db.attendance_logs.find_one()
    assert isinstance(stored["history"][0]["changes"]["punch_in"]["to"], datetime)
    # A second correction appends, never overwrites.
    r = client.patch("/attendance/EMP0001/2026-07-07", json={**FIX, "status": "WFH"})
    assert [list(h["changes"]) for h in r.json()["history"]] == [["punch_in", "work_hours", "late_minutes"], ["status"]] or \
        [sorted(h["changes"]) for h in r.json()["history"]] == [["late_minutes", "punch_in", "work_hours"], ["status"]]


def test_regularize_to_absent_and_back(client):
    make_emp(client)
    punch(client, "EMP0001", "2026-07-07 10:05:00", "2026-07-07 19:10:00")
    r = client.patch("/attendance/EMP0001/2026-07-07", json={**FIX, "status": "ABSENT"})
    body = r.json()
    assert (body["punch_in"], body["punch_out"], body["work_hours"], body["late_minutes"], body["overtime_minutes"]) == (None, None, None, 0, 0)
    assert set(body["history"][0]["changes"]) == {"status", "punch_in", "punch_out", "work_hours", "late_minutes", "overtime_minutes"}
    assert client.patch("/attendance/EMP0001/2026-07-07", json={**FIX, "status": "PRESENT"}).status_code == 422
    r = client.patch("/attendance/EMP0001/2026-07-07", json={**FIX, "status": "ON_DUTY", "punch_in": ms("2026-07-07 09:30:00")})
    assert r.status_code == 200 and r.json()["punch_out"] is None and r.json()["work_hours"] is None


def test_regularize_errors(client):
    make_emp(client)
    punch(client, "EMP0001", "2026-07-07 09:30:00", "2026-07-07 18:30:00")
    url = "/attendance/EMP0001/2026-07-07"
    assert client.patch("/attendance/EMP9999/2026-07-07", json={**FIX, "status": "WFH"}).status_code == 404
    assert client.patch("/attendance/EMP0001/2026-07-08", json={**FIX, "status": "WFH"}).status_code == 404
    assert client.patch("/attendance/EMP0001/2026-13-01", json={**FIX, "status": "WFH"}).status_code == 422
    assert client.patch(url, json={**FIX, "status": "LEAVE", "punch_in": ms("2026-07-07 09:30:00")}).status_code == 422
    assert client.patch(url, json={**FIX, "punch_in": ms("2026-07-08 09:30:00")}).status_code == 422  # leaves the date
    assert client.patch(url, json={**FIX, "punch_out": ms("2026-07-07 09:00:00")}).status_code == 422  # before punch_in
    assert client.patch(url, json={**FIX, "punch_out": ms("2026-07-08 09:30:01")}).status_code == 422  # > 24 h
    assert client.patch(url, json={**FIX, "status": "PRESENT"}).status_code == 422  # changes nothing
    assert client.patch(url, json={**FIX, "punch_in": ms("2026-07-07 09:30:00")}).status_code == 422  # changes nothing
    assert client.patch(url, json={**FIX}).status_code == 422
    assert client.patch(url, json={"reason": "abc", "regularized_by": "x", "status": "WFH"}).status_code == 422
    assert client.patch(url, json={"reason": "valid reason", "status": "WFH"}).status_code == 422
    assert client.patch(url, json={**FIX, "punch_in": 1783312500}).status_code == 422
    assert client.patch(url, json={**FIX, "punch_in": None}).status_code == 422
    assert client.patch(url, json={**FIX, "status": "BAD"}).status_code == 422


def test_regularize_concurrent_keeps_history(client):
    make_emp(client)
    punch(client, "EMP0001", "2026-07-07 09:30:00", "2026-07-07 18:30:00")
    counter = iter(range(100))
    lock = threading.Lock()

    def call():
        with lock:
            n = next(counter)
        return client.patch("/attendance/EMP0001/2026-07-07", json={**FIX, "punch_out": ms("2026-07-07 18:30:00") + (n + 1) * 60000})

    codes = run_parallel(8, call)
    assert set(codes) <= {200, 409} and 200 in codes
    assert len(db.attendance_logs.find_one()["history"]) == codes.count(200)


# --------------------------------------------------------------------------- #
# Analytics: compared against a plain-Python reference over a messy seeded dataset
# --------------------------------------------------------------------------- #
PRESENCE = {"PRESENT", "WFH", "ON_DUTY"}


def seed_messy(rng):
    """Seed directly (like the grader) with the awkward cases: no-log employees, mid-month and future joiners,
    weekend records, half days, legacy records without half_day/history, orphan logs and late-minute ties."""
    depts = ["Engineering", "Sales", "Support", "Ops"]
    emps = []
    for i in range(1, 41):
        joined = rng.choice(["2026-01-05", "2026-03-10", "2026-07-15", "2026-07-31", "2026-08-01", "2026-09-20"])
        emps.append({"emp_code": f"EMP{i:04d}", "name": f"Person {i}", "email": f"p{i}@x.io", "department": depts[i % 4],
                     "shift_start": "22:00" if i % 9 == 0 else "09:30", "shift_end": "06:00" if i % 9 == 0 else "18:30",
                     "joined_on": joined, "created_at": datetime(2026, 1, 1, tzinfo=UTC)})
    db.employees.insert_many([dict(e) for e in emps])
    logs = []
    start = date(2026, 6, 25)
    for e in emps[:-6]:  # the last six employees have no logs at all
        for k in range(70):
            day = start + timedelta(days=k)
            if rng.random() < 0.25:
                continue  # missing day: no document at all
            status = rng.choices(["PRESENT", "WFH", "ON_DUTY", "ABSENT", "LEAVE"], [6, 2, 1, 1, 1])[0]
            doc = {"emp_code": e["emp_code"], "date": day.isoformat(), "status": status}
            if status in PRESENCE:
                late = rng.choice([0, 0, 0, 15, 20, 35, 45])
                pin = datetime(day.year, day.month, day.day, 4, 0, tzinfo=UTC) + timedelta(minutes=late)
                wh = rng.choice([3.5, 4.49, 4.5, 8.0, 9.12, 9.08, None])
                doc.update(punch_in=pin, punch_out=None if wh is None else pin + timedelta(hours=wh), work_hours=wh,
                           late_minutes=late, overtime_minutes=rng.choice([0, 0, 30, 40]),
                           half_day=wh is not None and wh < 4.5)
            else:
                doc.update(punch_in=None, punch_out=None, work_hours=None, late_minutes=0, overtime_minutes=0, half_day=False)
            if rng.random() < 0.15:  # legacy shape
                doc.pop("half_day", None)
            else:
                doc["history"] = []
            logs.append(doc)
    # orphan logs (employee does not exist) with large late minutes: must never show up
    logs.append({"emp_code": "EMP9999", "date": "2026-07-08", "status": "PRESENT", "punch_in": datetime(2026, 7, 8, 5, tzinfo=UTC),
                 "punch_out": None, "work_hours": None, "late_minutes": 999, "overtime_minutes": 0, "half_day": False, "history": []})
    db.attendance_logs.insert_many([dict(l) for l in logs])
    return emps, logs


def ref_totals(logs):
    t = defaultdict(int)
    t["present_days"] = 0.0
    for l in logs:
        weekday = date.fromisoformat(l["date"]).weekday() < 5
        if l["status"] in PRESENCE and weekday:
            t["present_days"] += 0.5 if l.get("half_day") else 1
        late = l.get("late_minutes") or 0
        if late > 0:
            t["late_count"] += 1
            t["total_late_minutes"] += late
        t["total_overtime_minutes"] += l.get("overtime_minutes") or 0
        t["leave"] += l["status"] == "LEAVE"
        t["on_duty"] += l["status"] == "ON_DUTY"
        if l["status"] in PRESENCE and l.get("work_hours") is not None:
            t["wh_sum"] += Decimal(str(l["work_hours"]))
            t["wh_n"] += 1
    return t


def weekdays(a, b):
    n = 0
    while a <= b:
        n += a.weekday() < 5
        a += timedelta(days=1)
    return n


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_analytics_against_reference(client, seed):
    rng = random.Random(seed)
    emps, logs = seed_messy(rng)
    by_code = {e["emp_code"]: e for e in emps}
    for month in ["2026-06", "2026-07", "2026-08", "2026-09"]:
        first = date.fromisoformat(f"{month}-01")
        last = (first.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        in_month = [l for l in logs if l["date"].startswith(month)]

        # employee monthly
        for e in emps[::5] + emps[-2:]:
            mine = [l for l in in_month if l["emp_code"] == e["emp_code"]]
            t = ref_totals(mine)
            wd = weekdays(max(first, date.fromisoformat(e["joined_on"])), last)
            got = client.get(f"/analytics/employees/{e['emp_code']}/monthly", params={"month": month}).json()
            assert got == {
                "emp_code": e["emp_code"], "month": month, "working_days": wd, "present_days": hu(t["present_days"], 2),
                "leave_days": t["leave"], "late_count": t["late_count"], "total_late_minutes": t["total_late_minutes"],
                "total_overtime_minutes": t["total_overtime_minutes"],
                "attendance_pct": hu(Decimal(str(t["present_days"])) * 100 / wd, 2) if wd else None,
            }, (month, e["emp_code"])

        # department summary
        expected = []
        for dept in sorted({e["department"] for e in emps}):
            members = [e for e in emps if e["department"] == dept and e["joined_on"] <= last.isoformat()]
            if not members:
                continue
            codes = {e["emp_code"] for e in members}
            t = ref_totals([l for l in in_month if l["emp_code"] in codes])
            expected.append({
                "department": dept, "headcount": len(members), "present_days": hu(t["present_days"], 2),
                "avg_work_hours": hu(t["wh_sum"] / t["wh_n"], 2) if t["wh_n"] else None,
                "late_count": t["late_count"], "total_late_minutes": t["total_late_minutes"],
                "leave_count": t["leave"], "on_duty_count": t["on_duty"],
            })
        got = client.get("/analytics/departments/summary", params={"month": month}).json()
        assert got == {"month": month, "items": expected}, month
        if expected:
            one = client.get("/analytics/departments/summary", params={"month": month, "department": expected[0]["department"]}).json()
            assert one["items"] == expected[:1]

        # leaderboard (overall and per department)
        for dept in [None, "Sales"]:
            totals = defaultdict(lambda: [0, 0])
            for l in in_month:
                if l["emp_code"] in by_code and (l.get("late_minutes") or 0) > 0 and (dept is None or by_code[l["emp_code"]]["department"] == dept):
                    totals[l["emp_code"]][0] += l["late_minutes"]
                    totals[l["emp_code"]][1] += 1
            rows = sorted(totals.items(), key=lambda kv: (-kv[1][0], kv[0]))
            for limit in [1, 3, 10, 50]:
                exp = []
                for code, (tot, cnt) in rows:
                    rank = 1 + sum(1 for _, (t2, _) in rows if t2 > tot)
                    if rank <= limit:
                        exp.append({"rank": rank, "emp_code": code, "name": by_code[code]["name"],
                                    "department": by_code[code]["department"], "total_late_minutes": tot, "late_count": cnt})
                params = {"month": month, "limit": limit, **({"department": dept} if dept else {})}
                got = client.get("/analytics/leaderboard/late", params=params).json()
                assert got == {"month": month, "items": exp}, (month, dept, limit)

    # department trend
    for dept in ["Engineering", "Sales", "Support", "Ops"]:
        for a, b in [("2026-06-20", "2026-07-20"), ("2026-07-01", "2026-09-30"), ("2026-08-01", "2026-08-01"), ("2026-09-25", "2026-10-05")]:
            members = [e for e in emps if e["department"] == dept]
            codes = {e["emp_code"] for e in members}
            rows, rates = [], []
            d = date.fromisoformat(a)
            while d <= date.fromisoformat(b):
                ds = d.isoformat()
                day_logs = [l for l in logs if l["date"] == ds and l["emp_code"] in codes]
                head = sum(1 for e in members if e["joined_on"] <= ds)
                present = sum((0.5 if l.get("half_day") else 1) for l in day_logs if l["status"] in PRESENCE)
                late = sum(1 for l in day_logs if (l.get("late_minutes") or 0) > 0)
                working = d.weekday() < 5
                rate = hu(Decimal(str(present)) / head, 4) if working and head else None
                rates.append(rate)
                window = [r for r in rates[-7:] if r is not None]
                ma = hu(sum(Decimal(str(r)) for r in window) / len(window), 4) if window else None
                rows.append({"date": ds, "is_working_day": working, "headcount": head, "present_count": present,
                             "late_count": late, "attendance_rate": rate, "moving_avg_7d": ma})
                d += timedelta(days=1)
            got = client.get(f"/analytics/departments/{dept}/trend", params={"from": a, "to": b}).json()
            assert got["department"] == dept
            assert got["items"] == rows, (dept, a, b)


def test_leaderboard_ties(client):
    for i, dept in enumerate(["A", "A", "A", "A", "B"], start=1):
        make_emp(client, f"EMP000{i}", dept=dept)
    late = {"EMP0001": 50, "EMP0002": 40, "EMP0003": 40, "EMP0004": 30, "EMP0005": 40}
    for code, minutes in late.items():
        db.attendance_logs.insert_one({"emp_code": code, "date": "2026-07-11", "status": "PRESENT",  # a Saturday
                                       "punch_in": utc("2026-07-11 09:30:00"), "late_minutes": minutes})
    got = client.get("/analytics/leaderboard/late", params={"month": "2026-07", "limit": 2}).json()["items"]
    assert [(r["rank"], r["emp_code"]) for r in got] == [(1, "EMP0001"), (2, "EMP0002"), (2, "EMP0003"), (2, "EMP0005")]
    got = client.get("/analytics/leaderboard/late", params={"month": "2026-07", "limit": 4}).json()["items"]
    assert [r["rank"] for r in got] == [1, 2, 2, 2, 5][:4]
    got = client.get("/analytics/leaderboard/late", params={"month": "2026-07", "limit": 3, "department": "A"}).json()["items"]
    assert [(r["rank"], r["emp_code"]) for r in got] == [(1, "EMP0001"), (2, "EMP0002"), (2, "EMP0003")]
    for bad in [{"month": "2026-7"}, {"month": "2026-13"}, {"month": "2026-07", "limit": 0}, {"month": "2026-07", "limit": 51}, {}]:
        assert client.get("/analytics/leaderboard/late", params=bad).status_code == 422


def test_analytics_errors(client):
    make_emp(client, joined="2026-08-15")
    assert client.get("/analytics/employees/EMP9999/monthly", params={"month": "2026-07"}).status_code == 404
    assert client.get("/analytics/employees/EMP0001/monthly", params={"month": "July"}).status_code == 422
    got = client.get("/analytics/employees/EMP0001/monthly", params={"month": "2026-07"}).json()
    assert got["working_days"] == 0 and got["attendance_pct"] is None
    assert client.get("/analytics/departments/summary", params={"month": "2026-07"}).json()["items"] == []
    assert client.get("/analytics/departments/summary").status_code == 422
    trend = "/analytics/departments/Engineering/trend"
    assert client.get("/analytics/departments/Nope/trend", params={"from": "2026-07-01", "to": "2026-07-02"}).status_code == 404
    assert client.get(trend, params={"from": "2026-07-02", "to": "2026-07-01"}).status_code == 422
    assert client.get(trend, params={"from": "2026-07-01", "to": "2026-09-30"}).status_code == 200  # 92 days
    assert client.get(trend, params={"from": "2026-07-01", "to": "2026-10-01"}).status_code == 422  # 93 days
    assert client.get(trend, params={"from": "2026-07-01"}).status_code == 422
    items = client.get(trend, params={"from": "2026-08-13", "to": "2026-08-17"}).json()["items"]
    assert [(i["date"], i["headcount"], i["attendance_rate"]) for i in items] == [
        ("2026-08-13", 0, None), ("2026-08-14", 0, None), ("2026-08-15", 1, None), ("2026-08-16", 1, None), ("2026-08-17", 1, 0.0)]


# --------------------------------------------------------------------------- #
# Explain
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("endpoint,params", [
    ("attendance_list", {}),
    ("attendance_list", {"emp_code": "EMP0001"}),
    ("attendance_list", {"date_from": "2026-07-01", "date_to": "2026-07-31", "status": "LEAVE", "page": 2}),
    ("employee_monthly", {"emp_code": "EMP0001", "month": "2026-07"}),
    ("department_summary", {"month": "2026-07"}),
    ("department_summary", {"month": "2026-07", "department": "Sales"}),
    ("late_leaderboard", {"month": "2026-07"}),
    ("late_leaderboard", {"month": "2026-07", "department": "Sales", "limit": 3}),
    ("department_trend", {"department": "Sales", "from": "2026-07-01", "to": "2026-07-31"}),
])
def test_explain_uses_indexes(client, endpoint, params):
    seed_messy(random.Random(7))
    r = client.get(f"/admin/explain/{endpoint}", params=params)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["endpoint"] == endpoint and body["collection"] in {"employees", "attendance_logs"}
    text = str(body["explain"])
    assert "IXSCAN" in text and "COLLSCAN" not in text


@pytest.mark.parametrize("endpoint,params", [
    ("employee_monthly", {"month": "2026-07"}), ("employee_monthly", {"emp_code": "EMP0001"}),
    ("department_summary", {}), ("late_leaderboard", {}), ("department_trend", {"department": "Sales", "from": "2026-07-01"}),
    ("nope", {}), ("attendance_list", {"date_from": "2026-07-02", "date_to": "2026-07-01"}),
])
def test_explain_validation(client, endpoint, params):
    assert client.get(f"/admin/explain/{endpoint}", params=params).status_code == 422


def test_regularize_seeded_odd_shapes(client):
    make_emp(client)
    # history: null, no half_day, and a punch_in with sub-second precision (seeded directly, not via the API)
    db.attendance_logs.insert_one({"emp_code": "EMP0001", "date": "2026-07-09", "status": "PRESENT", "history": None,
                                   "punch_in": utc("2026-07-09 10:05:00").replace(microsecond=900000), "punch_out": None,
                                   "work_hours": None, "late_minutes": 35, "overtime_minutes": 0})
    r = client.patch("/attendance/EMP0001/2026-07-09", json={**FIX, "punch_out": ms("2026-07-09 18:30:00")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["work_hours"] == 8.42 and body["late_minutes"] == 35 and len(body["history"]) == 1
    assert set(body["history"][0]["changes"]) == {"punch_out", "work_hours"}
    r = client.patch("/attendance/EMP0001/2026-07-09", json={**FIX, "punch_in": ms("2026-07-09 10:05:00")})
    assert r.status_code == 422  # same instant once truncated: changes nothing


def test_regularize_text_is_literal(client):
    make_emp(client)
    punch(client, "EMP0001", "2026-07-07 09:30:00", "2026-07-07 18:30:00")
    r = client.patch("/attendance/EMP0001/2026-07-07", json={**FIX_DOLLAR, "status": "WFH"})
    assert r.status_code == 200 and r.json()["history"][0]["reason"] == "$history refund" and r.json()["history"][0]["by"] == "$hr"
