"""
Employee Attendance & Analytics API

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB (a local .env is loaded for convenience; real environment variables win)

Conventions (see README.md):
  * Instants are epoch milliseconds in the API and timezone-aware UTC datetimes in Python / BSON dates in MongoDB.
    Incoming instants are truncated to whole seconds (R1) before they are stored or used in a calculation.
  * Calendar dates ("YYYY-MM-DD"), months ("YYYY-MM") and shift times ("HH:MM", IST) stay plain strings.
  * Derived fields (work_hours, late_minutes, overtime_minutes, half_day) are only ever written by this API;
    analytics read the stored values and never recompute them.
"""
import calendar
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Any, Literal, Optional

from bson import Decimal128, json_util
from bson.json_util import JSONOptions, JSONMode
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.exceptions import RequestValidationError
from pydantic import AfterValidator, BaseModel, Field, model_validator
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.errors import ConnectionFailure, DuplicateKeyError, PyMongoError

load_dotenv()  # override=False: a real environment variable always wins over .env

log = logging.getLogger("attendance")

# tz_aware=True: PyMongo returns UTC-aware datetimes instead of naive ones, so arithmetic with IST values is safe.
client = MongoClient(
    os.getenv("MONGO_URI", "mongodb://localhost:27017"),
    tz_aware=True,
    tzinfo=timezone.utc,
    serverSelectionTimeoutMS=5000,
)
db = client[os.getenv("MONGO_DB", "attendance_db")]

UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

MIN_EPOCH_MS = 100_000_000_000  # anything smaller looks like seconds
MAX_EPOCH_MS = 4_102_444_800_000  # 2100-01-01
GRACE_SECONDS = 10 * 60  # R2
MIN_OVERTIME_MINUTES = 30  # R3
HALF_DAY_BELOW_HOURS = 4.5  # R5
MAX_SHIFT_LENGTH = timedelta(hours=24)
MAX_TREND_DAYS = 92

STATUSES = ("PRESENT", "WFH", "ON_DUTY", "ABSENT", "LEAVE")
PRESENCE = ("PRESENT", "WFH", "ON_DUTY")  # R6
StatusT = Literal["PRESENT", "WFH", "ON_DUTY", "ABSENT", "LEAVE"]
PresenceStatusT = Literal["PRESENT", "WFH", "ON_DUTY"]

ATTENDANCE_SORT = [("date", DESCENDING), ("emp_code", ASCENDING)]
ATTENDANCE_PROJECTION = {"_id": 0}

# Every index the endpoints rely on, created idempotently at startup. See DECISIONS.md.
INDEXES = {
    "employees": [
        ([("emp_code", ASCENDING)], {"name": "emp_code_unique", "unique": True}),
        ([("department", ASCENDING), ("emp_code", ASCENDING)], {"name": "department_emp_code"}),
    ],
    "attendance_logs": [
        ([("emp_code", ASCENDING), ("date", ASCENDING)], {"name": "emp_code_date_unique", "unique": True}),
        ([("date", DESCENDING), ("emp_code", ASCENDING)], {"name": "date_desc_emp_code"}),
        ([("emp_code", ASCENDING), ("punch_in", DESCENDING)], {"name": "emp_code_punch_in_desc"}),
    ],
}


_indexes_ready = False


def ensure_indexes() -> None:
    """create_index is a no-op when the index already exists, so this is safe on every start. If MongoDB is not
    reachable yet, give up quickly (startup must not hang) and let /health retry once the database answers."""
    global _indexes_ready
    for collection, specs in INDEXES.items():
        for keys, options in specs:
            try:
                db[collection].create_index(keys, **options)
            except ConnectionFailure:
                log.warning("MongoDB not reachable; indexes will be created on the next /health")
                return
            except PyMongoError:  # e.g. existing data violates a unique index: keep serving, but say so loudly
                log.exception("could not create index %s on %s", options["name"], collection)
    _indexes_ready = True


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_indexes()
    yield


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
def validation_error(loc: tuple, msg: str, value: Any = None):
    """A 422 with FastAPI's default body ({"detail": [...]}), for rules that need the database to check."""
    return RequestValidationError([{"type": "value_error", "loc": loc, "msg": msg, "input": value}])


def require_params(**params: Any) -> None:
    missing = [
        {"type": "missing", "loc": ("query", name), "msg": "Field required", "input": None}
        for name, value in params.items()
        if value is None
    ]
    if missing:
        raise RequestValidationError(missing)


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #
def to_ms(dt: Optional[datetime]) -> Optional[int]:
    if dt is None:
        return None
    if dt.tzinfo is None:  # defensive: everything from the driver is aware, but never guess a local zone
        dt = dt.replace(tzinfo=UTC)
    return (dt - EPOCH) // timedelta(milliseconds=1)


def from_ms(ms: int) -> datetime:
    """Epoch ms -> aware UTC datetime, truncated to whole seconds (R1)."""
    return EPOCH + timedelta(seconds=ms // 1000)


def now_utc() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def truncate(dt: datetime) -> datetime:
    return dt.replace(microsecond=0)


def parse_hhmm(value: str) -> time:
    h, m = map(int, value.split(":"))
    return time(h, m)


def is_overnight(emp: dict) -> bool:
    return parse_hhmm(emp["shift_end"]) <= parse_hhmm(emp["shift_start"])


def attendance_date(punch_in: datetime, emp: dict) -> date:
    """R1: the IST calendar date of the punch-in, except that an overnight-shift punch-in before shift_end
    belongs to the shift that started the previous day."""
    local = punch_in.astimezone(IST)
    day = local.date()
    if is_overnight(emp) and local.time() < parse_hhmm(emp["shift_end"]):
        day -= timedelta(days=1)
    return day


def shift_window(day: date, emp: dict) -> tuple[datetime, datetime]:
    """Shift start and end instants for the shift that starts on `day` (end is next day for overnight shifts)."""
    start = datetime.combine(day, parse_hhmm(emp["shift_start"]), IST)
    end = datetime.combine(day, parse_hhmm(emp["shift_end"]), IST)
    if end <= start:
        end += timedelta(days=1)
    return start, end


def half_up(value: Any, places: int) -> float:
    return float(Decimal(str(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


# --------------------------------------------------------------------------- #
# Business rules R2 - R5 (the starter's helpers, fixed)
# --------------------------------------------------------------------------- #
def compute_late_minutes(punch_in: datetime, day: date, emp: dict) -> int:
    """R2: late only when strictly more than 10:00 after shift start; then whole minutes since shift start."""
    start, _ = shift_window(day, emp)
    seconds = int((truncate(punch_in) - start).total_seconds())
    return seconds // 60 if seconds > GRACE_SECONDS else 0


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    """R4: hours rounded to 2 decimals half-up, done in Decimal so 0.005 never rounds down."""
    seconds = int((truncate(punch_out) - truncate(punch_in)).total_seconds())
    return float((Decimal(seconds) / Decimal(3600)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def compute_overtime(punch_out: datetime, day: date, emp: dict) -> int:
    """R3: whole minutes after shift end, only when that is at least 30."""
    _, end = shift_window(day, emp)
    minutes = int((truncate(punch_out) - end).total_seconds()) // 60
    return minutes if minutes >= MIN_OVERTIME_MINUTES else 0


def derived_fields(status: str, punch_in: Optional[datetime], punch_out: Optional[datetime], day: date, emp: dict) -> dict:
    """The stored derived values for a record, per the table in DATA_MODEL.md."""
    fields = {"work_hours": None, "late_minutes": 0, "overtime_minutes": 0, "half_day": False}
    if status not in PRESENCE or punch_in is None:
        return fields
    fields["late_minutes"] = compute_late_minutes(punch_in, day, emp)
    if punch_out is not None:
        work_hours = compute_work_hours(punch_in, punch_out)
        fields["work_hours"] = work_hours
        fields["overtime_minutes"] = compute_overtime(punch_out, day, emp)
        fields["half_day"] = work_hours < HALF_DAY_BELOW_HOURS  # R5 uses the rounded value
    return fields


# --------------------------------------------------------------------------- #
# Validation types and request models
# --------------------------------------------------------------------------- #
def _check_date(value: str) -> str:
    # date.fromisoformat alone is too lenient on 3.11+ (it accepts "20260706"), so pin the shape first.
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise ValueError("must be a date in YYYY-MM-DD format")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid date: {exc}") from None
    return value


DateStr = Annotated[str, AfterValidator(_check_date)]
MonthStr = Annotated[str, Query(pattern=r"^[0-9]{4}-(0[1-9]|1[0-2])$")]
HHMM = Annotated[str, Field(pattern=r"^([01][0-9]|2[0-3]):[0-5][0-9]$")]
# strict: rejects floats, strings and booleans; the bounds reject seconds-looking and out-of-range values.
EpochMillis = Annotated[int, Field(strict=True, ge=MIN_EPOCH_MS, le=MAX_EPOCH_MS)]
Page = Annotated[int, Query(ge=1)]
PageSize = Annotated[int, Query(ge=1, le=100)]


class EmployeeIn(BaseModel):
    emp_code: Annotated[str, Field(pattern=r"^EMP[0-9]{4,6}$")]
    name: Annotated[str, Field(min_length=1, max_length=100)]
    email: Annotated[str, Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")]
    department: Annotated[str, Field(min_length=1, max_length=50)]
    shift_start: HHMM = "09:30"
    shift_end: HHMM = "18:30"
    joined_on: DateStr

    @model_validator(mode="after")
    def _shift_times_differ(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


# Fields typed `EpochMillis = None` may be omitted, but an explicit null is rejected (the contract makes them
# non-nullable integers); pydantic does not validate defaults, only values that were actually sent.
class PunchInIn(BaseModel):
    emp_code: str
    punched_at: EpochMillis = None
    status: PresenceStatusT = "PRESENT"


class PunchOutIn(BaseModel):
    emp_code: str
    punched_at: EpochMillis = None


class RegularizeIn(BaseModel):
    """Derived fields are not declared, so a client that sends them is ignored (pydantic drops extras)."""

    status: StatusT = None
    punch_in: EpochMillis = None
    punch_out: EpochMillis = None
    reason: Annotated[str, Field(min_length=5, max_length=200)]
    regularized_by: Annotated[str, Field(min_length=1, max_length=50)]


# --------------------------------------------------------------------------- #
# Serialisation (MongoDB document -> API shape)
# --------------------------------------------------------------------------- #
def _instants_to_ms(value: Any) -> Any:
    if isinstance(value, datetime):
        return to_ms(value)
    if isinstance(value, dict):
        return {k: _instants_to_ms(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_instants_to_ms(v) for v in value]
    return value


def employee_out(doc: dict) -> dict:
    return {
        "emp_code": doc["emp_code"],
        "name": doc["name"],
        "email": doc["email"],
        "department": doc["department"],
        "shift_start": doc["shift_start"],
        "shift_end": doc["shift_end"],
        "joined_on": doc["joined_on"],
        "created_at": to_ms(doc.get("created_at")),
    }


def record_out(doc: dict) -> dict:
    """Missing legacy fields are normalised: late/overtime 0, half_day false, history []."""
    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": to_ms(doc.get("punch_in")),
        "punch_out": to_ms(doc.get("punch_out")),
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes") or 0,
        "overtime_minutes": doc.get("overtime_minutes") or 0,
        "half_day": bool(doc.get("half_day") or False),
        "history": [
            {
                "at": to_ms(h.get("at")),
                "by": h.get("by"),
                "reason": h.get("reason"),
                "changes": _instants_to_ms(h.get("changes") or {}),
            }
            for h in doc.get("history") or []
        ],
    }


def get_employee_or_404(emp_code: str) -> dict:
    emp = db.employees.find_one({"emp_code": emp_code})
    if emp is None:
        raise HTTPException(404, f"employee {emp_code} not found")
    return emp


# --------------------------------------------------------------------------- #
# System and employees
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    try:
        client.admin.command("ping")
    except PyMongoError:
        raise HTTPException(503, "database unavailable")
    if not _indexes_ready:
        ensure_indexes()
    return {"status": "ok"}


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    doc = body.model_dump()
    created = datetime.now(UTC)
    doc["created_at"] = created.replace(microsecond=created.microsecond // 1000 * 1000)  # BSON keeps ms only
    try:
        db.employees.insert_one(doc)  # the unique index decides; no check-then-insert race
    except DuplicateKeyError:
        raise HTTPException(409, f"emp_code {body.emp_code} already exists")
    return employee_out(doc)


@app.get("/employees")
def list_employees(department: Optional[str] = None, page: Page = 1, page_size: PageSize = 20):
    query = {"department": department} if department is not None else {}
    total = db.employees.count_documents(query)
    cursor = (
        db.employees.find(query)
        .sort("emp_code", ASCENDING)
        .skip((page - 1) * page_size)
        .limit(page_size)
    )
    return {"items": [employee_out(d) for d in cursor], "total": total, "page": page, "page_size": page_size}


# --------------------------------------------------------------------------- #
# Attendance
# --------------------------------------------------------------------------- #
@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp = get_employee_or_404(body.emp_code)
    ts = from_ms(body.punched_at) if body.punched_at is not None else now_utc()
    day = attendance_date(ts, emp)
    doc = {
        "emp_code": body.emp_code,
        "date": day.isoformat(),
        "status": body.status,
        "punch_in": ts,
        "punch_out": None,
        **derived_fields(body.status, ts, None, day, emp),
        "history": [],
    }
    try:
        # The unique (emp_code, date) index makes this atomic: of N identical concurrent requests one inserts,
        # the others get DuplicateKeyError.
        db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, f"{body.emp_code} already has a record for {doc['date']}")
    return record_out(doc)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    emp = get_employee_or_404(body.emp_code)
    ts = from_ms(body.punched_at) if body.punched_at is not None else now_utc()
    record = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": ts}},
        sort=[("punch_in", DESCENDING)],
    )
    if record is None:
        raise HTTPException(404, f"no punch-in found for {body.emp_code} at or before that time")
    if record.get("punch_out") is not None:
        raise HTTPException(409, f"record for {record['date']} is already punched out")
    started = truncate(record["punch_in"])
    if ts <= started or ts - started > MAX_SHIFT_LENGTH:
        raise validation_error(("body", "punched_at"), "punch-out must be after punch-in and within 24 hours", body.punched_at)

    day = date.fromisoformat(record["date"])
    updates = {"punch_out": ts, **derived_fields(record["status"], started, ts, day, emp)}
    # Compare-and-set: only succeeds if nobody punched out or corrected the record since we read it.
    updated = db.attendance_logs.find_one_and_update(
        {"_id": record["_id"], "punch_out": None, "punch_in": record["punch_in"], "status": record["status"]},
        {"$set": updates},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(409, f"record for {record['date']} was punched out or changed concurrently")
    return record_out(updated)


def attendance_filter(emp_code: Optional[str], date_from: Optional[str], date_to: Optional[str], status: Optional[str]) -> dict:
    if date_from and date_to and date_from > date_to:
        raise validation_error(("query", "date_from"), "date_from must be on or before date_to", date_from)
    query: dict = {}
    if emp_code is not None:
        query["emp_code"] = emp_code
    if date_from or date_to:
        query["date"] = {}
        if date_from:
            query["date"]["$gte"] = date_from
        if date_to:
            query["date"]["$lte"] = date_to
    if status is not None:
        query["status"] = status
    return query


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[DateStr] = None,
    date_to: Optional[DateStr] = None,
    status: Optional[StatusT] = None,
    page: Page = 1,
    page_size: PageSize = 20,
):
    query = attendance_filter(emp_code, date_from, date_to, status)
    total = db.attendance_logs.count_documents(query)
    cursor = (
        db.attendance_logs.find(query, ATTENDANCE_PROJECTION)
        .sort(ATTENDANCE_SORT)
        .skip((page - 1) * page_size)
        .limit(page_size)
    )
    return {"items": [record_out(d) for d in cursor], "total": total, "page": page, "page_size": page_size}


@app.patch("/attendance/{emp_code}/{date}")
def regularize_attendance(emp_code: str, record_date: Annotated[DateStr, Path(alias="date")], body: RegularizeIn):
    emp = get_employee_or_404(emp_code)
    record = db.attendance_logs.find_one({"emp_code": emp_code, "date": record_date})
    if record is None:
        raise HTTPException(404, f"no attendance record for {emp_code} on {record_date}")

    day = date.fromisoformat(record_date)
    old = {
        "status": record["status"],
        "punch_in": record.get("punch_in"),
        "punch_out": record.get("punch_out"),
        "work_hours": record.get("work_hours"),
        "late_minutes": record.get("late_minutes") or 0,
        "overtime_minutes": record.get("overtime_minutes") or 0,
        "half_day": bool(record.get("half_day") or False),
    }
    new_status = body.status if body.status is not None else old["status"]
    sent_in = from_ms(body.punch_in) if body.punch_in is not None else None
    sent_out = from_ms(body.punch_out) if body.punch_out is not None else None

    if new_status not in PRESENCE:
        if sent_in is not None or sent_out is not None:
            raise validation_error(("body", "status"), f"status {new_status} clears punch times; do not send punch_in/punch_out")
        new_in = new_out = None
    else:
        new_in = sent_in if sent_in is not None else old["punch_in"]
        new_out = sent_out if sent_out is not None else old["punch_out"]
        if new_in is None:
            raise validation_error(("body", "punch_in"), f"status {new_status} requires a punch_in")
        new_in = truncate(new_in)
        new_out = truncate(new_out) if new_out is not None else None
        if sent_in is not None and attendance_date(new_in, emp) != day:
            raise validation_error(("body", "punch_in"), f"punch_in must stay on attendance date {record_date}", body.punch_in)
        if new_out is not None and (new_out <= new_in or new_out - new_in > MAX_SHIFT_LENGTH):
            raise validation_error(("body", "punch_out"), "punch_out must be after punch_in and within 24 hours", body.punch_out)

    new = {"status": new_status, "punch_in": new_in, "punch_out": new_out, **derived_fields(new_status, new_in, new_out, day, emp)}

    def same(field: str) -> bool:
        before, after = old[field], new[field]
        if isinstance(before, datetime) and isinstance(after, datetime):
            return truncate(before) == after
        return before == after

    if all(same(f) for f in ("status", "punch_in", "punch_out")):
        raise validation_error(("body",), "the request does not change anything")

    entry = {
        "at": datetime.now(UTC),
        "by": body.regularized_by,
        "reason": body.reason,
        "changes": {f: {"from": old[f], "to": new[f]} for f in old if not same(f)},
    }
    history_len = len(record.get("history") or [])
    # Optimistic concurrency: the write only applies if the record is exactly as we read it (same times, same
    # status, same number of history entries). A concurrent correction or punch-out makes this match nothing,
    # and the loser gets 409 instead of silently overwriting the winner's change or history entry.
    guard = {
        "_id": record["_id"],
        "status": record["status"],
        "punch_in": record.get("punch_in"),
        "punch_out": record.get("punch_out"),
    }
    if history_len:
        guard["history"] = {"$size": history_len}
    else:
        guard["$or"] = [{"history": None}, {"history": {"$size": 0}}]  # None matches a missing field or null
    # Pipeline update so a legacy record whose history is missing or null can still be appended to ($push would
    # fail on null). $literal keeps client text such as a reason starting with "$" from being read as a field path.
    append = {"$concatArrays": [{"$ifNull": ["$history", []]}, [{"$literal": entry}]]}
    updated = db.attendance_logs.find_one_and_update(
        guard,
        [{"$set": {**{k: {"$literal": v} for k, v in new.items()}, "history": append}}],
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(409, "the record was modified concurrently; reload and retry")
    return record_out(updated)


# --------------------------------------------------------------------------- #
# Aggregation building blocks
# --------------------------------------------------------------------------- #
def month_bounds(month: str) -> tuple[str, str]:
    year, mon = map(int, month.split("-"))
    return f"{month}-01", f"{month}-{calendar.monthrange(year, mon)[1]:02d}"


def count_weekdays(start: date, end: date) -> int:
    """Mon-Fri days in [start, end] (R7)."""
    if start > end:
        return 0
    weeks, extra = divmod((end - start).days + 1, 7)
    first = start.weekday()
    return weeks * 5 + sum(1 for i in range(extra) if (first + i) % 7 < 5)


def agg_round(expr: Any, places: int, to_double: bool = True) -> dict:
    """Half-up rounding inside a pipeline. MongoDB's $round is half-to-even, so compute floor(x * 10^n + 0.5) / 10^n
    in Decimal128 (exact for the values we round, all of which are >= 0). Null in, null out."""
    factor = 10**places
    rounded = {
        "$divide": [{"$floor": {"$add": [{"$multiply": [{"$toDecimal": expr}, factor]}, Decimal128("0.5")]}}, factor]
    }
    return {"$toDouble": rounded} if to_double else rounded


IS_PRESENT = {"$in": ["$status", list(PRESENCE)]}
IS_WEEKDAY = {"$lte": [{"$isoDayOfWeek": {"$dateFromString": {"dateString": "$date", "format": "%Y-%m-%d"}}}, 5]}
DAY_VALUE = {"$cond": [{"$eq": [{"$ifNull": ["$half_day", False]}, True]}, 0.5, 1]}  # R5
IS_LATE = {"$gt": [{"$ifNull": ["$late_minutes", 0]}, 0]}
HAS_WORK_HOURS = {"$and": [IS_PRESENT, {"$isNumber": "$work_hours"}]}

# Per-record totals over a set of attendance logs. Used by the monthly and the department summaries so the two
# can never disagree on what "present", "late" or "leave" means.
LOG_TOTALS = {
    "_id": None,
    "present_days": {"$sum": {"$cond": [{"$and": [IS_PRESENT, IS_WEEKDAY]}, DAY_VALUE, 0]}},  # R6, R7
    "leave_days": {"$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}},
    "on_duty_days": {"$sum": {"$cond": [{"$eq": ["$status", "ON_DUTY"]}, 1, 0]}},
    "late_count": {"$sum": {"$cond": [IS_LATE, 1, 0]}},
    "total_late_minutes": {"$sum": {"$cond": [IS_LATE, "$late_minutes", 0]}},
    "total_overtime_minutes": {"$sum": {"$ifNull": ["$overtime_minutes", 0]}},
    "work_hours_sum": {"$sum": {"$cond": [HAS_WORK_HOURS, {"$toDecimal": "$work_hours"}, 0]}},
    "work_hours_count": {"$sum": {"$cond": [HAS_WORK_HOURS, 1, 0]}},
}


def monthly_pipeline(emp_code: str, month: str) -> list:
    first, last = month_bounds(month)
    return [
        {"$match": {"emp_code": emp_code, "date": {"$gte": first, "$lte": last}}},
        {"$group": LOG_TOTALS},
    ]


def department_summary_pipeline(month: str, department: Optional[str]) -> list:
    first, last = month_bounds(month)
    match: dict = {"joined_on": {"$lte": last}}  # R9: joined by the end of the month, logs or not
    if department is not None:
        match["department"] = department
    return [
        {"$match": match},
        # Streams employees in department order straight off the department_emp_code index.
        {"$sort": {"department": 1, "emp_code": 1}},
        # Headcount comes from employees, not from logs, so people with zero logs are counted.
        {"$group": {"_id": "$department", "headcount": {"$sum": 1}, "emp_codes": {"$push": "$emp_code"}}},
        # One lookup per department: an array localField matches any of its values, so this is an indexed
        # {emp_code: {$in: emp_codes}, date: <month>} scan on emp_code_date_unique. The sub-pipeline sums every
        # log of the department, so averages are over records, not averages of per-employee averages.
        {
            "$lookup": {
                "from": "attendance_logs",
                "localField": "emp_codes",
                "foreignField": "emp_code",
                "pipeline": [{"$match": {"date": {"$gte": first, "$lte": last}}}, {"$group": LOG_TOTALS}],
                "as": "totals",
            }
        },
        {"$unwind": {"path": "$totals", "preserveNullAndEmptyArrays": True}},  # no logs at all -> keep the row
        {"$sort": {"_id": 1}},
        {
            "$project": {
                "_id": 0,
                "department": "$_id",
                "headcount": 1,
                "present_days": agg_round({"$ifNull": ["$totals.present_days", 0]}, 2),
                "avg_work_hours": {
                    "$cond": [
                        {"$gt": ["$totals.work_hours_count", 0]},
                        agg_round({"$divide": ["$totals.work_hours_sum", "$totals.work_hours_count"]}, 2),
                        None,
                    ]
                },
                "late_count": {"$ifNull": ["$totals.late_count", 0]},
                "total_late_minutes": {"$toInt": {"$ifNull": ["$totals.total_late_minutes", 0]}},
                "leave_count": {"$ifNull": ["$totals.leave_days", 0]},
                "on_duty_count": {"$ifNull": ["$totals.on_duty_days", 0]},
            }
        },
    ]


def leaderboard_pipeline(month: str, limit: int, department: Optional[str]) -> list:
    first, last = month_bounds(month)
    pipeline: list = [
        {"$match": {"date": {"$gte": first, "$lte": last}, "late_minutes": {"$gt": 0}}},
        {"$group": {"_id": "$emp_code", "total_late_minutes": {"$sum": "$late_minutes"}, "late_count": {"$sum": 1}}},
        # Inner join: logs whose employee does not exist disappear here.
        {"$lookup": {"from": "employees", "localField": "_id", "foreignField": "emp_code", "as": "emp"}},
        {"$unwind": "$emp"},
    ]
    if department is not None:
        pipeline.append({"$match": {"emp.department": department}})  # rank inside the department only
    pipeline += [
        # Standard competition ranking (1, 2, 2, 4); the limit is applied to the rank, never to the row count.
        {"$setWindowFields": {"sortBy": {"total_late_minutes": -1}, "output": {"rank": {"$rank": {}}}}},
        {"$match": {"rank": {"$lte": limit}}},
        {"$sort": {"total_late_minutes": -1, "_id": 1}},
        {
            "$project": {
                "_id": 0,
                "rank": 1,
                "emp_code": "$_id",
                "name": "$emp.name",
                "department": "$emp.department",
                "total_late_minutes": {"$toInt": "$total_late_minutes"},
                "late_count": 1,
            }
        },
    ]
    return pipeline


def trend_pipeline(department: str, start: date, end: date) -> list:
    first, last = start.isoformat(), end.isoformat()
    lower = datetime(start.year, start.month, start.day, tzinfo=UTC)
    upper = datetime(end.year, end.month, end.day, tzinfo=UTC) + timedelta(days=1)  # $densify upper bound is exclusive
    return [
        {"$match": {"department": department}},
        {
            "$lookup": {
                "from": "attendance_logs",
                "localField": "emp_code",
                "foreignField": "emp_code",
                "pipeline": [
                    {"$match": {"date": {"$gte": first, "$lte": last}}},
                    {"$project": {"_id": 0, "date": 1, "status": 1, "half_day": 1, "late_minutes": 1}},
                ],
                "as": "logs",
            }
        },
        # Turn every employee into "events" keyed by day: one join event (+1 headcount from the day they joined,
        # clamped to the range start) plus one event per log. Employees joining after the range contribute 0.
        {
            "$project": {
                "_id": 0,
                "events": {
                    "$concatArrays": [
                        [
                            {
                                "date": {"$cond": [{"$lte": ["$joined_on", last]}, {"$max": ["$joined_on", first]}, first]},
                                "joins": {"$cond": [{"$lte": ["$joined_on", last]}, 1, 0]},
                                "present": 0,
                                "late": 0,
                            }
                        ],
                        {
                            "$map": {
                                "input": "$logs",
                                "as": "l",
                                "in": {
                                    "date": "$$l.date",
                                    "joins": 0,
                                    "present": {
                                        "$cond": [
                                            {"$in": ["$$l.status", list(PRESENCE)]},
                                            {"$cond": [{"$eq": [{"$ifNull": ["$$l.half_day", False]}, True]}, 0.5, 1]},
                                            0,
                                        ]
                                    },
                                    "late": {"$cond": [{"$gt": [{"$ifNull": ["$$l.late_minutes", 0]}, 0]}, 1, 0]},
                                },
                            }
                        },
                    ]
                },
            }
        },
        {"$unwind": "$events"},
        {
            "$group": {
                "_id": "$events.date",
                "joins": {"$sum": "$events.joins"},
                "present": {"$sum": "$events.present"},
                "late": {"$sum": "$events.late"},
            }
        },
        {"$project": {"_id": 0, "day": {"$dateFromString": {"dateString": "$_id", "format": "%Y-%m-%d", "timezone": "UTC"}}, "joins": 1, "present": 1, "late": 1}},
        # Gap filling in the database: one document per calendar day in [from, to], weekends and empty days too.
        {"$densify": {"field": "day", "range": {"step": 1, "unit": "day", "bounds": [lower, upper]}}},
        {"$sort": {"day": 1}},
        # R9 headcount = running total of join events up to and including the day.
        {"$setWindowFields": {"sortBy": {"day": 1}, "output": {"headcount": {"$sum": "$joins", "window": {"documents": ["unbounded", "current"]}}}}},
        {
            "$addFields": {
                "is_working_day": {"$lte": [{"$isoDayOfWeek": "$day"}, 5]},
                "present_count": {"$ifNull": ["$present", 0]},
                "late_count": {"$ifNull": ["$late", 0]},
            }
        },
        {
            "$addFields": {
                "rate": {
                    "$cond": [
                        {"$and": ["$is_working_day", {"$gt": ["$headcount", 0]}]},
                        agg_round({"$divide": [{"$toDecimal": "$present_count"}, "$headcount"]}, 4, to_double=False),
                        None,
                    ]
                }
            }
        },
        # $avg skips nulls and returns null when the window has no number. The window only sees rows inside the
        # requested range, so the first rows naturally use a shorter window.
        {"$setWindowFields": {"sortBy": {"day": 1}, "output": {"moving_avg": {"$avg": "$rate", "window": {"documents": [-6, 0]}}}}},
        {
            "$project": {
                "_id": 0,
                "date": {"$dateToString": {"date": "$day", "format": "%Y-%m-%d"}},
                "is_working_day": 1,
                "headcount": 1,
                "present_count": 1,
                "late_count": 1,
                "attendance_rate": {"$toDouble": "$rate"},
                "moving_avg_7d": agg_round("$moving_avg", 4),
            }
        },
    ]


def trend_range(from_: str, to: str) -> tuple[date, date]:
    start, end = date.fromisoformat(from_), date.fromisoformat(to)
    if end < start:
        raise validation_error(("query", "to"), "to must be on or after from", to)
    if (end - start).days + 1 > MAX_TREND_DAYS:
        raise validation_error(("query", "to"), f"the range may span at most {MAX_TREND_DAYS} days", to)
    return start, end


# --------------------------------------------------------------------------- #
# Analytics
# --------------------------------------------------------------------------- #
@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: MonthStr):
    emp = get_employee_or_404(emp_code)
    first, last = month_bounds(month)
    totals = next(db.attendance_logs.aggregate(monthly_pipeline(emp_code, month)), {})

    # R7: Mon-Fri from the later of the month start and joined_on. Pure calendar arithmetic, no data needed.
    start = max(date.fromisoformat(first), date.fromisoformat(emp["joined_on"]))
    working_days = count_weekdays(start, date.fromisoformat(last))
    present_days = half_up(totals.get("present_days", 0), 2)
    attendance_pct = (
        float((Decimal(str(present_days)) * 100 / working_days).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        if working_days
        else None
    )
    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": present_days,
        "leave_days": totals.get("leave_days", 0),
        "late_count": totals.get("late_count", 0),
        "total_late_minutes": int(totals.get("total_late_minutes", 0)),
        "total_overtime_minutes": int(totals.get("total_overtime_minutes", 0)),
        "attendance_pct": attendance_pct,
    }


@app.get("/analytics/departments/summary")
def department_summary(month: MonthStr, department: Optional[str] = None):
    items = list(db.employees.aggregate(department_summary_pipeline(month, department)))
    return {"month": month, "items": items}


@app.get("/analytics/leaderboard/late")
def late_leaderboard(month: MonthStr, limit: Annotated[int, Query(ge=1, le=50)] = 10, department: Optional[str] = None):
    items = list(db.attendance_logs.aggregate(leaderboard_pipeline(month, limit, department)))
    return {"month": month, "items": items}


@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    from_: Annotated[DateStr, Query(alias="from")],
    to: DateStr,
):
    start, end = trend_range(from_, to)
    if db.employees.find_one({"department": department}, {"_id": 1}) is None:
        raise HTTPException(404, f"department {department} has no employees")
    items = list(db.employees.aggregate(trend_pipeline(department, start, end)))
    return {"department": department, "items": items}


# --------------------------------------------------------------------------- #
# Admin
# --------------------------------------------------------------------------- #
_EXPLAIN_JSON = JSONOptions(json_mode=JSONMode.RELAXED, tz_aware=True)


@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: Literal["attendance_list", "employee_monthly", "department_summary", "late_leaderboard", "department_trend"],
    emp_code: Optional[str] = None,
    month: Optional[MonthStr] = None,
    department: Optional[str] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    date_from: Optional[DateStr] = None,
    date_to: Optional[DateStr] = None,
    status: Optional[StatusT] = None,
    from_: Annotated[Optional[DateStr], Query(alias="from")] = None,
    to: Optional[DateStr] = None,
    page: Page = 1,
    page_size: PageSize = 20,
):
    """Explains the main operation of an endpoint, built by the same function the endpoint itself uses."""
    if endpoint == "attendance_list":
        collection = "attendance_logs"
        command = {
            "find": collection,
            "filter": attendance_filter(emp_code, date_from, date_to, status),
            "projection": ATTENDANCE_PROJECTION,
            "sort": dict(ATTENDANCE_SORT),
            "skip": (page - 1) * page_size,
            "limit": page_size,
        }
    else:
        if endpoint == "employee_monthly":
            require_params(emp_code=emp_code, month=month)
            collection, pipeline = "attendance_logs", monthly_pipeline(emp_code, month)
        elif endpoint == "department_summary":
            require_params(month=month)
            collection, pipeline = "employees", department_summary_pipeline(month, department)
        elif endpoint == "late_leaderboard":
            require_params(month=month)
            collection, pipeline = "attendance_logs", leaderboard_pipeline(month, limit, department)
        else:  # department_trend
            require_params(department=department, **{"from": from_, "to": to})
            collection, pipeline = "employees", trend_pipeline(department, *trend_range(from_, to))
        command = {"aggregate": collection, "pipeline": pipeline, "cursor": {}}

    result = db.command({"explain": command, "verbosity": "executionStats"})
    return {"endpoint": endpoint, "collection": collection, "explain": json.loads(json_util.dumps(result, json_options=_EXPLAIN_JSON))}
