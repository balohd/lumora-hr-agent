"""Access layer for synthetic structured HR data (JSON files in mock_data/).

Mock write operations (tickets) go to data/runtime_tickets.json so the committed
seed data is never modified.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOCK = Path(os.environ.get("MOCK_DATA_DIR", ROOT / "mock_data"))
RUNTIME_TICKETS = Path(os.environ.get("RUNTIME_TICKETS_FILE", ROOT / "data" / "runtime_tickets.json"))
_lock = threading.Lock()


def today() -> date:
    """Reference 'today'. HR_TODAY pins it for reproducible demos/evaluation."""
    v = os.environ.get("HR_TODAY")
    return date.fromisoformat(v) if v else date.today()


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    return json.loads((MOCK / name).read_text(encoding="utf-8"))


def employees() -> dict[str, dict]:
    return {e["employee_id"]: e for e in _load("employees.json")["employees"]}


def pto() -> dict[str, dict]:
    return {b["employee_id"]: b for b in _load("pto_balances.json")["balances"]}


def benefits() -> dict:
    return _load("benefits.json")


def offices() -> dict[str, dict]:
    return {o["office"]: o for o in _load("offices.json")["offices"]}


def rules() -> dict:
    return _load("policy_rules.json")


def holidays(calendar: str) -> set[date]:
    return {date.fromisoformat(d) for d in _load("holidays_2026.json").get(calendar, [])}


def normalize_employee_id(raw: str | None) -> str | None:
    if not raw:
        return None
    s = str(raw).strip().upper()
    if s.isdigit():
        s = "E" + s
    return s


def holiday_calendar_for(emp: dict) -> str:
    return {"US": "US", "CA": "CA", "UK": "UK"}.get(emp.get("country", "US"), "US")


def tenure_days(emp: dict, ref: date | None = None) -> int:
    return ((ref or today()) - date.fromisoformat(emp["hire_date"])).days


def business_days_between(start: date, end: date, cal: str) -> int:
    """Number of business days strictly after `start` up to and including `end`."""
    hol = holidays(cal)
    n, d = 0, start
    while d < end:
        d += timedelta(days=1)
        if d.weekday() < 5 and d not in hol:
            n += 1
    return n


def workdays_in_range(start: date, end: date, cal: str) -> list[date]:
    hol = holidays(cal)
    out, d = [], start
    while d <= end:
        if d.weekday() < 5 and d not in hol:
            out.append(d)
        d += timedelta(days=1)
    return out


def add_workdays(start: date, n: int, cal: str) -> date:
    """Return the date of the n-th workday counting `start` (if a workday) as day 1."""
    hol = holidays(cal)
    d, count = start, 0
    while True:
        if d.weekday() < 5 and d not in hol:
            count += 1
            if count >= n:
                return d
        d += timedelta(days=1)


# ----------------------------------------------------------------------------- tickets
def _read_runtime() -> list[dict]:
    if RUNTIME_TICKETS.exists():
        return json.loads(RUNTIME_TICKETS.read_text(encoding="utf-8")).get("tickets", [])
    return []


def list_tickets(employee_id: str | None = None) -> list[dict]:
    seed = _load("tickets.json")["tickets"]
    allt = seed + _read_runtime()
    return [t for t in allt if employee_id is None or t["employee_id"] == employee_id]


def append_ticket(ticket: dict) -> dict:
    with _lock:
        existing = _read_runtime()
        n = len(existing) + 1
        ticket["ticket_id"] = f"MOCK-HR-{today().year}-{n:04d}"
        ticket["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        existing.append(ticket)
        RUNTIME_TICKETS.parent.mkdir(parents=True, exist_ok=True)
        RUNTIME_TICKETS.write_text(json.dumps({"tickets": existing}, indent=2), encoding="utf-8")
    return ticket
