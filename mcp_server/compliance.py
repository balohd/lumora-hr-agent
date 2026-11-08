"""Deterministic policy compliance checks over synthetic data.

Every check returns {check, status, detail, ref}. status is one of:
  pass | fail | needs_approval | warning | info
The overall verdict is derived from the individual checks. These checks encode the
numeric rules in mock_data/policy_rules.json, which cite the policy corpus.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

from . import data

EU_ALIASES = {"usa": "United States", "us": "United States", "u.s.": "United States", "united states of america": "United States",
              "uk": "United Kingdom", "england": "United Kingdom", "britain": "United Kingdom", "great britain": "United Kingdom"}
ENTITY_COUNTRY = {"US": "United States", "CA": "Canada", "UK": "United Kingdom"}
SANCTIONED = {"iran", "north korea", "cuba", "syria", "russia", "belarus", "crimea"}


def _check(name: str, status: str, detail: str, ref: str | list[str] | None = None) -> dict:
    return {"check": name, "status": status, "detail": detail, "ref": ref}


def _verdict(checks: list[dict]) -> str:
    st = {c["status"] for c in checks}
    if "fail" in st:
        return "not_compliant"
    if "needs_approval" in st or "warning" in st:
        return "allowed_with_approvals"
    return "compliant"


def _parse_date(v) -> date | None:
    if not v:
        return None
    if isinstance(v, date):
        return v
    return date.fromisoformat(str(v)[:10])


# ----------------------------------------------------------------------------- PTO
def check_pto(emp: dict, details: dict) -> dict:
    r = data.rules()["pto"]
    cal = data.holiday_calendar_for(emp)
    bal = data.pto().get(emp["employee_id"])
    today = _parse_date(details.get("request_date")) or data.today()
    start = _parse_date(details.get("start_date"))
    end = _parse_date(details.get("end_date"))
    num = details.get("num_days")
    checks: list[dict] = []

    if not start:
        return {"request_type": "pto_request", "verdict": "needs_clarification",
                "missing": ["start_date"], "checks": [], "message": "A start date is required to check a PTO request."}
    if not end:
        n = int(math.ceil(float(num or 1)))
        end = data.add_workdays(start, n, cal)
    workdays = data.workdays_in_range(start, end, cal)
    days_requested = float(num) if num and len(workdays) == int(math.ceil(float(num))) else float(len(workdays))
    skipped_holidays = sorted(d.isoformat() for d in data.holidays(cal) if start <= d <= end)

    if emp["employment_type"] == "contractor" or (bal and not bal.get("eligible", True)):
        checks.append(_check("eligibility", "fail", "Independent contractors are not eligible for Lumora PTO.", "POL-002 §2"))
        return {"request_type": "pto_request", "verdict": "not_compliant", "checks": checks,
                "request": {"start_date": start.isoformat(), "end_date": end.isoformat()}}
    checks.append(_check("eligibility", "pass", f"{emp['employment_type'].replace('_', '-')} employees accrue PTO.", "POL-002 §2"))

    balance = float(bal["current_balance"]) if bal else 0.0
    remaining = round(balance - days_requested, 2)
    if remaining >= 0:
        checks.append(_check("balance", "pass", f"Balance {balance} days covers {days_requested} requested; {remaining} days would remain.", "POL-002 §3"))
    else:
        tenure = data.tenure_days(emp, today)
        neg_ok = (-remaining <= r["negative_balance_limit_days"] * emp.get("fte", 1.0)) and tenure >= 90 and emp["employment_type"] != "intern"
        if neg_ok:
            checks.append(_check("balance", "needs_approval", f"Balance {balance} days is short by {-remaining}; a negative balance up to 3 days is allowed with manager approval.", r["negative_balance_ref"]))
        else:
            checks.append(_check("balance", "fail", f"Balance {balance} days is insufficient for {days_requested} days and exceeds the negative-balance allowance.", r["negative_balance_ref"]))

    consecutive = len(workdays)
    tier = next(t for t in r["notice_business_days"] if consecutive <= t["max_days"])
    notice_given = data.business_days_between(today, start, cal)
    if start <= today:
        checks.append(_check("advance_notice", "warning", "Start date is today or in the past; treat as a short-notice request.", tier["ref"]))
    elif notice_given >= tier["notice"]:
        checks.append(_check("advance_notice", "pass", f"{notice_given} business days' notice meets the {tier['notice']}-business-day requirement for {consecutive} consecutive workday(s).", tier["ref"]))
    else:
        checks.append(_check("advance_notice", "warning", f"Only {notice_given} business days' notice; {tier['notice']} required for {consecutive} consecutive workday(s). This is a short-notice request: manager discretion, include a brief note explaining the circumstances.", tier["ref"]))

    checks.append(_check("manager_approval", "needs_approval", "All PTO must be requested in PeopleHub and approved by the direct manager before it is taken.", r["manager_approval_ref"]))
    if consecutive > r["extended_pto_threshold_days"]:
        checks.append(_check("extended_pto", "needs_approval", "More than 10 consecutive workdays requires department head approval and People Team notification.", r["extended_pto_ref"]))

    dept = emp.get("department")
    for b in r["blackouts"]:
        if b["department"] != dept:
            continue
        if dept == "Customer Support":
            hit = any((d.month == 12 and d.day >= 15) or (d.month == 1 and d.day <= 5) for d in workdays)
        else:  # Finance: last 5 business days of quarter, first 5 of January
            hit = any((d.month in (3, 6, 9, 12) and d.day >= 24) or (d.month == 1 and d.day <= 9) for d in workdays)
        if hit:
            checks.append(_check("blackout", "needs_approval", f"Dates fall in the {dept} blackout period; requires approval from the {b['approver']}.", b["ref"]))
        else:
            checks.append(_check("blackout", "pass", f"Dates do not fall in the {dept} blackout period.", b["ref"]))

    return {
        "request_type": "pto_request", "verdict": _verdict(checks), "checks": checks,
        "request": {"start_date": start.isoformat(), "end_date": end.isoformat(), "workdays_requested": days_requested,
                    "workdays": [d.isoformat() for d in workdays], "company_holidays_in_range": skipped_holidays,
                    "request_date": today.isoformat(), "business_days_notice": notice_given},
        "required_approvals": ["Direct manager (PeopleHub)"] + (["Department head"] if consecutive > 10 else []),
    }


# ----------------------------------------------------------------------------- temporary remote work
def _norm_country(c: str | None) -> str | None:
    if not c:
        return None
    s = c.strip()
    return EU_ALIASES.get(s.lower(), s.title() if s.islower() else s)


def check_temporary_remote_work(emp: dict, details: dict) -> dict:
    r = data.rules()["temporary_remote_work"]
    today = _parse_date(details.get("request_date")) or data.today()
    dest_country = _norm_country(details.get("destination_country"))
    dest_state = details.get("destination_state")
    start = _parse_date(details.get("start_date"))
    workdays = details.get("workdays")
    if workdays is None and details.get("duration_weeks") is not None:
        workdays = int(round(float(details["duration_weeks"]) * 5))
    if workdays is None and details.get("duration_days") is not None:
        workdays = int(round(float(details["duration_days"]) * 5 / 7))
    missing = [m for m, v in (("destination_country", dest_country), ("duration (workdays or weeks)", workdays)) if v is None]
    if missing:
        return {"request_type": "temporary_remote_work", "verdict": "needs_clarification", "missing": missing, "checks": [],
                "message": "Destination and duration are required to check a temporary remote work request."}
    workdays = int(workdays)
    home_country = ENTITY_COUNTRY.get(emp["country"], emp["country"])
    international = dest_country != home_country
    checks: list[dict] = []

    if emp["employment_type"] in ("intern", "contractor"):
        checks.append(_check("employment_type", "fail", f"{emp['employment_type'].title()}s are not eligible for temporary remote work arrangements.", "POL-005 §4.2" if international else "POL-004 §3.2"))
    if not emp.get("role_remote_eligible", False) and emp["employment_type"] not in ("intern", "contractor"):
        checks.append(_check("role_eligibility", "fail", f"The role '{emp['title']}' requires physical presence or is not remote-eligible.", r["non_remote_roles_ref"]))
    elif emp["employment_type"] not in ("intern", "contractor"):
        checks.append(_check("role_eligibility", "pass", f"The role '{emp['title']}' is remote-eligible.", r["non_remote_roles_ref"]))

    if not international:
        same_state = dest_state and dest_state.upper() == str(emp.get("state_province", "")).upper()
        if same_state:
            checks.append(_check("location", "pass", "Same state/province: only manager awareness is required.", "POL-005 §3.1"))
        elif workdays <= r["domestic_other_state_max_workdays_per_year"]:
            checks.append(_check("duration", "pass", f"{workdays} workdays is within the 20-workday annual limit for another state.", r["domestic_ref"]))
            checks.append(_check("manager_approval", "needs_approval", "Manager approval in PeopleHub at least 10 business days in advance.", r["domestic_ref"]))
        elif workdays <= r["domestic_relocation_threshold_workdays"]:
            checks.append(_check("duration", "needs_approval", f"{workdays} workdays exceeds 20 workdays: requires People Team and Payroll Tax review (state withholding may apply).", r["domestic_ref"]))
        else:
            checks.append(_check("duration", "fail", f"{workdays} workdays exceeds 60: treated as a change of approved work location (relocation request).", "POL-004 §9"))
        approvals = ["Manager"] + (["People Team", "Payroll Tax"] if workdays > 20 else [])
        return {"request_type": "temporary_remote_work", "scope": "domestic", "verdict": _verdict(checks), "checks": checks,
                "request": {"destination_country": dest_country, "destination_state": dest_state, "workdays": workdays},
                "required_approvals": approvals}

    # ---- international
    sanctioned = dest_country.lower() in SANCTIONED
    approved = dest_country in r["approved_countries"]
    if sanctioned:
        checks.append(_check("destination", "fail", f"{dest_country} is subject to sanctions/high-risk designation; temporary remote work is never permitted.", r["approved_countries_ref"]))
    elif approved:
        checks.append(_check("destination", "pass", f"{dest_country} is on the approved country list.", r["approved_countries_ref"]))
    else:
        checks.append(_check("destination", "needs_approval", f"{dest_country} is not on the approved list; case-by-case Legal review required.", r["approved_countries_ref"]))

    tenure_months = data.tenure_days(emp, today) / 30.44
    if tenure_months >= r["international_min_tenure_months"]:
        checks.append(_check("tenure", "pass", f"Tenure of {tenure_months:.0f} months meets the 6-month minimum.", r["international_eligibility_ref"]))
    else:
        checks.append(_check("tenure", "fail", f"Tenure of {tenure_months:.1f} months is below the 6-month minimum.", r["international_eligibility_ref"]))

    rating = emp.get("performance_rating", "")
    if emp.get("on_pip"):
        checks.append(_check("performance", "fail", "Employee is on a performance improvement plan; not eligible until the PIP is completed.", ["POL-005 §4.2", "POL-013 §7"]))
    elif rating in r["eligible_ratings"]:
        checks.append(_check("performance", "pass", f"Most recent rating '{rating}' meets the requirement.", r["international_eligibility_ref"]))
    else:
        checks.append(_check("performance", "fail", f"Most recent rating '{rating}' does not meet 'Meets Expectations' or higher.", r["international_eligibility_ref"]))

    allowance = r["international_max_workdays_rolling_12m"]
    used = int(details.get("workdays_already_used_12m", 0))
    if workdays + used <= allowance:
        checks.append(_check("duration", "pass", f"{workdays} workdays is within the {allowance}-workday rolling 12-month allowance.", r["international_ref"]))
    else:
        checks.append(_check("duration", "fail", f"{workdays} workdays (plus {used} already used) exceeds the {allowance}-workday allowance in any rolling 12 months. Requires a Global Mobility exception (department head, VP of People and Legal; 4-6 week assessment; rarely granted).", [r["international_ref"], r["exception_ref"]]))

    if start:
        notice = (start - today).days
        if notice >= r["international_notice_calendar_days"]:
            checks.append(_check("advance_notice", "pass", f"{notice} calendar days' notice meets the 30-day requirement.", "POL-005 §4.3"))
        else:
            checks.append(_check("advance_notice", "warning", f"Only {notice} calendar days' notice; 30 required. May be declined for insufficient notice.", "POL-005 §4.3"))
    else:
        checks.append(_check("advance_notice", "info", "Submit the Work From Abroad form at least 30 calendar days before the first workday abroad.", "POL-005 §4.3"))

    level = emp.get("data_access_level")
    if level == "Restricted":
        checks.append(_check("data_access", "needs_approval", "Restricted data access: specific written Information Security approval required; travel loaner laptop may be required and Restricted access may be suspended.", [r["restricted_data_ref"], r["security_ref"]]))
    else:
        checks.append(_check("data_access", "info", f"'{level}' access: Lumora-managed device + VPN required; do not download Confidential data locally while abroad.", ["POL-005 §6.1", "POL-005 §6.2", "POL-007 §5"]))

    checks.append(_check("approvals", "needs_approval", "Approvals required in order: Manager, People Team (Global Mobility), Information Security.", r["approvals_ref"]))
    alternatives = []
    if workdays + used > allowance:
        alternatives.append(f"Work up to {allowance - used} workdays abroad within the allowance and take PTO for the remaining {workdays + used - allowance} workdays so no work is performed beyond the allowance (POL-005 §7).")
        alternatives.append("Request a Global Mobility exception with a written business justification (POL-005 §7).")
    approvals = list(r["approvals_international"]) + (["Department head", "VP of People", "Legal (exception)"] if workdays + used > allowance else [])
    return {"request_type": "temporary_remote_work", "scope": "international", "verdict": _verdict(checks), "checks": checks,
            "request": {"destination_country": dest_country, "workdays": workdays, "start_date": start.isoformat() if start else None,
                        "allowance_workdays": allowance, "workdays_already_used_12m": used},
            "required_approvals": approvals, "alternatives": alternatives,
            "expenses_note": "Travel and accommodation for temporary remote work are not reimbursed (POL-005 §9)."}


# ----------------------------------------------------------------------------- expenses
EXPENSE_KEYWORDS = {
    "laptop": ["laptop", "macbook", "notebook computer", "computer", "tablet", "ipad"],
    "phone": ["phone", "iphone", "mobile device"],
    "monitor": ["monitor", "display", "screen"],
    "office_chair": ["chair"],
    "desk": ["desk", "standing desk", "sit-stand"],
    "internet": ["internet", "wifi", "broadband"],
    "hotel": ["hotel", "lodging", "accommodation"],
    "meal_travel_international": ["international meal", "meal abroad"],
    "meal_travel_domestic": ["meal", "dinner", "lunch", "breakfast", "food"],
    "client_entertainment": ["client dinner", "client entertainment", "customer dinner"],
    "training": ["course", "conference", "certification", "training", "book"],
    "gym": ["gym", "fitness"],
}


def classify_expense(text: str) -> str | None:
    t = text.lower()
    for cat in ("client_entertainment", "meal_travel_international"):
        if any(k in t for k in EXPENSE_KEYWORDS[cat]):
            return cat
    for cat, kws in EXPENSE_KEYWORDS.items():
        if any(k in t for k in kws):
            return cat
    return None


def check_expense(emp: dict, details: dict) -> dict:
    r = data.rules()["expenses"]
    raw_cat = str(details.get("category") or details.get("description") or "")
    cat = raw_cat if raw_cat in r["categories"] else classify_expense(raw_cat)
    amount = details.get("amount")
    amount = float(amount) if amount not in (None, "") else None
    checks: list[dict] = []
    if not cat:
        return {"request_type": "expense", "verdict": "needs_clarification", "missing": ["category"], "checks": [],
                "message": "Could not determine the expense category; please describe the item (e.g. chair, laptop, hotel, meal)."}
    rule = r["categories"][cat]
    arrangement = emp.get("work_arrangement")
    reimb = rule["reimbursable"]
    if reimb is False:
        checks.append(_check("reimbursable", "fail", rule["reason"], rule["refs"]))
    elif reimb == "stipend" and cat in ("office_chair", "desk"):
        cap = {"remote": 500, "hybrid": 250}.get(arrangement, 0)
        if emp["employment_type"] == "intern" or cap == 0:
            checks.append(_check("stipend_eligibility", "fail", f"{arrangement.title()} employees/interns are not eligible for the home office stipend.", "POL-011 §5.1"))
        else:
            checks.append(_check("stipend_eligibility", "pass", f"{arrangement.title()} employees are eligible for a one-time home office stipend of up to ${cap} (once per 3 years).", "POL-011 §5.1"))
            if amount is not None and amount > cap:
                checks.append(_check("amount", "warning", f"${amount:.2f} exceeds the ${cap} stipend; the excess is the employee's responsibility.", "POL-011 §5.4"))
            checks.append(_check("how_to_claim", "info", "Claim in ExpenseIQ under 'Home Office Stipend' within 30 days of purchase with receipts.", "POL-011 §5.4"))
    elif reimb == "stipend" and cat == "training":
        used = next((b.get("learning_stipend_used_ytd", 0) for b in data.benefits()["elections"] if b["employee_id"] == emp["employee_id"]), 0)
        remaining = rule["cap"] - used if emp["employment_type"] == "full_time" else 0
        st = "pass" if amount is None or amount <= remaining else "warning"
        checks.append(_check("learning_stipend", st, f"Learning stipend remaining this year: ${remaining} (annual ${rule['cap']}, full-time only).", rule["refs"]))
    elif reimb == "monthly_cap":
        cap = rule["caps"].get(arrangement, 0)
        st = "fail" if cap == 0 else ("warning" if amount and amount > cap else "pass")
        checks.append(_check("internet", st, f"{arrangement.title()} employees may claim up to ${cap}/month for home internet.", rule["refs"]))
    elif reimb == "cap":
        cap = rule["cap"]
        city = str(details.get("city", ""))
        if cat == "hotel" and any(c.lower() in city.lower() for c in rule["high_cost_cities"]):
            cap = rule["high_cost_cap"]
        st = "pass" if amount is None or amount <= cap else "needs_approval"
        checks.append(_check("limit", st, f"Limit ${cap}{' per person' if rule.get('per_person') else ''}; {rule['reason']}", rule["refs"]))
        if rule.get("requires_preapproval"):
            checks.append(_check("preapproval", "needs_approval", "Manager pre-approval required before the expense.", "POL-006 §3"))
    if amount is not None and amount > r["preapproval_threshold"] and reimb is not False:
        checks.append(_check("preapproval", "needs_approval", "Single expenses over $500 require manager pre-approval in ExpenseIQ.", "POL-006 §3"))
    if reimb is not False:
        checks.append(_check("receipts", "info", "Submit in ExpenseIQ within 30 days; itemized receipts required for $25 or more.", "POL-006 §4"))
    return {"request_type": "expense", "category": cat, "verdict": _verdict(checks), "checks": checks,
            "request": {"description": raw_cat, "amount": amount, "work_arrangement": arrangement}}


# ----------------------------------------------------------------------------- benefits
def check_benefit(emp: dict, details: dict) -> dict:
    benefit = str(details.get("benefit", "")).lower()
    et, fte = emp["employment_type"], float(emp.get("fte", 1.0))
    hours = fte * 40
    region = emp["country"]
    table = {
        "medical": (et == "full_time", "POL-008 §2"),
        "dental": (et == "full_time" or (et == "part_time" and hours >= 20), "POL-008 §2"),
        "vision": (et == "full_time" or (et == "part_time" and hours >= 20), "POL-008 §2"),
        "401k": (et in ("full_time", "part_time"), "POL-008 §7"),
        "hsa": (et == "full_time", "POL-008 §4.2"),
        "life": (et == "full_time", "POL-008 §8"),
        "learning_stipend": (et == "full_time", "POL-008 §10"),
        "wellness_stipend": (et == "full_time" or (et == "part_time" and hours >= 20), "POL-008 §9"),
        "eap": (et in ("full_time", "part_time", "intern"), "POL-008 §9"),
        "parental_leave": (et in ("full_time", "part_time") and data.tenure_days(emp) >= 90, "POL-009 §3.1"),
        "home_office_stipend": (et != "intern" and emp.get("work_arrangement") in ("remote", "hybrid"), "POL-011 §5.1"),
    }
    key = next((k for k in table if k.replace("_", " ") in benefit or k in benefit), None)
    if key is None:
        for alias, k in (("health", "medical"), ("retire", "401k"), ("pension", "401k"), ("maternity", "parental_leave"),
                         ("paternity", "parental_leave"), ("gym", "wellness_stipend"), ("counsel", "eap"), ("learning", "learning_stipend")):
            if alias in benefit:
                key = k
                break
    if key is None:
        return {"request_type": "benefits_eligibility", "verdict": "needs_clarification", "missing": ["benefit"], "checks": [],
                "message": f"Unknown benefit '{benefit}'. Known: {', '.join(table)}."}
    ok, ref = table[key]
    checks = [_check("eligibility", "pass" if ok else "fail",
                     f"{key.replace('_', ' ')}: {'eligible' if ok else 'not eligible'} as {et.replace('_', '-')} (FTE {fte}){' in ' + region if region != 'US' else ''}.", ref)]
    if region != "US" and key in ("medical", "dental", "vision", "401k", "hsa"):
        checks.append(_check("region", "info", "Canadian and UK employees have local plans (extended health, Group RRSP, UK private medical, workplace pension).", "POL-008 §13"))
    return {"request_type": "benefits_eligibility", "benefit": key, "verdict": _verdict(checks), "checks": checks}


HANDLERS = {"pto_request": check_pto, "temporary_remote_work": check_temporary_remote_work,
            "expense": check_expense, "benefits_eligibility": check_benefit}


def run(emp: dict, request_type: str, details: dict) -> dict:
    rt = request_type.strip().lower().replace(" ", "_").replace("-", "_")
    rt = {"pto": "pto_request", "remote_work": "temporary_remote_work", "work_from_abroad": "temporary_remote_work",
          "remote_work_location": "temporary_remote_work", "benefits": "benefits_eligibility", "benefit": "benefits_eligibility"}.get(rt, rt)
    if rt not in HANDLERS:
        return {"request_type": request_type, "verdict": "unsupported",
                "message": f"Unsupported request_type. Use one of: {', '.join(HANDLERS)}."}
    res = HANDLERS[rt](emp, details or {})
    res["employee_id"] = emp["employee_id"]
    res["evaluated_on"] = (_parse_date((details or {}).get("request_date")) or data.today()).isoformat()
    return res


def next_monday(ref: date) -> date:
    return ref + timedelta(days=(7 - ref.weekday()) or 7)
