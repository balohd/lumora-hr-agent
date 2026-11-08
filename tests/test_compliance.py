"""Unit tests for the encoded policy rules (used by the check_policy_compliance MCP tool)."""
from mcp_server import compliance, data


def emp(eid):
    return data.employees()[eid]


def statuses(res):
    return {c["check"]: c["status"] for c in res["checks"]}


def test_six_weeks_abroad_exceeds_allowance():
    res = compliance.run(emp("E1001"), "temporary_remote_work", {"destination_country": "Portugal", "duration_weeks": 6})
    st = statuses(res)
    assert res["verdict"] == "not_compliant"
    assert st["duration"] == "fail" and st["destination"] == "pass" and st["tenure"] == "pass"
    assert res["alternatives"]


def test_two_weeks_abroad_allowed_with_approvals():
    res = compliance.run(emp("E1001"), "temporary_remote_work", {"destination_country": "Spain", "workdays": 10, "start_date": "2027-01-11"})
    assert res["verdict"] == "allowed_with_approvals"


def test_pip_new_hire_and_non_remote_roles_blocked():
    assert statuses(compliance.run(emp("E1008"), "temporary_remote_work", {"destination_country": "France", "workdays": 5}))["performance"] == "fail"
    assert statuses(compliance.run(emp("E1004"), "temporary_remote_work", {"destination_country": "Italy", "workdays": 5}))["tenure"] == "fail"
    assert statuses(compliance.run(emp("E1009"), "temporary_remote_work", {"destination_country": "Mexico", "workdays": 5}))["role_eligibility"] == "fail"


def test_sanctioned_destination_fails():
    assert statuses(compliance.run(emp("E1001"), "temporary_remote_work", {"destination_country": "Iran", "workdays": 5}))["destination"] == "fail"


def test_pto_short_notice_and_balance():
    res = compliance.run(emp("E1002"), "pto_request", {"start_date": "2026-11-02", "num_days": 3})
    st = statuses(res)
    assert st["balance"] == "pass" and st["advance_notice"] == "warning"
    assert res["request"]["end_date"] == "2026-11-04"


def test_pto_insufficient_balance_and_holidays_skipped():
    res = compliance.run(emp("E1008"), "pto_request", {"start_date": "2026-11-23", "end_date": "2026-11-27"})
    assert "2026-11-26" in res["request"]["company_holidays_in_range"]
    assert res["request"]["workdays_requested"] == 3  # Thanksgiving + day after are holidays
    assert statuses(res)["balance"] in ("needs_approval", "fail")


def test_support_blackout_and_contractor():
    res = compliance.run(emp("E1002"), "pto_request", {"start_date": "2026-12-21", "num_days": 2})
    assert statuses(res)["blackout"] == "needs_approval"
    assert compliance.run(emp("E1010"), "pto_request", {"start_date": "2026-12-01"})["verdict"] == "not_compliant"


def test_expenses():
    assert compliance.run(emp("E1002"), "expense", {"description": "new laptop"})["verdict"] == "not_compliant"
    chair_onsite = compliance.run(emp("E1002"), "expense", {"description": "office chair"})
    assert chair_onsite["verdict"] == "not_compliant"
    chair_hybrid = compliance.run(emp("E1001"), "expense", {"description": "ergonomic chair", "amount": 450})
    assert statuses(chair_hybrid)["stipend_eligibility"] == "pass" and statuses(chair_hybrid)["amount"] == "warning"


def test_benefits_eligibility():
    assert compliance.run(emp("E1003"), "benefits_eligibility", {"benefit": "medical"})["verdict"] == "not_compliant"
    assert compliance.run(emp("E1003"), "benefits_eligibility", {"benefit": "dental"})["verdict"] == "compliant"
    assert compliance.run(emp("E1011"), "benefits_eligibility", {"benefit": "401k"})["verdict"] == "not_compliant"


def test_missing_details_request_clarification():
    assert compliance.run(emp("E1001"), "temporary_remote_work", {})["verdict"] == "needs_clarification"
    assert compliance.run(emp("E1001"), "pto_request", {})["verdict"] == "needs_clarification"
