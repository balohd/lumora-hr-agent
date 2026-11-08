"""Intent classification, slot extraction and guardrail detectors.

Deterministic rules run on every request (they are cheap, testable and form the safety
layer). When an LLM is configured, `refine_with_llm` may fill missing slots, but the
guardrail decisions (sensitive / out-of-scope / privacy) always come from these rules.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
MONTHS["sept"] = 9
NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
             "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "half": 0.5}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

COUNTRIES = ["Portugal", "Spain", "France", "Germany", "Italy", "Ireland", "Netherlands", "Belgium", "Austria", "Greece",
             "Poland", "Sweden", "Denmark", "Finland", "Czech Republic", "Croatia", "Romania", "Hungary", "Malta", "Cyprus",
             "United Kingdom", "UK", "England", "Canada", "Mexico", "Japan", "Australia", "New Zealand", "Singapore",
             "India", "Brazil", "Argentina", "Colombia", "Thailand", "Vietnam", "Indonesia", "Philippines", "China",
             "South Korea", "Korea", "Turkey", "Egypt", "Morocco", "South Africa", "Nigeria", "Kenya", "Israel",
             "United Arab Emirates", "UAE", "Iran", "North Korea", "Cuba", "Syria", "Russia", "Belarus", "Costa Rica",
             "Peru", "Chile", "Iceland", "Norway", "Switzerland"]
US_STATES = {"alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO",
             "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
             "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
             "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
             "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
             "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC",
             "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
             "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
             "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
             "wisconsin": "WI", "wyoming": "WY"}

SENSITIVE_RE = re.compile(
    r"\b(harass\w*|sexual(ly)?|inappropriate(ly)? touch\w*|groped|discriminat\w*|racis\w*|sexis\w*|retaliat\w*|"
    r"bull(y|ied|ying)|hostile work|threat\w*|violen\w*|weapon|assault\w*|abus\w*|unsafe|whistleblow\w*|"
    r"kill (my|him|her|them)self|suicid\w*|self[- ]harm|hurt (my|him|her)self)\b", re.I)
SELF_HARM_RE = re.compile(r"\b(suicid\w*|kill (myself|me)|self[- ]harm|hurt myself|end my life)\b", re.I)

HR_TERMS = re.compile(
    r"\b(pto|vacation|time off|days? off|holiday|leave|sick|remote|hybrid|wfh|work from|abroad|office|expense|reimburs\w*|"
    r"benefit|medical|dental|vision|401|hsa|fsa|insurance|parental|maternity|paternity|bereavement|jury|onboard\w*|"
    r"offboard\w*|laptop|equipment|stipend|chair|desk|internet|security|vpn|password|policy|policies|harass\w*|conduct|"
    r"payroll|paycheck|pay|salary|bonus|overtime|manager|hr|case|ticket|travel|hotel|meal|per diem|mileage|"
    r"gift|conflict of interest|performance|review|pip|promotion|relocat\w*|visa|tax|enrollment|eap|wellness|"
    r"learning|training|i-9|badge|resign\w*|quit|termination|accommodation|disability|fmla|carryover|accrual|"
    r"floating|anchor day|data|ai tool|phone|monitor|contractor|intern|employee|lumora|work|job|role)\b|\btake\b[^.?!]{0,40}\boff\b", re.I)
OFF_TOPIC_RE = re.compile(
    r"\b(weather|recipe|capital of|stock price|bitcoin|crypto price|football|soccer|nba|movie|song|poem|joke|"
    r"write (me )?(a |some )?(python|code|essay|story)|translate|horoscope|lottery|celebrity)\b", re.I)
OTHER_PERSON_RE = re.compile(r"\b(E\d{4})\b", re.I)
PERSONAL_RE = re.compile(r"\b(i|i'm|im|my|me|mine|can i|do i|am i|should i)\b", re.I)


@dataclass
class Parsed:
    intent: str
    confidence: float
    slots: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)
    sub_questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"intent": self.intent, "confidence": self.confidence, "slots": self.slots, "flags": self.flags,
                "sub_questions": self.sub_questions}


# ----------------------------------------------------------------------------- slot extraction
def _num(tok: str) -> float | None:
    tok = tok.lower()
    if tok in NUM_WORDS:
        return NUM_WORDS[tok]
    try:
        return float(tok)
    except ValueError:
        return None


def extract_duration(text: str) -> dict:
    t = text.lower()
    m = re.search(r"\b(\d+(?:\.\d+)?|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|half)\s*(?:-|\s)?"
                  r"(work\s?days?|business days?|days?|weeks?|months?)\b", t)
    if not m:
        return {}
    n, unit = _num(m.group(1)), m.group(2)
    if n is None:
        return {}
    if unit.startswith("week"):
        return {"duration_weeks": n, "workdays": int(round(n * 5))}
    if unit.startswith("month"):
        return {"duration_weeks": round(n * 4.33, 1), "workdays": int(round(n * 21.7))}
    return {"num_days": n, "workdays": int(round(n))}


def _mk_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def extract_dates(text: str, today: date) -> dict:
    t = text.lower()
    out: dict = {}
    iso = re.findall(r"\b(20\d\d-\d\d-\d\d)\b", t)
    if iso:
        out["start_date"] = iso[0]
        if len(iso) > 1:
            out["end_date"] = iso[1]
        return out
    # "november 2-4", "nov 2 to nov 4", "dec 14 - dec 18", "november 2"
    mon = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    m = re.search(mon + r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s*(?:-|–|to|through|until)\s*(?:" + mon + r"\.?\s+)?(\d{1,2})(?:st|nd|rd|th)?)?", t)
    if m:
        m1 = MONTHS[m.group(1)[:3]] if m.group(1)[:4] != "sept" else 9
        d1 = int(m.group(2))
        year = today.year if (m1, d1) >= (today.month, today.day) else today.year + 1
        s = _mk_date(year, m1, d1)
        if s:
            out["start_date"] = s.isoformat()
            if m.group(4):
                m2 = MONTHS[m.group(3)[:3]] if m.group(3) else m1
                e = _mk_date(year if m2 >= m1 else year + 1, m2, int(m.group(4)))
                if e:
                    out["end_date"] = e.isoformat()
        return out
    if "next week" in t:
        nm = today + timedelta(days=(7 - today.weekday()) or 7)
        out["start_date"] = nm.isoformat()
        out["relative"] = "next week"
    elif "tomorrow" in t:
        out["start_date"] = (today + timedelta(days=1)).isoformat()
    elif "next month" in t:
        nmonth = date(today.year + (today.month == 12), today.month % 12 + 1, 1)
        while nmonth.weekday() >= 5:
            nmonth += timedelta(days=1)
        out["start_date"] = nmonth.isoformat()
        out["relative"] = "next month"
    else:
        for i, wd in enumerate(WEEKDAYS[:5]):
            if re.search(rf"\b(next |this |on )?{wd}\b", t):
                delta = (i - today.weekday()) % 7 or 7
                out["start_date"] = (today + timedelta(days=delta)).isoformat()
                break
    return out


def extract_location(text: str) -> dict:
    out = {}
    for c in sorted(COUNTRIES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(c)}\b", text, re.I if len(c) > 3 else 0):
            out["destination_country"] = c
            break
    for name, code in US_STATES.items():
        if re.search(rf"\b{name}\b", text, re.I) and not (name == "washington" and "washington dc" in text.lower()):
            if name == "georgia" and out.get("destination_country"):
                continue
            out["destination_state"] = code
            out.setdefault("destination_country", "United States")
            break
    if "destination_country" not in out and re.search(r"\b(another|different|other) (us )?state\b", text, re.I):
        out["destination_country"] = "United States"
        out["destination_state_unknown"] = True
    return out


def extract_amount(text: str) -> float | None:
    m = re.search(r"(?:\$|usd\s?)(\d[\d,]*(?:\.\d\d)?)|\b(\d[\d,]*(?:\.\d\d)?)\s?(?:dollars|usd)\b", text, re.I)
    if not m:
        return None
    return float((m.group(1) or m.group(2)).replace(",", ""))


def extract_benefit(text: str) -> str | None:
    t = text.lower()
    for k, pats in [("parental_leave", ["parental", "maternity", "paternity"]), ("hsa", ["hsa", "health savings"]),
                    ("401k", ["401", "retirement", "pension"]), ("dental", ["dental"]), ("vision", ["vision", "eye"]),
                    ("medical", ["medical", "health insurance", "health plan"]), ("life", ["life insurance"]),
                    ("learning_stipend", ["learning", "education stipend", "training budget"]),
                    ("wellness_stipend", ["wellness", "gym"]), ("eap", ["eap", "counsel"]),
                    ("home_office_stipend", ["home office stipend"])]:
        if any(p in t for p in pats):
            return k
    return None


# ----------------------------------------------------------------------------- classification
PRONOUN_RE = re.compile(r"\b(it|they|them|those|these|that|this|one)\b", re.I)


def split_sub_questions(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\?\s+|\?$|;\s+|\.\s+(?=[A-Z])|\balso\b", text) if p and len(p.strip()) > 12]
    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r",?\s+and (?=(?:what|how|when|which|is|are|can|do|does|will|should)\b)", text, flags=re.I) if len(p.strip()) > 12]
    if len(parts) <= 1:
        return []
    # a follow-up that depends on a pronoun ("when must I use them") inherits the first part's topic
    head = re.sub(r"^(how (many|much)|what|when|which|can|do|does|is|are)\b\s*", "", parts[0], flags=re.I)
    fixed = [parts[0]] + [p if not PRONOUN_RE.search(p) or len(p.split()) > 9 else f"{p} ({head})" for p in parts[1:]]
    return [text] + fixed[:3]


def classify(text: str, today: date) -> Parsed:
    t = text.lower().strip()
    flags: dict = {}
    slots: dict = {}

    others = {m.upper() for m in OTHER_PERSON_RE.findall(text)}
    if others:
        flags["mentioned_employee_ids"] = sorted(others)
    personal = bool(PERSONAL_RE.search(t))
    flags["personal"] = personal

    if SENSITIVE_RE.search(t):
        flags["sensitive"] = True
        flags["self_harm"] = bool(SELF_HARM_RE.search(t))
        # policy questions *about* harassment rules (no incident) are still answered via RAG, but escalation
        # applies when the user reports something happening to them or someone else
        reporting = re.search(r"\b(my manager|my boss|coworker|co-worker|colleague|someone|he|she|they|is being|was|keeps|kept|has been|"
                              r"happened|i am being|i'm being|i was|report)\b", t)
        if reporting or flags["self_harm"]:
            return Parsed("hr_case_sensitive", 0.95, slots, flags)

    if OFF_TOPIC_RE.search(t) and not re.search(r"\b(pto|leave|policy|benefit|expense|remote|holiday)\b", t):
        return Parsed("out_of_scope", 0.9, slots, flags)
    if not HR_TERMS.search(t):
        return Parsed("out_of_scope", 0.7, slots, flags)

    if re.search(r"\b(onboard\w*|first day|new hire|start date|starting on|checklist)\b", t) and \
            re.search(r"\b(checklist|onboard\w*|first (day|week)|what do i need to do|to-?do)\b", t):
        return Parsed("onboarding", 0.85, slots, flags)

    loc = extract_location(text)
    remote_words = re.search(r"\b(work(ing)? (remotely |remote )?from|remote(ly)?|wfh|work from abroad|abroad|another country|"
                             r"another state|other state|overseas|digital nomad|relocat\w*)\b", t)
    requesty = re.search(r"\b(can i|could i|may i|am i allowed|i want|i'd like|i would like|i'm planning|i am planning|"
                         r"i plan|i'm thinking|thinking of|is it ok|would it be ok|i need to)\b", t)
    if remote_words and (loc or "abroad" in t or "another" in t or "remote" in t) and (requesty or (loc and extract_duration(text))):
        if loc or re.search(r"\b(abroad|another country|another state|other state|overseas)\b", t):
            slots.update(loc)
            slots.update({k: v for k, v in extract_duration(text).items() if k in ("duration_weeks", "workdays")})
            slots.update({k: v for k, v in extract_dates(text, today).items() if k == "start_date"})
            return Parsed("remote_work", 0.9, slots, flags)

    if re.search(r"\b(pto|vacation|time off) balance\b|how (many|much) (days of )?(pto|vacation)( days)? (do i have|have i got|is left|do i have left|remaining)|(pto|vacation) (days )?(do i have )?left\b", t):
        return Parsed("pto_request", 0.8, {"balance_only": True}, flags)

    pto_words = re.search(r"\b(pto|vacation|time off|days? off|day off|take (a |\w+ )?(days?|week)|holiday leave|"
                          r"long weekend|annual leave)\b|\btake\b[^.?!]{0,40}\boff\b", t)
    policy_style = re.search(r"\b(carry ?over|carried over|accru\w*|payout|paid out|cap|maximum|policy|how does|explain|"
                             r"rollover|roll over|forfeit\w*|notice (is )?required|how much notice)\b", t)
    if pto_words and personal and not policy_style:
        slots.update(extract_dates(text, today))
        dur = extract_duration(text)
        if "num_days" in dur:
            slots["num_days"] = dur["num_days"]
        elif "duration_weeks" in dur:
            slots["num_days"] = dur["workdays"]
        if slots.get("relative") == "next week" and "num_days" not in slots:
            slots["num_days"] = 5
        return Parsed("pto_request", 0.85, slots, flags)
    expense_words = re.search(r"\b(reimburs\w*|expense\w*|expens|pay (me )?back|claim|stipend|can i buy|purchase|"
                              r"company pay for|get paid back)\b", t)
    if expense_words and (personal and (requesty or re.search(r"\b(i bought|i purchased|i paid|i spent|my)\b", t))):
        from mcp_server.compliance import classify_expense  # shared keyword map

        cat = classify_expense(t)
        if cat:
            slots["expense_description"] = text
            slots["expense_category"] = cat
            amt = extract_amount(text)
            if amt is not None:
                slots["amount"] = amt
            city = re.search(r"\bin (new york|san francisco|boston|seattle|washington dc|london|toronto|zurich|tokyo)\b", t)
            if city:
                slots["city"] = city.group(1).title()
            return Parsed("expense", 0.85, slots, flags)

    benefit = extract_benefit(t)
    if benefit and (personal or "eligible" in t) and re.search(r"\b(eligible|eligibility|qualify|enrolled|my|do i get|can i get|am i|coverage)\b", t):
        slots["benefit"] = benefit
        return Parsed("benefits", 0.8, slots, flags)

    subs = split_sub_questions(text)
    return Parsed("policy_qa", 0.7, slots, flags, subs)


def is_vague(text: str) -> bool:
    t = text.lower().strip()
    return len(t.split()) <= 3 and not re.search(r"\b(pto|holiday|401k|hsa|vpn|fmla|bereavement|jury)\b", t)
