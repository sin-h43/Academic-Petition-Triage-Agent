import re
from typing import List, Optional, Literal
from pydantic import BaseModel, Field

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"you\s+are\s+now\s+the\s+(examination\s+officer|dean|director)",
    r"system\s*prompt",
    r"override\s+regulation",
    r"force\s+route\s*=",
]

FORBIDDEN_DETERMINATION_PHRASES = [
    "i hereby grant",
    "your petition is approved",
    "your petition is officially rejected",
    "final decision: granted",
]

class PetitionRequest(BaseModel):
    student_id: str = Field(..., examples=["SYNTH-2026-001"])
    course_code: str = Field(..., examples=["CS3001"])
    ltp: str = Field(..., description="Lecture-Tutorial-Practical structure, e.g., 3-0-0")
    semester_type: Literal["regular", "summer"] = "regular"
    raw_absences: int
    medical_leave_days: int = 0
    medical_sanctioned: bool = False
    petition_text: str = Field(..., min_length=10, max_length=2000)

class CitationItem(BaseModel):
    clause_id: str
    page_number: int
    snippet: str
    verified: bool

class TriageRunStatus(BaseModel):
    run_id: str
    correlation_id: str
    status: Literal["queued", "running", "completed", "quarantined", "blocked", "escalated"]
    route: Optional[Literal["grant-under-provision", "refer", "reject", "escalate", "quarantine", "blocked"]] = None
    referred_to: List[str] = []
    citations: List[CitationItem] = []
    rationale: Optional[str] = None
    officer_notice: str = "ADVISORY ONLY: An examination officer always makes the final decision."
    errors: List[str] = []
    elapsed_ms: Optional[float] = None

def validate_and_guard(req: PetitionRequest) -> tuple[str, List[str]]:
    """
    Returns ('ok' | 'quarantine' | 'blocked', list_of_errors)
    Role 3 boundary validation + Role 4 injection guardrail.
    """
    errors = []

    # 1. Role 4: Prompt Injection Defence on free-text fields
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, req.petition_text, re.IGNORECASE):
            return "blocked", [f"Guardrail triggered: potential prompt injection matching '{pattern}'."]

    # 2. Role 3: Range and Cross-Field Quarantine Checks
    if req.raw_absences < 0:
        errors.append("raw_absences cannot be negative.")
    if req.medical_leave_days < 0:
        errors.append("medical_leave_days cannot be negative.")
    if req.medical_leave_days > req.raw_absences and req.raw_absences >= 0:
        errors.append("medical_leave_days cannot exceed raw_absences.")
    if not re.match(r"^\d+-\d+-\d+$", req.ltp):
        errors.append("ltp must follow L-T-P numeric format (e.g., '3-0-0').")

    if errors:
        return "quarantine", errors

    return "ok", []

def sanitize_output_rationale(text: str) -> str:
    """Role 4 Output Guardrail: Ensures the system never issues a final determination."""
    lowered = text.lower()
    for phrase in FORBIDDEN_DETERMINATION_PHRASES:
        if phrase in lowered:
            return "[REDACTED BY OUTPUT GUARDRAIL: System attempted to issue a binding determination. Escalating to Officer.]"
    return text