"""
main.py — LungCare Triage Backend
====================================
FastAPI application — the central hub of the LungCare Triage system.

Routes:
    POST /api/analyze           → Run AI risk scoring on a patient scan
    POST /api/generate_report   → Draft radiology report via LangGraph
    GET  /api/patients/{id}/history → Fetch patient scan history + growth rate
    POST /api/patient_chat      → Patient Q&A via Telegram + LangGraph
    POST /api/patients/{id}/upload_document → Upload historical records
    GET  /api/patients/{id}/summary → Synthesize unified patient history
    POST /api/scans/{id}/remarks → Save final radiologist sign-off

Start with:
    uvicorn main:app --reload --port 8000
"""

import logging
import os
from datetime import datetime, date
from typing import List, Optional

from dotenv import load_dotenv, find_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ai_pipeline import calculate_growth_rate, calculate_risk_score, predict_nodule
from database import Patient, ScanRecord, PatientDocument, PatientSummary, get_db, init_db
from langgraph_agent import answer_patient_question, generate_radiology_report, generate_patient_summary

# ─── Setup ────────────────────────────────────────────────────────────────────
load_dotenv(find_dotenv())
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Initialise the database (creates tables + seeds demo patients on first run)
init_db()

# ─── FastAPI App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="🫁 LungCare Triage API",
    description=(
        "Production-ready Radiology Triage & Monitoring System.\n\n"
        "Built on: **FastAPI** + **PyTorch 3D CNN** + **LangGraph/Groq** + **SQLite**.\n\n"
        "For demo mode, the scan analysis endpoint accepts a mocked tumour diameter "
        "instead of a real .mhd file upload so you can explore the full pipeline immediately."
    ),
    version="1.0.0",
    contact={"name": "LungCare Dev Team"},
    license_info={"name": "MIT"},
)

# ─── CORS Middleware ──────────────────────────────────────────────────────────
# Allow the Streamlit dashboard (localhost:8501) and all other origins for demo.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════════════════

class AnalyzeRequest(BaseModel):
    """(Deprecated) Request body for POST /api/analyze"""
    pass


class AnalyzeResponse(BaseModel):
    """Response body for POST /api/analyze"""
    scan_id                : int
    patient_id             : int
    patient_name           : str
    tumor_diameter_mm      : float
    risk_level             : str     # "Low" | "Medium" | "High"
    status                 : str
    message                : str
    requires_attention     : bool    # True if emergency/high-risk
    x_coordinate           : int
    malignancy_probability : float   # 0.0–1.0 raw CNN output
    predicted_class        : str     # "Benign" | "Malignant"
    confidence             : float   # model confidence 0.0–1.0


class GenerateReportRequest(BaseModel):
    """Request body for POST /api/generate_report"""
    scan_id : int = Field(..., description="ID of the ScanRecord to generate a report for", example=1)

    class Config:
        json_schema_extra = {"example": {"scan_id": 1}}


class GenerateReportResponse(BaseModel):
    """Response body for POST /api/generate_report"""
    scan_id      : int
    patient_name : str
    risk_level   : str
    report_draft : str
    status       : str


class ApproveReportRequest(BaseModel):
    """Request body for POST /api/approve_report"""
    scan_id       : int
    approved_text : str = Field(..., description="The (optionally edited) final report text")


class ScanHistoryItem(BaseModel):
    """Single item in patient history response"""
    scan_id           : int
    date              : str
    tumor_diameter_mm : float
    risk_level        : str
    status            : str
    has_report        : bool
    doctor_notes      : Optional[str]
    x_coordinate      : int


class PatientHistoryResponse(BaseModel):
    """Response for GET /api/patients/{id}/history"""
    patient_id   : int
    patient_name : str
    scans        : List[ScanHistoryItem]
    growth_rate  : dict


class PatientListItem(BaseModel):
    """Single patient summary"""
    id               : int
    patient_number   : Optional[str]
    name             : str
    telegram_chat_id : Optional[str]
    scan_count       : int
    latest_risk      : Optional[str]
    emergency_alert  : bool
    latest_scan_date : Optional[str]


class CreatePatientRequest(BaseModel):
    """Request body for POST /api/patients"""
    patient_number : str = Field(..., min_length=1, description="Unique patient number (e.g. P-1001)")
    name           : str = Field(..., min_length=1, description="Full name of the new patient")


class CreatePatientResponse(BaseModel):
    """Response body for POST /api/patients"""
    id             : int
    patient_number : str
    name           : str


class ChatRequest(BaseModel):
    """Request body for POST /api/patient_chat"""
    telegram_chat_id : str  = Field(..., description="Telegram chat ID of the patient", example="123456789")
    message          : str  = Field(..., description="Patient's natural language question", example="What does my risk score mean?")
    chat_history     : list = Field(default=[], description="List of previous messages: [{'role': 'user', 'content': '...'}, ...]")

    class Config:
        json_schema_extra = {
            "example": {
                "telegram_chat_id": "123456789",
                "message"         : "What does my risk score mean?",
                "chat_history"    : []
            }
        }


class ChatResponse(BaseModel):
    """Response body for POST /api/patient_chat"""
    patient_name : Optional[str]
    response     : str


# ══════════════════════════════════════════════════════════════════════════════
# CLINICIAN API ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/api/patients",
    response_model=List[PatientListItem],
    tags=["Clinician API"],
    summary="List all patients",
)
def list_patients(db: Session = Depends(get_db)):
    """Return all registered patients with their latest risk level."""
    patients = db.query(Patient).all()
    result = []
    for p in patients:
        scans      = p.scan_records
        latest     = max(scans, key=lambda s: s.date) if scans else None
        result.append(PatientListItem(
            id               = p.id,
            patient_number   = p.patient_number,
            name             = p.name,
            telegram_chat_id = p.telegram_chat_id,
            scan_count       = len(scans),
            latest_risk      = latest.risk_level if latest else None,
            emergency_alert  = bool(p.emergency_alert),
            latest_scan_date = latest.date.strftime("%Y-%m-%d %H:%M") if latest and latest.date else None,
        ))
    return result


@app.post(
    "/api/patients",
    response_model=CreatePatientResponse,
    tags=["Clinician API"],
    summary="Create a new patient",
    status_code=status.HTTP_201_CREATED,
)
def create_patient(payload: CreatePatientRequest, db: Session = Depends(get_db)):
    """Register a new patient in the triage system."""
    existing = db.query(Patient).filter(Patient.patient_number == payload.patient_number.strip()).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Patient number {payload.patient_number} already exists.")
    patient = Patient(name=payload.name.strip(), patient_number=payload.patient_number.strip())
    db.add(patient)
    db.commit()
    db.refresh(patient)
    logger.info(f"[/api/patients] Created patient: {patient.name} ({patient.patient_number})")
    return CreatePatientResponse(id=patient.id, patient_number=patient.patient_number, name=patient.name)


@app.get("/api/patients/lookup", tags=["Clinician API"], summary="Lookup patient by number")
def lookup_patient(number: str, db: Session = Depends(get_db)):
    """Find a patient by their patient_number. Returns patient data or 404."""
    patient = db.query(Patient).filter(Patient.patient_number == number.strip()).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")
    scans = sorted(patient.scan_records, key=lambda s: s.date, reverse=True)
    latest = scans[0] if scans else None
    return {
        "id": patient.id,
        "patient_number": patient.patient_number,
        "name": patient.name,
        "scan_count": len(scans),
        "latest_risk": latest.risk_level if latest else None,
        "latest_scan_date": latest.date.strftime("%Y-%m-%d %H:%M") if latest and latest.date else None,
    }


@app.get("/api/patients/by-folder", tags=["Clinician API"], summary="Patients grouped by triage folder")
def patients_by_folder(db: Session = Depends(get_db)):
    """Returns patients grouped into Critical / Under Observation / Clear folders."""
    patients = db.query(Patient).all()
    folders = {"critical": [], "under_observation": [], "clear": []}

    for p in patients:
        scans = sorted(p.scan_records, key=lambda s: s.date, reverse=True)
        latest = scans[0] if scans else None
        risk = latest.risk_level if latest else None

        item = {
            "id": p.id,
            "patient_number": p.patient_number,
            "name": p.name,
            "scan_count": len(scans),
            "latest_risk": risk,
            "latest_scan_date": latest.date.strftime("%Y-%m-%d %H:%M") if latest and latest.date else None,
            "emergency_alert": bool(p.emergency_alert),
        }

        if risk == "High":
            folders["critical"].append(item)
        elif risk == "Medium":
            folders["under_observation"].append(item)
        else:
            folders["clear"].append(item)

    return folders


@app.get("/api/patients/{patient_id}/full", tags=["Clinician API"], summary="Full patient detail")
def get_patient_full(patient_id: int, db: Session = Depends(get_db)):
    """Comprehensive patient detail: info, all scans, growth rate, approved reports."""
    patient = db.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")

    scans = sorted(patient.scan_records, key=lambda s: s.date)
    diameters = [s.tumor_diameter_mm for s in scans]
    growth = calculate_growth_rate(diameters)

    scan_items = []
    for s in scans:
        scan_items.append({
            "scan_id": s.id,
            "date": s.date.strftime("%Y-%m-%d %H:%M") if s.date else "Unknown",
            "tumor_diameter_mm": s.tumor_diameter_mm,
            "risk_level": s.risk_level,
            "status": s.status,
            "has_report": bool(s.ai_report_draft),
            "doctor_notes": s.doctor_notes,
            "x_coordinate": getattr(s, 'x_coordinate', 256) or 256,
        })

    latest = scans[-1] if scans else None

    return {
        "id": patient.id,
        "patient_number": patient.patient_number,
        "name": patient.name,
        "latest_risk": latest.risk_level if latest else None,
        "emergency_alert": bool(patient.emergency_alert),
        "scan_count": len(scans),
        "scans": scan_items,
        "growth_rate": growth,
    }

@app.post(
    "/api/analyze",
    response_model=AnalyzeResponse,
    tags=["Clinician API"],
    summary="Run AI scan analysis",
    status_code=status.HTTP_201_CREATED,
)
def analyze_scan(
    patient_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Accepts an uploaded image file (2D slice), converts to 3D tensor,
    runs the PyTorch model inference, mapping probability back to size.
    """
    # 1. Verify patient exists
    patient = db.get(Patient, patient_id)
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with ID {patient_id} not found.",
        )

    # 2. Run real PyTorch inference
    try:
        image_bytes = file.file.read()
        inference_result = predict_nodule(image_bytes)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    prob = inference_result["malignancy_probability"]
    
    # 3. Size heuristic based on AI probability
    # If probability is high, size is likely larger.
    # We map 0.0 -> <3mm, 0.5 -> ~6mm, 1.0 -> >10mm linearly for the sake of demonstration
    assumed_diameter_mm = prob * 15.0
    if assumed_diameter_mm < 2.0:
        assumed_diameter_mm = 2.5  # base minimum size

    # 4. Calculate risk using WHO / Fleischner thresholds
    risk_level = calculate_risk_score(assumed_diameter_mm)
    requires_attention = True if risk_level == "High" else False
    
    logger.info(f"[/api/analyze] Patient {patient.name}: {assumed_diameter_mm:.1f}mm → {risk_level} (Prob: {prob})")

    # 5. Save new ScanRecord to DB
    import random
    x_coord = random.randint(100, 400)
    scan = ScanRecord(
        patient_id        = patient_id,
        date              = datetime.utcnow(),
        tumor_diameter_mm = round(assumed_diameter_mm, 2),
        risk_level        = risk_level,
        status            = "Pending",
        x_coordinate      = x_coord,
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    message = f"Scan recorded. Risk level: {risk_level} (AI Prob: {prob:.2f})."
    if requires_attention:
        message += " WARNING: Immediate review recommended."

    return AnalyzeResponse(
        scan_id                = scan.id,
        patient_id             = patient.id,
        patient_name           = patient.name,
        tumor_diameter_mm      = round(assumed_diameter_mm, 2),
        risk_level             = risk_level,
        status                 = "Pending",
        message                = message,
        requires_attention     = requires_attention,
        x_coordinate           = x_coord,
        malignancy_probability = inference_result["malignancy_probability"],
        predicted_class        = inference_result["predicted_class"],
        confidence             = inference_result["confidence"],
    )


@app.post(
    "/api/generate_report",
    response_model=GenerateReportResponse,
    tags=["Clinician API"],
    summary="Draft AI radiology report",
)
def generate_report(payload: GenerateReportRequest, db: Session = Depends(get_db)):
    """
    Triggers LangGraph Graph 1 (Radiologist Assistant) to draft a professional
    clinical radiology report and saves it to the ScanRecord.
    """
    # 1. Look up the scan
    scan = db.get(ScanRecord, payload.scan_id)
    if not scan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ScanRecord with ID {payload.scan_id} not found.",
        )

    patient = db.get(Patient, scan.patient_id)

    # 2. Run LangGraph Radiologist Agent
    scan_date_str = scan.date.strftime("%Y-%m-%d") if scan.date else str(date.today())
    logger.info(f"[/api/generate_report] Generating report for scan {scan.id}...")
    report_text = generate_radiology_report(
        patient_name      = patient.name if patient else "Unknown Patient",
        tumor_diameter_mm = scan.tumor_diameter_mm,
        risk_level        = scan.risk_level,
        scan_date         = scan_date_str,
    )

    # 3. Save draft to DB
    scan.ai_report_draft = report_text
    scan.status          = "Draft"
    db.commit()
    db.refresh(scan)

    return GenerateReportResponse(
        scan_id      = scan.id,
        patient_name = patient.name if patient else "Unknown",
        risk_level   = scan.risk_level,
        report_draft = report_text,
        status       = "Draft",
    )


@app.post(
    "/api/approve_report",
    tags=["Clinician API"],
    summary="Approve & finalise a radiology report",
)
def approve_report(payload: ApproveReportRequest, db: Session = Depends(get_db)):
    """
    Clinician reviews the AI draft, optionally edits it, then approves.
    Sets status to 'Approved' and saves the final text.
    """
    scan = db.get(ScanRecord, payload.scan_id)
    if not scan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"ScanRecord with ID {payload.scan_id} not found.",
        )

    scan.ai_report_draft = payload.approved_text
    scan.status          = "Approved"
    db.commit()

    return {"message": f"Report for scan {payload.scan_id} approved successfully.", "status": "Approved"}


@app.get(
    "/api/patients/{patient_id}/history",
    response_model=PatientHistoryResponse,
    tags=["Clinician API"],
    summary="Get patient scan history & growth rate",
)
def get_patient_history(patient_id: int, db: Session = Depends(get_db)):
    """
    Returns all historical scan records for a patient, plus a computed
    nodule growth rate analysis (stable / slow growth / rapid growth).
    """
    patient = db.get(Patient, patient_id)
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with ID {patient_id} not found.",
        )

    # Sort scans oldest-first for growth rate calculation
    scans = sorted(patient.scan_records, key=lambda s: s.date)
    diameters = [s.tumor_diameter_mm for s in scans]
    growth    = calculate_growth_rate(diameters)

    scan_items = [
        ScanHistoryItem(
            scan_id           = s.id,
            date              = s.date.strftime("%Y-%m-%d %H:%M") if s.date else "Unknown",
            tumor_diameter_mm = s.tumor_diameter_mm,
            risk_level        = s.risk_level,
            status            = s.status,
            has_report        = bool(s.ai_report_draft),
            doctor_notes      = s.doctor_notes,
            x_coordinate      = getattr(s, 'x_coordinate', 256) or 256,
        )
        for s in scans
    ]

    return PatientHistoryResponse(
        patient_id   = patient.id,
        patient_name = patient.name,
        scans        = scan_items,
        growth_rate  = growth,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PATIENT API ROUTES  (consumed by the Telegram Bot)
# ══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/api/patient_chat",
    response_model=ChatResponse,
    tags=["Patient API"],
    summary="Patient Q&A via LangGraph (used by Telegram Bot)",
)
def patient_chat(payload: ChatRequest, db: Session = Depends(get_db)):
    """
    The Telegram bot hits this endpoint with the patient's message.

    1. Looks up the patient via telegram_chat_id.
    2. Fetches their latest ScanRecord as context.
    3. Runs LangGraph Graph 2 (Patient Support Bot).
    4. Returns the safe, empathetic LLM response.
    """
    # 1. Find patient by Telegram chat ID
    patient = (
        db.query(Patient)
        .filter(Patient.telegram_chat_id == payload.telegram_chat_id)
        .first()
    )

    # If patient not found, still give a generic answer
    medical_record = {}
    patient_name   = None

    if patient:
        patient_name = patient.name
        scans = sorted(patient.scan_records, key=lambda s: s.date, reverse=True)
        if scans:
            latest = scans[0]
            medical_record = {
                "risk_level"       : latest.risk_level,
                "tumor_diameter_mm": latest.tumor_diameter_mm,
                "date"             : latest.date.strftime("%Y-%m-%d") if latest.date else None,
                "status"           : latest.status,
                "unified_history"  : patient.summary.summary_text if patient.summary else None,
            }

    logger.info(f"[/api/patient_chat] Processing question from chat_id={payload.telegram_chat_id}")

    # 2. Run LangGraph Patient Support Agent
    agent_result = answer_patient_question(
        question       = payload.message,
        chat_history   = payload.chat_history,
        medical_record = medical_record,
    )

    response_text = agent_result["safe_response"]
    emergency_flag = agent_result["emergency_flag"]

    if emergency_flag and patient:
        logger.warning(f"[/api/patient_chat] Emergency flag triggered for patient {patient.id}")
        patient.emergency_alert = 1
        db.commit()

    return ChatResponse(patient_name=patient_name, response=response_text)


# ══════════════════════════════════════════════════════════════════════════════
# DATA ROOM API ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/patients/{patient_id}/upload_document", tags=["Data Room"], summary="Upload a historical patient document")
def upload_document(patient_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Uploads a PDF or Image locally and registers it in the PatientDocument DB."""
    patient = db.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")
        
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", f"{patient_id}_{int(datetime.now().timestamp())}_{file.filename}")
    
    with open(file_path, "wb") as f:
        f.write(file.file.read())
        
    ext = file.filename.split('.')[-1].lower()
    doc_type = "pdf" if ext == "pdf" else "image"
    
    doc = PatientDocument(patient_id=patient_id, file_name=file.filename, file_path=file_path, doc_type=doc_type)
    db.add(doc)
    db.commit()
    logger.info(f"[main] Uploaded {doc_type} for patient {patient_id} -> {file.filename}")
    return {"status": "success", "file_name": file.filename, "doc_id": doc.id}


@app.get("/api/patients/{patient_id}/summary", tags=["Data Room"], summary="Generate Unified Patient History")
def get_patient_summary(patient_id: int, refresh: bool = False, db: Session = Depends(get_db)):
    """Fetches the existing PatientSummary from DB, or invokes Graph 3 if missing or refresh=True."""
    patient = db.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")
        
    if not refresh and patient.summary:
        logger.info(f"[main] Returning cached summary for {patient.name}")
        return {"summary": patient.summary.summary_text}
        
    docs = [{"file_path": d.file_path, "doc_type": d.doc_type} for d in patient.documents]
    if not docs:
        return {"summary": "No historical documents found to summarize."}
    
    logger.info(f"[main] Requesting unified summary for {patient.name} ({len(docs)} documents)")
    summary_text = generate_patient_summary(patient.name, docs)
    
    if patient.summary:
        patient.summary.summary_text = summary_text
        patient.summary.generated_at = datetime.utcnow()
    else:
        new_summary = PatientSummary(patient_id=patient_id, summary_text=summary_text)
        db.add(new_summary)
        
    db.commit()
    return {"summary": summary_text}


from pydantic import BaseModel
class RemarksRequest(BaseModel):
    remarks: str

@app.get("/api/scans/{scan_id}", tags=["Clinician API"], summary="Get scan details including AI report")
def get_scan(scan_id: int, db: Session = Depends(get_db)):
    scan = db.get(ScanRecord, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found.")
    
    return {
        "scan_id": scan.id,
        "date": scan.date.isoformat() if scan.date else None,
        "tumor_diameter_mm": scan.tumor_diameter_mm,
        "risk_level": scan.risk_level,
        "status": scan.status,
        "ai_report_draft": scan.ai_report_draft,
        "doctor_notes": scan.doctor_notes,
        "x_coordinate": getattr(scan, 'x_coordinate', 256) or 256,
    }


@app.post("/api/scans/{scan_id}/remarks", tags=["Data Room"], summary="Save doctor notes")
def save_remarks(scan_id: int, payload: RemarksRequest, db: Session = Depends(get_db)):
    """Appends final radiologist physician notes to the scan record."""
    scan = db.get(ScanRecord, scan_id)
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found.")
        
    scan.doctor_notes = payload.remarks
    db.commit()
    logger.info(f"[main] Saved doctor notes for scan {scan_id}")
    return {"status": "success"}


@app.post(
    "/api/bot/register",
    tags=["Patient API"],
    summary="Link Telegram chat ID to a patient record",
)
def register_bot_patient(
    telegram_chat_id : str,
    patient_id       : int,
    db               : Session = Depends(get_db),
):
    """
    Called by the Telegram bot's /start command to link a patient's DB record
    to their Telegram chat ID.
    """
    patient = db.get(Patient, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")

    patient.telegram_chat_id = telegram_chat_id
    db.commit()
    return {"message": f"Patient '{patient.name}' linked to Telegram chat {telegram_chat_id}."}


# ─── Stats / Overview ─────────────────────────────────────────────────────────

@app.get("/api/stats", tags=["Clinician API"], summary="Dashboard aggregate statistics")
def get_stats(db: Session = Depends(get_db)):
    """Returns aggregate statistics for the overview dashboard."""
    patients = db.query(Patient).all()
    all_scans = db.query(ScanRecord).order_by(ScanRecord.date.desc()).all()

    high_risk_count = 0
    pending_reviews = 0
    emergency_count = 0

    for p in patients:
        if p.emergency_alert:
            emergency_count += 1
        scans = sorted(p.scan_records, key=lambda s: s.date, reverse=True)
        if scans:
            if scans[0].risk_level == "High":
                high_risk_count += 1

    for s in all_scans:
        if s.status == "Pending":
            pending_reviews += 1

    recent_scans = []
    for s in all_scans[:5]:
        patient = db.get(Patient, s.patient_id)
        recent_scans.append({
            "scan_id": s.id,
            "patient_name": patient.name if patient else "Unknown",
            "date": s.date.strftime("%Y-%m-%d %H:%M") if s.date else "Unknown",
            "risk_level": s.risk_level,
            "status": s.status,
            "tumor_diameter_mm": s.tumor_diameter_mm,
        })

    return {
        "total_patients": len(patients),
        "high_risk_count": high_risk_count,
        "pending_reviews": pending_reviews,
        "emergency_count": emergency_count,
        "recent_scans": recent_scans,
    }


# ─── Health Check ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"], summary="Health check")
def health_check():
    """Simple heartbeat endpoint for uptime monitoring."""
    return {"status": "ok", "service": "LungCare Triage API", "version": "1.0.0"}


# ─── Dev Runner ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
