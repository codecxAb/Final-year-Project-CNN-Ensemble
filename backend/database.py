"""
database.py — LungCare
==============================
SQLAlchemy SQLite database setup and ORM models.

Models:
    - Patient     : Represents a registered patient (linked to Telegram chat).
    - ScanRecord  : Each CT scan analysis result tied to a Patient.

Usage:
    from database import Base, engine, get_db, Patient, ScanRecord
"""

import os
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

# ─── Database URL ─────────────────────────────────────────────────────────────
# Stored in the same directory as the backend for portability.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'lungcare.db')}")

# ─── Engine & Session ─────────────────────────────────────────────────────────
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Required for SQLite + FastAPI
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ─── Base Model ───────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ─── ORM Models ───────────────────────────────────────────────────────────────

class Patient(Base):
    """
    Represents a patient in the triage system.

    Attributes:
        id               : Auto-increment primary key.
        patient_number   : Unique identifier entered by radiologist (e.g., P-1001).
        name             : Full name of the patient.
        telegram_chat_id : Telegram chat ID used to link the patient to the bot.
        scan_records     : One-to-many relationship with ScanRecord.
    """
    __tablename__ = "patients"

    id               = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_number   = Column(String(50), unique=True, nullable=True, index=True)
    name             = Column(String(200), nullable=False)
    telegram_chat_id = Column(String(50), unique=True, nullable=True)
    emergency_alert  = Column(Integer, default=0)
    unified_summary_cache = Column(Text, nullable=True)

    scan_records = relationship("ScanRecord", back_populates="patient", cascade="all, delete-orphan")
    documents    = relationship("PatientDocument", back_populates="patient", cascade="all, delete-orphan")
    summary      = relationship("PatientSummary", back_populates="patient", uselist=False, cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Patient id={self.id} number='{self.patient_number}' name='{self.name}'>"


class ScanRecord(Base):
    """
    Represents one CT scan analysis session for a patient.

    Attributes:
        id                 : Auto-increment primary key.
        patient_id         : FK → patients.id
        date               : When this scan was recorded (defaults to now).
        tumor_diameter_mm  : Measured largest diameter of nodule in millimetres.
        risk_level         : "Low" | "Medium" | "High" (WHO guideline thresholds).
        ai_report_draft    : LangGraph-generated radiology report text (nullable until generated).
        status             : Workflow status: "Pending" | "Draft" | "Approved".
        patient            : Back-reference to the Patient object.
    """
    __tablename__ = "scan_records"

    id                = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_id        = Column(Integer, ForeignKey("patients.id"), nullable=False)
    date              = Column(DateTime, default=datetime.utcnow, nullable=False)
    tumor_diameter_mm = Column(Float, nullable=False)
    risk_level        = Column(String(10), nullable=False)          # "Low" | "Medium" | "High"
    ai_report_draft   = Column(Text, nullable=True)                 # populated after /api/generate_report
    status            = Column(String(20), default="Pending", nullable=False)  # Pending | Draft | Approved
    doctor_notes      = Column(Text, nullable=True)                 # manual remarks from doctor
    x_coordinate      = Column(Integer, nullable=True, default=256) # 0-512 input grid

    # Many-to-one back-reference
    patient = relationship("Patient", back_populates="scan_records")

    def __repr__(self) -> str:
        return (
            f"<ScanRecord id={self.id} patient_id={self.patient_id} "
            f"risk='{self.risk_level}' status='{self.status}'>"
        )


class PatientDocument(Base):
    """
    Represents an uploaded historical medical document (PDF or Image).

    Attributes:
        id           : Auto-increment primary key.
        patient_id   : FK → patients.id
        file_name    : Original name of the uploaded file.
        file_path    : Absolute or relative path on disk where it is stored.
        doc_type     : Enum-like string ("pdf", "image").
        upload_date  : Automatically timestamped on insert.
    """
    __tablename__ = "patient_documents"

    id          = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_id  = Column(Integer, ForeignKey("patients.id"), nullable=False)
    file_name   = Column(String(255), nullable=False)
    file_path   = Column(String(500), nullable=False)
    doc_type    = Column(String(50), nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow, nullable=False)

    patient = relationship("Patient", back_populates="documents")

    def __repr__(self) -> str:
        return f"<PatientDocument id={self.id} type='{self.doc_type}' file='{self.file_name}'>"

class PatientSummary(Base):
    """
    Represents the synthesized, unified LangGraph summary of historical patient documents.
    """
    __tablename__ = "patient_summaries"

    id           = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_id   = Column(Integer, ForeignKey("patients.id"), unique=True, nullable=False)
    summary_text = Column(Text, nullable=False)
    generated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    patient = relationship("Patient", back_populates="summary")

    def __repr__(self) -> str:
        return f"<PatientSummary id={self.id} patient_id={self.patient_id}>"

# ─── DB Initializer ───────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables and seed demo patients if the DB is empty."""
    Base.metadata.create_all(bind=engine)

    from sqlalchemy import text
    # Safe migrations for existing databases
    migrations = [
        "ALTER TABLE scan_records ADD COLUMN x_coordinate INTEGER DEFAULT 256",
        "ALTER TABLE patients ADD COLUMN patient_number VARCHAR(50)",
    ]
    with SessionLocal() as session:
        for sql in migrations:
            try:
                session.execute(text(sql))
                session.commit()
            except Exception:
                session.rollback()

    # Backfill patient_number for existing patients that don't have one
    with SessionLocal() as session:
        patients_without_number = session.query(Patient).filter(
            (Patient.patient_number == None) | (Patient.patient_number == "")
        ).all()
        for p in patients_without_number:
            p.patient_number = f"P-{1000 + p.id}"
        if patients_without_number:
            session.commit()
            print(f"[DB] Backfilled patient_number for {len(patients_without_number)} patients.")

    # Seed demo patients on first run
    with SessionLocal() as session:
        if session.query(Patient).count() == 0:
            demo_patients = [
                Patient(name="Alice Johnson",  patient_number="P-1001", telegram_chat_id=None),
                Patient(name="Bob Williams",   patient_number="P-1002", telegram_chat_id=None),
                Patient(name="Carol Martinez", patient_number="P-1003", telegram_chat_id=None),
            ]
            session.add_all(demo_patients)
            session.commit()
            print("[DB] Seeded 3 demo patients.")


# ─── FastAPI Dependency ────────────────────────────────────────────────────────

def get_db():
    """
    FastAPI dependency that yields a DB session and ensures it is closed
    after the request completes (even on error).

    Usage in route:
        @app.get("/items")
        def read_items(db: Session = Depends(get_db)):
            ...
    """
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
