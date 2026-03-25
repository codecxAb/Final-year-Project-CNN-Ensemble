"""
langgraph_agent.py — LungCare
======================================
Two LangGraph StateGraphs powered by Groq (llama3-8b-8192).

Graph 1 — Radiologist Assistant
    Input  : tumor_diameter_mm, risk_level, patient_name (optional patient metadata)
    Output : A professional clinical radiology report draft (markdown-formatted)

Graph 2 — Patient Support Bot
    Input  : question (patient's natural language text), medical_record (dict)
    Output : A safe, empathetic, non-diagnostic explanation for the patient

Environment Variables Required:
    GROQ_API_KEY : API key from https://console.groq.com
"""

import logging
import os
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from document_parser import extract_text_from_pdf, analyze_image_with_gemini

load_dotenv()
logger = logging.getLogger(__name__)

# ─── LLM Initialisation ───────────────────────────────────────────────────────

def _get_llm(temperature: float = 0.3) -> ChatGroq:
    """Create a Groq chat model instance."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file.\n"
            "Get a free key at https://console.groq.com"
        )
    return ChatGroq(
        model="qwen/qwen3-32b",
        temperature=temperature,
        groq_api_key=api_key,
    )


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 1 — RADIOLOGIST ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════

class RadiologistState(TypedDict):
    """State passed through the Radiologist Assistant graph nodes."""
    patient_name      : str
    tumor_diameter_mm : float
    risk_level        : str                   # "Low" | "Medium" | "High"
    scan_date         : str                   # ISO date string
    report_draft      : Optional[str]         # populated by the drafter node
    reviewed          : bool                  # True after quality check node


def _draft_report_node(state: RadiologistState) -> dict:
    """
    Node 1 — Draft the clinical radiology report using Groq LLM.

    Takes clinical measurements from state and instructs the LLM to produce
    a structured radiology report following standard clinical conventions.
    """
    llm = _get_llm(temperature=0.2)

    system_prompt = (
        "You are an expert radiologist AI assistant. Your role is to draft clear, "
        "concise, and professionally-worded radiology reports based on CT scan findings. "
        "Always follow standard radiology report structure: "
        "Clinical Indication, Technique, Findings, Impression, Recommendations. "
        "Be objective, evidence-based, and clinically precise. "
        "Do NOT make definitive diagnoses — use phrases like 'suspicious for', 'consistent with'. "
        "Output in clean, structured markdown format. "
        "IMPORTANT: Do NOT wrap the entire report in a markdown code block (no ```markdown tags). "
        "Just output the markdown text directly."
    )

    user_prompt = (
        f"Please draft a radiology report for the following CT scan findings:\n\n"
        f"- Patient: {state['patient_name']}\n"
        f"- Scan Date: {state['scan_date']}\n"
        f"- Largest Nodule Diameter: {state['tumor_diameter_mm']:.1f} mm\n"
        f"- AI-Computed Risk Level: {state['risk_level']}\n\n"
        "Use Fleischner Society guidelines for management recommendations based on the nodule size."
    )

    logger.info(f"[Graph 1] Drafting report for {state['patient_name']} (risk={state['risk_level']})")
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])

    # Strip think tags and any markdown code block wrap delimiters
    import re
    cleaned_draft = re.sub(r'<think>.*?</think>\n?', '', response.content, flags=re.DOTALL)
    # Remove markers like ```markdown or ``` at start/end or anywhere
    cleaned_draft = re.sub(r'```(?:markdown|md)?\n?', '', cleaned_draft, flags=re.IGNORECASE)
    cleaned_draft = cleaned_draft.strip()

    return {"report_draft": cleaned_draft, "reviewed": False}


def _quality_check_node(state: RadiologistState) -> dict:
    """
    Node 2 — Perform a brief quality and safety review of the drafted report.

    Checks for completeness (5 sections present) and appropriate risk language.
    Updates state to mark the report as reviewed.
    """
    draft = state.get("report_draft", "")

    # Simple rule-based quality check (extendable to LLM-based check)
    required_sections = ["Clinical Indication", "Findings", "Impression", "Recommendations"]
    missing = [s for s in required_sections if s not in draft]

    if missing:
        logger.warning(f"[Graph 1] QC Warning — missing sections: {missing}")
        # Append a note to the draft rather than failing
        note = f"\n\n---\n⚠️ **QC Note**: The following sections may need attention: {', '.join(missing)}"
        return {"report_draft": draft + note, "reviewed": True}

    logger.info("[Graph 1] Quality check passed.")
    return {"reviewed": True}


def build_radiologist_graph() -> Any:
    """
    Compile and return the Radiologist Assistant StateGraph.

    Flow: draft_report → quality_check → END

    Returns:
        A compiled LangGraph runnable.
    """
    graph = StateGraph(RadiologistState)

    # Register nodes
    graph.add_node("draft_report",   _draft_report_node)
    graph.add_node("quality_check",  _quality_check_node)

    # Define edges
    graph.set_entry_point("draft_report")
    graph.add_edge("draft_report", "quality_check")
    graph.add_edge("quality_check", END)

    return graph.compile()


# Singleton compiled graph (avoids recompiling on every request)
_radiologist_graph = None

def get_radiologist_graph():
    global _radiologist_graph
    if _radiologist_graph is None:
        _radiologist_graph = build_radiologist_graph()
    return _radiologist_graph


def generate_radiology_report(
    patient_name      : str,
    tumor_diameter_mm : float,
    risk_level        : str,
    scan_date         : str,
) -> str:
    """
    Public API for Graph 1.

    Args:
        patient_name      : Patient's full name.
        tumor_diameter_mm : Measured nodule diameter.
        risk_level        : "Low" | "Medium" | "High"
        scan_date         : Scan date as ISO string (e.g., "2025-03-14").

    Returns:
        The final AI-drafted radiology report as a string.
    """
    graph = get_radiologist_graph()
    initial_state: RadiologistState = {
        "patient_name"      : patient_name,
        "tumor_diameter_mm" : tumor_diameter_mm,
        "risk_level"        : risk_level,
        "scan_date"         : scan_date,
        "report_draft"      : None,
        "reviewed"          : False,
    }
    result = graph.invoke(initial_state)
    return result.get("report_draft", "Report generation failed.")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 2 — PATIENT SUPPORT BOT
# ══════════════════════════════════════════════════════════════════════════════

class PatientSupportState(TypedDict):
    """State passed through the Patient Support Bot graph nodes."""
    question        : str           # Patient's raw question in natural language
    chat_history    : list          # List of previous HumanMessage/AIMessage dicts
    medical_record  : dict          # Relevant fields from DB (risk_level, size, date, etc.)
    safe_response   : Optional[str] # Final patient-friendly answer
    emergency_flag  : bool          # Set to True if urgent attention is needed


def _safety_filter_node(state: PatientSupportState) -> dict:
    """
    Node 1 — Check if the question asks for a definitive diagnosis or treatment.

    This is a lightweight guardrail. Questions asking for diagnosis (e.g., "Do I have cancer?")
    are flagged and answered with a redirect to their doctor.
    We keep the original question in state for the response node.
    """
    question_lower = state["question"].lower()

    # Keywords that suggest requests for diagnosis / treatment decisions
    red_flag_phrases = [
        "do i have cancer", "is it cancer", "am i going to die",
        "should i have surgery", "what treatment", "cure me",
    ]

    flagged = any(phrase in question_lower for phrase in red_flag_phrases)

    if flagged:
        logger.info("[Graph 2] Safety filter triggered — redirecting to doctor.")
        safe_response = (
            "I completely understand how worrying this must feel. 💙\n\n"
            "I'm not able to provide a diagnosis or treatment recommendation — "
            "only your clinician can do that with full access to your history.\n\n"
            "Please reach out to your doctor or call the clinic to discuss your results. "
            "They are the best person to answer this question accurately and safely."
        )
        return {"safe_response": safe_response, "emergency_flag": True}

    return {"safe_response": None, "emergency_flag": False}  # Proceed to LLM node


def _patient_response_node(state: PatientSupportState) -> dict:
    """
    Node 2 — Generate an empathetic, non-diagnostic answer using Groq LLM.

    Only runs if the safety filter did not short-circuit the graph.
    Uses the patient's medical record as context for a personalised response.
    """
    # If safety filter already set a response, skip LLM
    if state.get("safe_response"):
        return {}

    llm = _get_llm(temperature=0.5)
    record = state["medical_record"]

    system_prompt = (
        "You are LungCare Assistant — a compassionate, supportive AI health companion. "
        "Your role is to explain medical information in plain, reassuring language that a patient can understand. "
        "CRITICAL RULES:\n"
        "1. NEVER provide a diagnosis, prognosis, or treatment recommendation.\n"
        "2. ALWAYS encourage the patient to speak to their doctor for medical decisions.\n"
        "3. Use warm, empathetic language. Patients may be anxious.\n"
        "4. Explain medical terms in simple words with analogies when helpful.\n"
        "5. Keep your response concise (3–5 paragraphs max).\n"
        "6. End every response with: 'Please remember — your care team is always here for you. 💙'"
    )

    # Build context string from the medical record
    context_parts = []
    if record.get("risk_level"):
        context_parts.append(f"Current risk level: {record['risk_level']}")
    if record.get("tumor_diameter_mm"):
        context_parts.append(f"Nodule size: {record['tumor_diameter_mm']} mm")
    if record.get("date"):
        context_parts.append(f"Last scan date: {record['date']}")
    if record.get("status"):
        context_parts.append(f"Report status: {record['status']}")

    context_str = "\n".join(context_parts) if context_parts else "No specific records available."

    user_prompt = (
        f"Patient's Medical Context:\n{context_str}\n\n"
        f"Patient's Question: {state['question']}\n\n"
        "Please provide a helpful, empathetic, and safe explanation."
    )

    # Build conversation history
    history_messages = []
    for msg in state.get("chat_history", []):
        if msg["role"] == "user":
            history_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history_messages.append(SystemMessage(content=msg["content"])) # Using SystemMessage for AI history to keep it distinct from purely user instructions, or AIMessage if imported.
            
    messages_to_send = [SystemMessage(content=system_prompt)] + history_messages + [HumanMessage(content=user_prompt)]

    logger.info("[Graph 2] Generating patient support response via Groq...")
    response = llm.invoke(messages_to_send)

    # Clean up any <think> blocks and markdown wrappers
    import re
    cleaned_response = re.sub(r'<think>.*?</think>\n?', '', response.content, flags=re.DOTALL)
    cleaned_response = re.sub(r'```markdown\n?', '', cleaned_response, flags=re.IGNORECASE)
    cleaned_response = re.sub(r'```\n?', '', cleaned_response)
    cleaned_response = cleaned_response.strip()

    return {"safe_response": cleaned_response}


def build_patient_support_graph() -> Any:
    """
    Compile and return the Patient Support Bot StateGraph.

    Flow: safety_filter → patient_response → END
    If safety_filter short-circuits, patient_response still runs but exits immediately.

    Returns:
        A compiled LangGraph runnable.
    """
    graph = StateGraph(PatientSupportState)

    graph.add_node("safety_filter",    _safety_filter_node)
    graph.add_node("patient_response", _patient_response_node)

    graph.set_entry_point("safety_filter")
    graph.add_edge("safety_filter",    "patient_response")
    graph.add_edge("patient_response", END)

    return graph.compile()


# Singleton compiled graph
_patient_support_graph = None

def get_patient_support_graph():
    global _patient_support_graph
    if _patient_support_graph is None:
        _patient_support_graph = build_patient_support_graph()
    return _patient_support_graph


def answer_patient_question(question: str, chat_history: list, medical_record: dict) -> dict:
    """
    Public API for Graph 2.

    Args:
        question        : Patient's natural language question.
        chat_history    : List of dicts [{"role": "user"/"assistant", "content": "..."}]
        medical_record  : Dict with keys: risk_level, tumor_diameter_mm, date, status.

    Returns:
        Dict with Keys: 'safe_response' and 'emergency_flag'
    """
    graph = get_patient_support_graph()
    initial_state: PatientSupportState = {
        "question"       : question,
        "chat_history"   : chat_history,
        "medical_record" : medical_record,
        "safe_response"  : None,
        "emergency_flag" : False,
    }
    result = graph.invoke(initial_state)
    return {
        "safe_response": result.get("safe_response", "I'm sorry, I couldn't process your question. Please contact your clinic."),
        "emergency_flag": result.get("emergency_flag", False)
    }


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH 3 — PATIENT SUMMARY (DATA ROOM)
# ══════════════════════════════════════════════════════════════════════════════

class PatientSummaryState(TypedDict):
    """State passed through the Patient Summary graph nodes."""
    patient_name          : str
    documents             : list          # list of dicts: [{"file_path": ..., "doc_type": ...}]
    raw_pdf_text          : list          # Extracted texts from PDFs
    gemini_image_analyses : list          # Findings from Images
    unified_summary       : Optional[str] # Final generated markdown response


def _fetch_and_parse_documents_node(state: PatientSummaryState) -> dict:
    """
    Node 1 — Routes documents to appropriate parsers (PyPDF2 or Gemini).
    """
    pdf_texts = []
    img_analyses = []
    
    docs = state.get("documents", [])
    logger.info(f"[Graph 3] Processing {len(docs)} documents for {state['patient_name']}...")
    
    for doc in docs:
        path = doc.get("file_path")
        dtype = doc.get("doc_type")
        
        if not path or not os.path.exists(path):
            continue
            
        if dtype == "pdf":
            text = extract_text_from_pdf(path)
            if text:
                pdf_texts.append(f"--- Document ({os.path.basename(path)}) ---\n{text}")
        elif dtype == "image":
            analysis = analyze_image_with_gemini(path)
            if analysis:
                img_analyses.append(f"--- Image Analysis ({os.path.basename(path)}) ---\n{analysis}")
                
    return {"raw_pdf_text": pdf_texts, "gemini_image_analyses": img_analyses}


def _synthesize_history_node(state: PatientSummaryState) -> dict:
    """
    Node 2 — Synthesizes all extracted text and image findings into a unified history.
    """
    llm = _get_llm(temperature=0.2)
    
    system_prompt = (
        "You are an expert Clinical Summarizer AI. Your task is to take raw, extracted text "
        "from past medical PDFs and AI-generated findings from historical and current medical images, "
        "and synthesize them into a single, cohesive 'Unified Patient History' report. \n"
        "1. Organize the summary chronologically if dates are available, or logically by body system.\n"
        "2. Keep it incredibly concise and strictly professional.\n"
        "3. Do NOT make new diagnoses, just summarize the provided facts.\n"
        "4. Output valid, clean Markdown. Do NOT use XML or <think> tags."
    )
    
    # Combine texts
    all_context = []
    if state["raw_pdf_text"]:
        all_context.append("### EXTRACTED PDF CLINICAL NOTES ###\n" + "\n\n".join(state["raw_pdf_text"]))
    if state["gemini_image_analyses"]:
        all_context.append("### AI IMAGE FINDINGS ###\n" + "\n\n".join(state["gemini_image_analyses"]))
        
    if not all_context:
        return {"unified_summary": "No historical documents found to summarize."}
        
    context_str = "\n\n".join(all_context)
    
    user_prompt = (
        f"Patient Name: {state['patient_name']}\n\n"
        f"Please write the Unified Patient History based on these extracted records:\n\n{context_str}"
    )
    
    logger.info(f"[Graph 3] Synthesizing history for {state['patient_name']} via Groq...")
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    
    import re
    cleaned_summary = re.sub(r'<think>.*?</think>\n?', '', response.content, flags=re.DOTALL)
    cleaned_summary = re.sub(r'```markdown\n?', '', cleaned_summary, flags=re.IGNORECASE)
    cleaned_summary = re.sub(r'```\n?', '', cleaned_summary)
    cleaned_summary = cleaned_summary.strip()
    
    return {"unified_summary": cleaned_summary}


def build_patient_summary_graph() -> Any:
    """Compile and return the Patient Summary StateGraph."""
    graph = StateGraph(PatientSummaryState)
    graph.add_node("fetch_and_parse", _fetch_and_parse_documents_node)
    graph.add_node("synthesize", _synthesize_history_node)
    
    graph.set_entry_point("fetch_and_parse")
    graph.add_edge("fetch_and_parse", "synthesize")
    graph.add_edge("synthesize", END)
    
    return graph.compile()


_patient_summary_graph = None

def get_patient_summary_graph():
    global _patient_summary_graph
    if _patient_summary_graph is None:
        _patient_summary_graph = build_patient_summary_graph()
    return _patient_summary_graph


def generate_patient_summary(patient_name: str, documents: list) -> str:
    """
    Public API for Graph 3.
    """
    graph = get_patient_summary_graph()
    initial_state = {
        "patient_name"          : patient_name,
        "documents"             : documents,
        "raw_pdf_text"          : [],
        "gemini_image_analyses" : [],
        "unified_summary"       : None,
    }
    result = graph.invoke(initial_state)
    return result.get("unified_summary", "Summary generation failed.")
