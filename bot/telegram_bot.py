"""
telegram_bot.py — LungCare Triage Patient Bot
===============================================
Patient-facing Telegram bot using python-telegram-bot v20+ (async).

Commands:
    /start  — Greet the patient and link their Telegram account to the DB
    /help   — Show available commands
    /status — Quick summary of the patient's latest scan status

Message Handler:
    Any free-text message is forwarded to POST /api/patient_chat on the
    FastAPI backend, which runs LangGraph Graph 2 and returns an empathetic,
    safe explanation.

Environment Variables Required (.env):
    TELEGRAM_BOT_TOKEN : Bot token from @BotFather
    FASTAPI_BASE_URL   : Base URL of FastAPI app (default: http://localhost:8000)
    DEFAULT_PATIENT_ID : Demo patient ID to link on /start (default: 1)
"""

import asyncio
import logging
import os

import httpx
from dotenv import load_dotenv, find_dotenv
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ─── Config ───────────────────────────────────────────────────────────────────
load_dotenv(find_dotenv())

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
FASTAPI_BASE_URL   = os.getenv("FASTAPI_BASE_URL", "http://localhost:8000")
DEFAULT_PATIENT_ID = int(os.getenv("DEFAULT_PATIENT_ID", "1"))

logging.basicConfig(
    format  = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level   = logging.INFO,
)
logger = logging.getLogger(__name__)


# ─── Helper: Call FastAPI ──────────────────────────────────────────────────────

async def call_fastapi_post(endpoint: str, payload: dict) -> dict | None:
    """Make an async POST request to the FastAPI backend."""
    url = f"{FASTAPI_BASE_URL}{endpoint}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError:
            logger.error(f"[Bot] Cannot connect to FastAPI at {url}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"[Bot] HTTP error {e.response.status_code}: {e.response.text}")
            return None


async def call_fastapi_get(endpoint: str) -> dict | None:
    """Make an async GET request to the FastAPI backend."""
    url = f"{FASTAPI_BASE_URL}{endpoint}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"[Bot] GET error: {e}")
            return None


async def register_patient(telegram_chat_id: str, patient_id: int) -> bool:
    """Link the Telegram chat ID to a patient record in the DB."""
    url = f"{FASTAPI_BASE_URL}/api/bot/register"
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            r = await client.post(url, params={
                "telegram_chat_id": telegram_chat_id,
                "patient_id"      : patient_id,
            })
            r.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"[Bot] Registration failed: {e}")
            return False


# ─── Command Handlers ─────────────────────────────────────────────────────────

# In-memory conversational history for the demo (in production use a DB for this too)
user_chat_history = {}

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /start — Welcome message and patient registration.

    In a real deployment, patients would enter their Patient ID securely.
    For the demo, we securely lock them to DEFAULT_PATIENT_ID.
    """
    chat_id  = str(update.effective_chat.id)
    username = update.effective_user.first_name or "there"

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    # Clean history on start
    user_chat_history[chat_id] = []

    # Attempt to link this Telegram account to a patient DB record
    success = await register_patient(chat_id, DEFAULT_PATIENT_ID)

    if success:
        welcome_msg = (
            f"👋 Hello, *{username}*! Welcome to **LungCare Triage**.\n\n"
            "🫁 I'm your personal lung health companion. Your account has been "
            "linked securely to your medical record.\n\n"
            "You can talk to me naturally. Try asking:\n"
            "• _What does my risk score mean?_\n"
            "• _When is my next follow-up?_\n"
            "• _Is my nodule getting bigger?_\n\n"
            "I'll remember our conversation, but remember — "
            "**always consult your doctor** for medical decisions. 💙\n"
        )
    else:
        welcome_msg = (
            f"👋 Hello, *{username}*! Welcome to **LungCare Triage**.\n\n"
            "⚠️ I couldn't link your account right now, but you can still "
            "ask me general questions about lung health.\n\n"
            "Type /help for more information."
        )

    await update.message.reply_text(welcome_msg, parse_mode=ParseMode.MARKDOWN)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/help — Show available commands."""
    help_text = (
        "🫁 *LungCare Triage Bot — Help*\n\n"
        "Here's what I can do:\n\n"
        "*/start* — Register your account and get started\n"
        "*/status* — Check your latest scan status\n"
        "*/help* — Show this help message\n\n"
        "💬 Or just *type any question* about your lung health report and "
        "I'll give you a clear, safe explanation.\n\n"
        "_Examples:_\n"
        "• _What does High risk mean for me?_\n"
        "• _How big is my nodule?_\n"
        "• _What happens if my nodule grows?_\n\n"
        "⚕️ *Reminder:* I provide information only. Always speak to your "
        "care team for medical advice. 💙"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/status — Show the patient's latest scan status."""
    chat_id = str(update.effective_chat.id)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    # Find patient by chat_id via the chat endpoint (quick check)
    result = await call_fastapi_post("/api/patient_chat", {
        "telegram_chat_id": chat_id,
        "message"         : "Can you give me a brief summary of my latest scan status?",
    })

    if result:
        patient_name = result.get("patient_name", "Patient")
        response     = result.get("response", "No status information available.")
        await update.message.reply_text(
            f"📊 *Status for {patient_name}*\n\n{response}",
            parse_mode=ParseMode.MARKDOWN,
        )
    else:
        await update.message.reply_text(
            "❌ I couldn't retrieve your status right now. "
            "Please try again or contact the clinic directly.",
        )


# ─── Message Handler ──────────────────────────────────────────────────────────

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle any plain-text message from the patient.

    1. Show typing indicator for better UX.
    2. Forward the question to POST /api/patient_chat.
    3. Return the LangGraph-generated safe response.
    """
    chat_id  = str(update.effective_chat.id)
    question = update.message.text.strip()

    if not question:
        return

    logger.info(f"[Bot] Message from chat_id={chat_id}: {question[:80]}...")

    # Show typing action (async, so it runs while we wait for the LLM)
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    # Get recent history (max 6 messages to save Groq context limits)
    history = user_chat_history.get(chat_id, [])[-6:]
    
    # Call FastAPI → LangGraph Patient Support Agent
    result = await call_fastapi_post("/api/patient_chat", {
        "telegram_chat_id": chat_id,
        "message"         : question,
        "chat_history"    : history,
    })

    if result:
        response_text = result.get("response", "I'm sorry, I couldn't process your question.")
        patient_name  = result.get("patient_name")
        
        # Save to history
        if chat_id not in user_chat_history:
            user_chat_history[chat_id] = []
        user_chat_history[chat_id].append({"role": "user", "content": question})
        user_chat_history[chat_id].append({"role": "assistant", "content": response_text})

        # Add a personalised greeting prefix if it's the very first message
        if patient_name and len(user_chat_history[chat_id]) <= 2:
            prefix = f"Hi {patient_name.split()[0]}! 👋\n\n"
        else:
            prefix = ""

        await update.message.reply_text(
            f"{prefix}{response_text}",
            parse_mode=ParseMode.MARKDOWN,
        )
    else:
        await update.message.reply_text(
            "😔 I'm having trouble connecting to the medical system right now.\n\n"
            "Please try again in a moment, or contact your clinic directly if this is urgent.",
        )


# ─── Error Handler ────────────────────────────────────────────────────────────

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors raised by telegram-bot handlers."""
    logger.error(f"[Bot] Unhandled error: {context.error}", exc_info=context.error)

    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text(
            "⚠️ Something went wrong on my end. Please try again shortly."
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point — build and start the bot with polling."""
    if not TELEGRAM_BOT_TOKEN:
        raise EnvironmentError(
            "TELEGRAM_BOT_TOKEN is not set!\n"
            "1. Message @BotFather on Telegram to create a bot.\n"
            "2. Add the token to your .env file as TELEGRAM_BOT_TOKEN=..."
        )

    logger.info("[Bot] Starting LungCare Telegram bot...")

    # Build the Application
    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    # Register handlers
    application.add_handler(CommandHandler("start",  start_command))
    application.add_handler(CommandHandler("help",   help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
    application.add_error_handler(error_handler)

    logger.info("[Bot] Bot is running. Press Ctrl+C to stop.")

    # Start polling (long-polling for development; switch to webhooks for production)
    application.run_polling(
        allowed_updates = ["message"],
        drop_pending_updates = True,
    )


if __name__ == "__main__":
    main()
