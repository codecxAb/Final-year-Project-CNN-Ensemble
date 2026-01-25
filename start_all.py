"""
start_all.py — LungCare Triage Multi-Service Runner
===================================================
A single script to start the FastAPI backend, Streamlit frontend,
and Telegram bot simultaneously.

Features:
    - Runs all 3 processes in parallel using subprocess.
    - Captures stdout/stderr from all processes.
    - Pipes output to both the console (with coloured prefixes) 
      and to dedicated log files in the `logs/` directory.
    - Graceful shutdown of all child processes on Ctrl+C.

Usage:
    python start_all.py
"""

import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

# ─── Resolve Python Executable ────────────────────────────────────────────────
# Prefer .venv/bin/python so all services use the correct environment
_SCRIPT_DIR = Path(__file__).parent
_VENV_PYTHON = _SCRIPT_DIR / ".venv" / "bin" / "python"
PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable

# ─── Config ───────────────────────────────────────────────────────────────────

# Define the 3 services
SERVICES = {
    "Backend": {
        "command": [PYTHON, "-m", "uvicorn", "main:app", "--port", "8000"],
        "cwd": "backend",
        "color": "\033[94m",  # Blue
    },
    "Frontend": {
        "command": [PYTHON, "-m", "streamlit", "run", "app.py", "--server.port", "8501"],
        "cwd": "frontend-streamlit",
        "color": "\033[95m",  # Magenta
    },
    "Bot": {
        "command": [PYTHON, "telegram_bot.py"],
        "cwd": "bot",
        "color": "\033[92m",  # Green
    },
}

RESET_COLOR = "\033[0m"

# ─── Setup Logging Directory ──────────────────────────────────────────────────

os.makedirs("logs", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def print_prefix(name: str, color: str, line: str, log_file):
    """Print to console with a coloured prefix and write to a log file."""
    line = line.rstrip() # remove trailing newlines
    if line:
        # Console output
        print(f"{color}[{name}]{RESET_COLOR} {line}")
        # File output (no colour codes)
        timestamped_line = f"{datetime.now().strftime('%H:%M:%S')} | {line}\n"
        log_file.write(timestamped_line)
        log_file.flush()


def stream_reader(pipe, name: str, color: str, log_file):
    """Reads lines from a subprocess pipe and routes them to print_prefix."""
    with pipe:
        for line in iter(pipe.readline, ""):
            print_prefix(name, color, line, log_file)


# ─── Main Runner ──────────────────────────────────────────────────────────────

def main():
    print("🚀 Starting LungCare Triage System...")
    print(f"📂 Logs are being written to the ./logs/ directory\n")

    processes = []
    threads = []
    log_files = []

    try:
        # Start all services
        for name, config in SERVICES.items():
            log_path = f"logs/{name.lower()}_{timestamp}.log"
            log_file = open(log_path, "w", encoding="utf-8")
            log_files.append(log_file)

            print_prefix("SYSTEM", "\033[96m", f"Starting {name} (Logs: {log_path})", log_file)

            # Start the subprocess
            # We use text=True, bufsize=1 for line-buffered text output
            process = subprocess.Popen(
                config["command"],
                cwd=config["cwd"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Merge stderr into stdout
                text=True,
                bufsize=1,
            )
            processes.append(process)

            # Start a thread to read the output stream without blocking
            t = threading.Thread(
                target=stream_reader,
                args=(process.stdout, name, config["color"], log_file),
                daemon=True
            )
            t.start()
            threads.append(t)

        print_prefix("SYSTEM", "\033[96m", "All services started. Press Ctrl+C to stop.", log_file)

        # Wait for all processes (they run indefinitely)
        for p in processes:
            p.wait()

    except KeyboardInterrupt:
        print("\n🛑 Shutting down all services...")
        
        # Terminate all child processes
        for p in processes:
            if p.poll() is None: # If still running
                p.terminate()
        
        # Give them a moment to terminate gracefully, then kill
        for p in processes:
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()

        print("✅ Shutdown complete.")

    finally:
        # Clean up log file handles
        for f in log_files:
            if not f.closed:
                f.close()

if __name__ == "__main__":
    main()
