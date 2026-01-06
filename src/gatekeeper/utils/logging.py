import sys
import json
import re
from typing import Any, Optional

# ANSI colors
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"

# Regex to strip ANSI codes for file logging
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

class Logger:
    def __init__(self, verbose: bool = False, log_file: Optional[str] = None):
        self.verbose = verbose
        self.log_file = log_file
        self._file_handle = None
        
        if self.log_file:
            # Open file in append or write mode? Let's use write mode to start fresh for each run
            try:
                self._file_handle = open(self.log_file, "w", encoding="utf-8")
            except Exception as e:
                sys.stderr.write(f"Warning: Could not open log file {self.log_file}: {e}\n")

    def _strip_ansi(self, text: str) -> str:
        return ANSI_ESCAPE.sub('', text)

    def _write(self, text: str):
        # Console output (with colors)
        if self.verbose:
            sys.stderr.write(text + "\n")
            
        # File output (without colors)
        if self._file_handle:
            clean_text = self._strip_ansi(text)
            self._file_handle.write(clean_text + "\n")
            self._file_handle.flush()

    def close(self):
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None

    def header(self, title: str):
        # Always log header to file even if not verbose
        # But our _write checks verbose for console.
        # Logic: If verbose=False, we don't print to console, but we might want to log to file?
        # Requirement says "--log <log_path> 讓我們的log能被記錄下來".
        # It implies logging should happen regardless of --verbose if --log is present.
        # Let's adjust _write logic slightly or just call it.
        
        sep = f"{BOLD}{'='*60}{RESET}"
        self._write(f"\n{sep}")
        self._write(f" {BOLD}{title}{RESET}")
        self._write(f"{sep}")

    def info(self, msg: str):
        self._write(msg)

    def success(self, msg: str):
        self._write(f"{GREEN}{msg}{RESET}")

    def warning(self, msg: str):
        self._write(f"{YELLOW}{msg}{RESET}")

    def error(self, msg: str):
        self._write(f"{RED}{msg}{RESET}")
        
    def section(self, title: str):
        self._write(f"\n{BOLD}>> {title}{RESET}")

    def json(self, data: Any):
        # JSON dump is large, only print if needed.
        # If logging to file, we probably want it there too.
        text = json.dumps(data, indent=2, ensure_ascii=False)
        self._write(text)

    def vote(self, agent: str, vote: str, reason: str, mods: list = None):
        color = GREEN
        if vote == "MODIFY": color = YELLOW
        elif vote == "REJECT": color = RED
        
        icon = f"{color}{vote:<8}{RESET}"
        self._write(f"{icon} [{BOLD}{agent}{RESET}]")
        self._write(f"   Reason: {reason}")
        
        for m in mods or []:
            self._write(f"   -> Suggestion: {m.get('field')} = {m.get('proposed_value')} ({m.get('priority')})")
        self._write("")

    def debate(self, agent: str, stance: str, targets: list, message: str):
        icon = f"{BLUE}DEFEND{RESET}" if stance == "DEFEND" else f"{RED}ATTACK{RESET}"
        tgt_str = ", ".join(targets)
        self._write(f"{icon:<8} [{BOLD}{agent}{RESET}] -> [{tgt_str}]")
        self._write(f"   \"{message}\"\n")
