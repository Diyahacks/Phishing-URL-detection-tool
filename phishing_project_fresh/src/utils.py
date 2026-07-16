import os
import sys

# Reconfigure stdout to use UTF-8 to prevent UnicodeEncodeError on Windows command prompt
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from tabulate import tabulate

# ANSI Escape Sequences for Terminal Colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'

def clear_screen():
    """Clears the console screen based on OS."""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_banner() -> str:
    """Returns a highly professional ASCII banner for the ME project."""
    # Box dimensions
    # Logo text width is 86. Let's add 4 spaces on each side.
    # Total inner width is 4 + 86 + 4 = 94.
    
    logo_lines = [
        "██████╗ ██╗  ██╗██╗███████╗██╗  ██╗██╗███╗   ██╗ ██████╗     ██╗   ██╗██████╗ ██╗     ",
        "██╔══██╗██║  ██║██║██╔════╝██║  ██║██║████╗  ██║██╔════╝     ██║   ██║██╔══██╗██║     ",
        "██████╔╝███████║██║███████╗███████║██║██╔██╗ ██║██║  ███╗    ██║   ██║██████╔╝██║     ",
        "██╔═══╝ ██╔══██║██║╚════██║██╔══██║██║██║╚██╗██║██║   ██║    ██║   ██║██╔══██╗██║     ",
        "██║     ██║  ██║██║███████║██║  ██║██║██║ ╚████║╚██████╔╝    ╚██████╔╝██║  ██║███████╗",
        "╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝      ╚═════╝ ╚═╝  ╚═╝╚══════╝"
    ]
    
    # Subtitles
    subtitle1 = "PHISHING URL DETECTION SYSTEM USING MACHINE LEARNING"
    subtitle2 = "Master of Technology (M.Tech)"
    
    # Center subtitles inside 94 characters
    sub1_padded = subtitle1.center(94)
    sub2_padded = subtitle2.center(94)
    
    # Border characters
    top_border = "╔" + "═" * 94 + "╗"
    bottom_border = "╚" + "═" * 94 + "╝"
    empty_line = "║" + " " * 94 + "║"
    
    # Build banner
    banner_lines = [
        f"{Colors.CYAN}{Colors.BOLD}{top_border}",
        f"{Colors.CYAN}{Colors.BOLD}{empty_line}",
    ]
    
    # Add logo lines
    for line in logo_lines:
        banner_lines.append(f"{Colors.CYAN}{Colors.BOLD}║    {Colors.YELLOW}{line}{Colors.CYAN}{Colors.BOLD}    ║")
        
    banner_lines.extend([
        f"{Colors.CYAN}{Colors.BOLD}{empty_line}",
        f"{Colors.CYAN}{Colors.BOLD}║{Colors.GREEN}{Colors.BOLD}{sub1_padded}{Colors.CYAN}{Colors.BOLD}║",
        f"{Colors.CYAN}{Colors.BOLD}║{Colors.BLUE}{sub2_padded}{Colors.CYAN}{Colors.BOLD}║",
        f"{Colors.CYAN}{Colors.BOLD}{bottom_border}{Colors.RESET}"
    ])
    
    return "\n" + "\n".join(banner_lines) + "\n"

def print_banner():
    """Prints the banner to the console."""
    print(get_banner())

def print_colored(text: str, color: str):
    """Prints text in a specific ANSI color."""
    print(f"{color}{text}{Colors.RESET}")

def print_success(text: str):
    """Prints a green success message."""
    print(f"{Colors.GREEN}{Colors.BOLD}[✓] SUCCESS: {text}{Colors.RESET}")

def print_error(text: str):
    """Prints a red error message."""
    print(f"{Colors.RED}{Colors.BOLD}[✗] ERROR: {text}{Colors.RESET}")

def print_warning(text: str):
    """Prints a yellow warning message."""
    print(f"{Colors.YELLOW}{Colors.BOLD}[!] WARNING: {text}{Colors.RESET}")

def print_info(text: str):
    """Prints a blue informational message."""
    print(f"{Colors.BLUE}[i] {text}{Colors.RESET}")

def print_header(text: str):
    """Prints a styled subsection header."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}=== {text.upper()} ==={Colors.RESET}\n")

def print_table(headers: list, rows: list, table_fmt: str = "grid"):
    """
    Prints a beautiful table using the tabulate library.
    Supported formats include 'grid', 'fancy_grid', 'pipe', etc.
    """
    print(tabulate(rows, headers=headers, tablefmt=table_fmt))
