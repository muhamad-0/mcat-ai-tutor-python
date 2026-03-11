from __future__ import annotations

import re


_GREEK_MAP = {
    "ρ": "rho",
    "Ρ": "rho",
    "μ": "mu",
    "Μ": "mu",
    "Δ": "Delta",
    "δ": "delta",
    "λ": "lambda",
    "Λ": "lambda",
    "π": "pi",
    "Π": "pi",
    "θ": "theta",
    "Θ": "theta",
    "ω": "omega",
    "Ω": "omega",
    "η": "eta",
    "γ": "gamma",
    "α": "alpha",
    "β": "beta",
    "σ": "sigma",
    "Σ": "sigma",
}

_SUPERSCRIPTS = {
    "⁰": "^0",
    "¹": "^1",
    "²": "^2",
    "³": "^3",
    "⁴": "^4",
    "⁵": "^5",
    "⁶": "^6",
    "⁷": "^7",
    "⁸": "^8",
    "⁹": "^9",
}

_SUBSCRIPTS = {
    "₀": "_0",
    "₁": "_1",
    "₂": "_2",
    "₃": "_3",
    "₄": "_4",
    "₅": "_5",
    "₆": "_6",
    "₇": "_7",
    "₈": "_8",
    "₉": "_9",
}


def normalize_math_text(text: str) -> str:
    """Normalize OCR text to keep equations stable for retrieval & prompting.

    Goal: keep content readable and consistent, not perfect LaTeX.
    """
    if not text:
        return ""

    # Fix hyphenation across line breaks: "incom-\npressible" -> "incompressible"
    text = re.sub(r"([A-Za-z])\-\n([A-Za-z])", r"\1\2", text)

    # Convert common unicode to ASCII-ish equivalents.
    for k, v in _GREEK_MAP.items():
        text = text.replace(k, v)
    for k, v in _SUPERSCRIPTS.items():
        text = text.replace(k, v)
    for k, v in _SUBSCRIPTS.items():
        text = text.replace(k, v)

    text = text.replace("×", "*")
    text = text.replace("·", "*")
    text = text.replace("÷", "/")

    # Standardize weird quotes
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")

    # Collapse extra spaces but keep newlines
    text = re.sub(r"[ \t]+", " ", text)

    # Clean up repeated blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Trim whitespace on each line
    text = "\n".join([ln.strip() for ln in text.splitlines()])

    return text.strip()


def looks_like_math_line(line: str) -> bool:
    """Heuristic: detect lines that are likely to contain equations.

    We use this to avoid splitting chunks mid-equation.
    """
    if not line:
        return False
    # Common equation tokens
    tokens = ["=", "≈", "+", "-", "*", "/", "^", "rho", "mu", "Delta", "pi", "sqrt", "g", "P", "v", "A"]
    if any(t in line for t in tokens):
        # If it has at least one digit OR a greek word, it's probably math-ish
        if re.search(r"\d", line) or re.search(r"\b(rho|mu|Delta|pi|theta|sigma)\b", line):
            return True
        # Or it contains multiple operators
        if sum(1 for t in ["=", "+", "-", "*", "/"] if t in line) >= 2:
            return True
    return False
