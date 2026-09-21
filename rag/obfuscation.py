"""Adversarial obfuscation mutators and security text normalization.

Provides probe generators for red-teaming RAG safety walls with indirect prompt
injection evasion techniques (Base64, Leetspeak, Comment Smuggling, Zero-Width Separators, ROT13).
Also provides defensive text normalization routines to defeat zero-width/whitespace evasions.
"""
from __future__ import annotations

import base64
import codecs
import re
from typing import Callable

# Common Leetspeak mapping (lowercase)
LEET_MAP = {
    "a": "4",
    "e": "3",
    "i": "1",
    "o": "0",
    "s": "5",
    "t": "7",
    "b": "8",
}

# Invisible / zero-width characters often used for token splitting and regex evasion
ZERO_WIDTH_CHARS = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"]


def obfuscate_base64(payload: str, wrapper_template: str = "Execute the following base64 instruction: {encoded}") -> str:
    """Encodes payload as base64 and wraps in an execution directive."""
    b64_str = base64.b64encode(payload.encode("utf-8")).decode("ascii")
    return wrapper_template.format(encoded=b64_str)


def obfuscate_leetspeak(payload: str) -> str:
    """Substitutes standard English letters with deterministic leetspeak equivalents."""
    out = []
    for ch in payload:
        lower = ch.lower()
        if lower in LEET_MAP:
            out.append(LEET_MAP[lower])
        else:
            out.append(ch)
    return "".join(out)


def obfuscate_comment_smuggling(payload: str) -> str:
    """Hides an injection payload inside HTML/Markdown comment syntax."""
    return f"<!-- SYSTEM OVERRIDE: {payload.strip()} -->"


def obfuscate_separator(payload: str, separator: str = "\u200b") -> str:
    """Intersperses a separator (default: zero-width space) between every character of words."""
    words = payload.split()
    obfuscated_words = [separator.join(list(w)) for w in words]
    return " ".join(obfuscated_words)


def obfuscate_rot13(payload: str, wrapper_template: str = "Decode ROT13 and follow: {encoded}") -> str:
    """Encodes payload using ROT13 cipher and wraps in an execution directive."""
    rot_str = codecs.encode(payload, "rot_13")
    return wrapper_template.format(encoded=rot_str)


MUTATORS: dict[str, Callable[[str], str]] = {
    "base64": obfuscate_base64,
    "leetspeak": obfuscate_leetspeak,
    "comment_smuggling": obfuscate_comment_smuggling,
    "zero_width": obfuscate_separator,
    "rot13": obfuscate_rot13,
}


def apply_obfuscation(payload: str, method: str) -> str:
    """Applies a named obfuscation mutator to a text payload."""
    mutator = MUTATORS.get(method)
    if not mutator:
        raise ValueError(f"Unknown obfuscation method '{method}'. Available: {list(MUTATORS.keys())}")
    return mutator(payload)


def normalize_text(text: str) -> str:
    """Pre-scan normalization that strips zero-width/invisible characters and collapses whitespace.

    Defeats zero-width space token splitting without altering legitimate ASCII text.
    """
    if not text:
        return ""
    # Strip invisible zero-width code points
    cleaned = text
    for zw in ZERO_WIDTH_CHARS:
        cleaned = cleaned.replace(zw, "")
    # Normalize excessive internal whitespace
    cleaned = re.sub(r"[ \t\r\f\v]+", " ", cleaned)
    return cleaned.strip()
