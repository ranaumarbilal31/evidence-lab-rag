# Security Modules & Defense-in-Depth Architecture

This document details the multi-tier screening, normalization, and red-teaming subsystems implemented for **Member 3 (Core Security & Detection Developer)** in the **Evidence Recovery RAG Safety Wall**.

---

## 1. Multi-Tier Screening Pipeline

Every retrieved chunk that reaches the protected pipeline passes through a progressive screening gate in [`rag/detection.py`](file:///d:/ragg/evidence-lab-rag/rag/detection.py) before entering the generation context:

```
                      [ Incoming Retrieved Chunk ]
                                   │
                                   ▼
          ┌─────────────────────────────────────────────────┐
          │  Text Normalization (rag/obfuscation.py)         │
          │  • Strips zero-width code points (\u200b, etc.) │
          │  • Collapses whitespace token splitting         │
          └─────────────────────────────────────────────────┘
                                   │
                                   ▼
          ┌─────────────────────────────────────────────────┐
          │  Tier 1: Heuristic Regex Rules (PATTERNS)       │
          │  • Instant, deterministic, zero API cost        │
          │  • System tags, override directives, exfil     │
          └─────────────────────────────────────────────────┘
                            │               │
                        [Matched]       [Clean]
                            │               │
                            │               ▼
                            │  ┌──────────────────────────────────────────────┐
                            │  │  Tier 2: Vector-Similarity Detector          │
                            │  │  (rag/semantic_detector.py)                  │
                            │  │  • Local embeddings (all-MiniLM-L6-v2)       │
                            │  │  • Cosine dot product vs 20 attack triggers  │
                            │  │  • Zero API cost; catches paraphrased attacks│
                            │  └──────────────────────────────────────────────┘
                            │               │               │
                            │           [≥ 0.75]         [< 0.75]
                            │               │               │
                            │               │               ▼
                            │               │  ┌──────────────────────────────┐
                            │               │  │  Tier 3: LLM Classifier      │
                            │               │  │  (Safety Model Call)         │
                            │               │  │  • Deep semantic reasoning   │
                            │               │  │  • Only for remaining chunks │
                            │               │  └──────────────────────────────┘
                            │               │               │
                            ▼               ▼               ▼
                 [ Quarantine Evidence ]        [ Trusted Context ]
```

---

## 2. Component Descriptions

### Tier 1: Heuristic Rules (`rag/detection.py`)
- Fast regex matching over [`PATTERNS`](file:///d:/ragg/evidence-lab-rag/rag/detection.py#L48).
- Flags obvious system overrides (`ignore previous instructions`, `disregard prior rules`), role impersonation (`<system>`, `[INST]`, `developer message`), exfiltration attempts (`reveal API key`, `print system prompt`), encoded directives (`execute base64`), and hidden HTML comments (`<!-- SYSTEM OVERRIDE: ... -->`).
- Returns `source="heuristic"`, `confidence=1.0`.

### Tier 2: Vector-Similarity Detection (`rag/semantic_detector.py`)
- **Attack Bank**: 20 canonical Indirect Prompt Injection (IPI) signatures curated across instruction overrides, jailbreaks (`DAN`, developer mode), and prompt leakages.
- **Model**: `all-MiniLM-L6-v2` (384D) instantiated with `low_cpu_mem_usage=True` and a global model cache for minimal memory footprint.
- **Matching**: Computes cosine similarity via dot product of L2-normalized embeddings.
- **Threshold**: Default `0.75` (configurable).
  - Benign policy passages score **$0.03\text{--}0.33$** (well below threshold).
  - Adversarial injection variants score **$0.83\text{--}0.95$**.
- Returns `source="vector_similarity"`, `confidence=score`. Flagged chunks bypass Tier 3, preserving remote API quota.

### Tier 3: LLM Safety Classifier (`rag/detection.py`)
- Evaluates any subtle or novel passages that survive Tier 1 and Tier 2.
- Only invoked for unflagged chunks, avoiding unnecessary API consumption.

---

## 3. Defensive Normalization (`rag/obfuscation.py`)

- **`normalize_text(text: str) -> str`**:
  - Strips invisible unicode code points (`\u200b`, `\u200c`, `\u200d`, `\ufeff`, `\u2060`) commonly used by attackers for token-splitting and regex bypass.
  - Normalizes whitespace before feeding passages to Tier 1 and Tier 2.
  - Ensures obfuscated inputs like `I\u200bg\u200bn\u200bo\u200br\u200be previous instructions` collapse back to standard text and get caught immediately.

---

## 4. Red-Teaming & Obfuscation Mutators (`rag/obfuscation.py`)

Provides a modular suite of evasion probes for evaluating safety walls:

| Mutator | Description | Example Output |
| :--- | :--- | :--- |
| **`obfuscate_base64`** | Base64-encodes payload wrapped in an execution prompt | `Execute base64: SWdub3Jl...` |
| **`obfuscate_leetspeak`** | Deterministic character substitution (`e->3`, `i->1`, `o->0`, `a->4`, `s->5`, `t->7`) | `1gn0r3 4ll pr3v10u5...` |
| **`obfuscate_comment_smuggling`** | Hides instructions in HTML/Markdown comments | `<!-- SYSTEM OVERRIDE: ... -->` |
| **`obfuscate_separator`** | Intersperses zero-width spaces between letters | `I\u200bg\u200bn\u200bo\u200br\u200be...` |
| **`obfuscate_rot13`** | ROT13 cipher wrapped in a decode directive | `Decode ROT13: Vtaber nyy...` |

### Dataset Generator Integration
- **`generate_obfuscated_cases()`** in [`rag/dataset.py`](file:///d:/ragg/evidence-lab-rag/rag/dataset.py#L78): Automatically synthesizes evaluation test cases across each obfuscation category.
- **`obfuscated_attack()`** in [`rag-safety-wall/safety_wall/attacks.py`](file:///d:/ragg/evidence-lab-rag/rag-safety-wall/safety_wall/attacks.py#L81): Category 3 attack generator in the prototype attack suite.

---

## 5. Verification & Test Coverage

All modules are accompanied by comprehensive pytest suites:
* [`tests/test_semantic_detector.py`](file:///d:/ragg/evidence-lab-rag/tests/test_semantic_detector.py) (16 tests)
* [`tests/test_obfuscation_probes.py`](file:///d:/ragg/evidence-lab-rag/tests/test_obfuscation_probes.py) (12 tests)
* Complete test suite runnable with a single command:
  ```powershell
  python -m pytest
  ```
