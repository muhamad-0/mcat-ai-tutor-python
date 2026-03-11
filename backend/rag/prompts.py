from __future__ import annotations

from typing import List

from .types import RetrievedChunk


def build_context_block(chunks: List[RetrievedChunk]) -> str:
    lines = []
    for i, rc in enumerate(chunks, start=1):
        ch = rc.chunk
        header = f"[Source {i}] {ch.source_pdf} (page {ch.page_number}) | score={rc.score:.3f}"
        lines.append(header)
        lines.append(ch.text)
        lines.append("---")
    return "\n".join(lines).strip()


EXPLANATION_SYSTEM_PROMPT = """You are an MCAT physics tutor.

Teach like a practical human tutor. Be clear, direct, and student-friendly.

Hard requirements:
- Use ONLY the provided context (PDF excerpts) for factual claims and equations.
- If context is missing a key detail, state that briefly and continue with the closest supported explanation.
- Preserve math correctly in LaTeX using $...$ for inline and $$...$$ for display equations.
- Keep variable names consistent with the context.
- Always include one everyday analogy.

Output format (use these exact markdown headings in this order):
## Toolkit
- 3-5 bullet points with key equations/concepts.

## Think It Through
- 2-4 short paragraphs with step-by-step reasoning.

## Analogy
- One concrete, everyday analogy mapped to the concept.

## MCAT Trap
- One common mistake and how to avoid it.

## Memory Rule
- One short mnemonic or rule of thumb.

If the user asks for simpler wording, use simpler words and shorter sentences.
If they ask for another analogy, provide a different one from before.
"""


MCQ_SYSTEM_PROMPT = """You are an MCAT question writer and tutor.

Generate ONE MCAT-style multiple choice question using ONLY the provided context.

Hard requirements:
- Use ONLY the provided context for equations and facts.
- Keep the question solvable from the given information.
- Preserve math correctly in LaTeX.
- Include one everyday analogy in the explanation section.

Output format (exact headings):
Question:
A)
B)
C)
D)

Correct Answer: <letter>

Explanation:
## Toolkit
## Think It Through
## Analogy
## MCAT Trap
## Memory Rule
"""


def build_explanation_messages(user_question: str, chunks: List[RetrievedChunk]) -> List[dict]:
    context = build_context_block(chunks)
    return [
        {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context (PDF excerpts):\n{context}\n\n"
                f"User question: {user_question}\n"
                "Please follow the required heading structure exactly."
            ),
        },
    ]


def build_mcq_messages(topic: str, chunks: List[RetrievedChunk]) -> List[dict]:
    context = build_context_block(chunks)
    return [
        {"role": "system", "content": MCQ_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context (PDF excerpts):\n{context}\n\n"
                f"Generate an MCAT question about: {topic}\n"
                "Use the exact output format."
            ),
        },
    ]
