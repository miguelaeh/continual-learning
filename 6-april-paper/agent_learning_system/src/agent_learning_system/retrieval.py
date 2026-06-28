"""Simple retrieval over interaction traces."""

from __future__ import annotations

import re
from collections import Counter


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def retrieve_relevant_traces(
    query: str,
    traces: list[dict],
    limit: int = 5,
) -> list[dict]:
    query_counts = Counter(_tokenize(query))
    if not query_counts:
        return traces[-limit:]

    scored: list[tuple[int, int, dict]] = []
    for idx, trace in enumerate(traces):
        haystack = " ".join(
            [
                str(trace.get("user_message", "")),
                str(trace.get("assistant_response", "")),
                str(trace.get("summary", "")),
                " ".join(str(tag) for tag in trace.get("tags", [])),
            ]
        )
        doc_counts = Counter(_tokenize(haystack))
        overlap = sum(min(query_counts[token], doc_counts[token]) for token in query_counts)
        if overlap > 0:
            scored.append((overlap, idx, trace))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[:limit]]
