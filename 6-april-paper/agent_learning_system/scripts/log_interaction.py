#!/usr/bin/env python3
"""Append an interaction trace to the episodic store."""

from __future__ import annotations

import argparse

from agent_learning_system.store import EpisodicStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-dir", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--user-message", required=True)
    parser.add_argument("--assistant-response", required=True)
    parser.add_argument("--outcome", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--learned-type", default="general")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--correction", default=None)
    args = parser.parse_args()

    store = EpisodicStore(args.store_dir)
    trace = store.append_interaction(
        session_id=args.session_id,
        user_message=args.user_message,
        assistant_response=args.assistant_response,
        outcome=args.outcome,
        summary=args.summary,
        learned_type=args.learned_type,
        tags=args.tag,
        correction=args.correction,
    )
    print(trace.trace_id)


if __name__ == "__main__":
    main()
