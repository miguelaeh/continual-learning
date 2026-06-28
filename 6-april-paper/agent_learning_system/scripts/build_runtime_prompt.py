#!/usr/bin/env python3
"""Assemble the runtime prompt from profile memory and episodic retrieval."""

from __future__ import annotations

import argparse
import json

from agent_learning_system.profile import ProfileMemoryStore
from agent_learning_system.runtime import build_runtime_prompt
from agent_learning_system.store import EpisodicStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-dir", required=True)
    parser.add_argument("--user-message", required=True)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--retrieval-limit", type=int, default=5)
    parser.add_argument("--recent-limit", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    episodic = EpisodicStore(args.store_dir)
    profile = ProfileMemoryStore(args.store_dir)
    payload = build_runtime_prompt(
        store=episodic,
        profile_store=profile,
        user_message=args.user_message,
        session_id=args.session_id,
        retrieval_limit=args.retrieval_limit,
        recent_limit=args.recent_limit,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(payload["prompt"])


if __name__ == "__main__":
    main()
