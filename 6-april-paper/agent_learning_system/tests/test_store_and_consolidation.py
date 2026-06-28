import json

from agent_learning_system.consolidation import build_consolidation_examples
from agent_learning_system.profile import ProfileMemoryStore
from agent_learning_system.registry import CheckpointRegistry
from agent_learning_system.retrieval import retrieve_relevant_traces
from agent_learning_system.runtime import build_runtime_prompt
from agent_learning_system.store import EpisodicStore


def test_store_append_and_recent(tmp_path):
    store = EpisodicStore(tmp_path / "runtime")
    store.append_interaction(
        session_id="s1",
        user_message="What is my name?",
        assistant_response="Your name is Miguel.",
        outcome="success",
    )
    recent = store.recent(limit=1)
    assert len(recent) == 1
    assert recent[0]["session_id"] == "s1"


def test_consolidation_examples_from_successful_traces(tmp_path):
    store = EpisodicStore(tmp_path / "runtime")
    store.append_interaction(
        session_id="s1",
        user_message="What is my name?",
        assistant_response="Your name is Miguel.",
        outcome="success",
    )
    store.append_interaction(
        session_id="s1",
        user_message="What is 2 plus 2?",
        assistant_response="5",
        outcome="failure",
        correction="2 plus 2 is 4.",
    )

    examples = build_consolidation_examples(store.load_all(), include_outcomes={"success", "corrected"})
    assert len(examples) == 1
    assert examples[0].messages[-1]["content"] == "Your name is Miguel."


def test_retrieval_returns_overlap_matches():
    traces = [
        {"user_message": "What is my name?", "assistant_response": "Your name is Miguel.", "summary": "", "tags": []},
        {"user_message": "What is the capital of France?", "assistant_response": "Paris", "summary": "", "tags": []},
    ]
    matches = retrieve_relevant_traces("name", traces, limit=1)
    assert len(matches) == 1
    assert "Miguel" in matches[0]["assistant_response"]


def test_registry_promote_archives_previous(tmp_path):
    registry = CheckpointRegistry(tmp_path / "registry.json")
    registry.register_candidate("a", "/tmp/a")
    registry.register_candidate("b", "/tmp/b")
    registry.promote("a")
    registry.promote("b")

    rows = registry.list_records()
    by_id = {row["checkpoint_id"]: row for row in rows}
    assert by_id["a"]["status"] == "archived"
    assert by_id["b"]["status"] == "promoted"


def test_profile_memory_upsert_roundtrip(tmp_path):
    profile = ProfileMemoryStore(tmp_path / "runtime")
    profile.upsert(
        memory_type="fact",
        key="user.name",
        value="Miguel",
        source="user stated it directly",
        tags=["identity"],
    )
    profile.upsert(
        memory_type="fact",
        key="user.name",
        value="Miguel A.",
        source="user corrected full name",
        tags=["identity", "updated"],
    )
    rows = profile.load_all()
    assert len(rows) == 1
    assert rows[0]["key"] == "user.name"
    assert rows[0]["value"] == "Miguel A."


def test_runtime_prompt_includes_profile_and_retrieval(tmp_path):
    root = tmp_path / "runtime"
    episodic = EpisodicStore(root)
    profile = ProfileMemoryStore(root)
    profile.upsert("fact", "user.name", "Miguel", tags=["identity"])
    profile.upsert("preference", "answer_style", "concise", tags=["style"])
    episodic.append_interaction(
        session_id="s1",
        user_message="What is my name?",
        assistant_response="Your name is Miguel.",
        outcome="success",
        summary="Confirmed the user's name.",
        learned_type="fact",
        tags=["identity"],
    )
    episodic.append_interaction(
        session_id="s1",
        user_message="Be concise.",
        assistant_response="Understood.",
        outcome="success",
        summary="User prefers concise answers.",
        learned_type="preference",
        tags=["style"],
    )

    payload = build_runtime_prompt(
        store=episodic,
        profile_store=profile,
        user_message="Can you remind me what my name is?",
        session_id="s1",
        retrieval_limit=2,
        recent_limit=2,
    )
    prompt = payload["prompt"]
    assert "user.name: Miguel" in prompt
    assert "answer_style: concise" in prompt
    assert "Your name is Miguel." in prompt
    assert "Be concise." in prompt
