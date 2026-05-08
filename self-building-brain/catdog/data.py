from __future__ import annotations

import random
from dataclasses import dataclass

from catdog.vocab import encode_tokens


@dataclass(frozen=True)
class Episode:
    stage_index: int
    sentence_tokens: tuple[str, ...]
    token_ids: tuple[int, ...]


def build_sentence_stages() -> list[list[list[str]]]:
    dets = ["the", "a"]
    nouns = ["cat", "dog"]
    verbs = ["runs", "jumps"]
    colors = ["red", "blue"]
    sizes = ["big", "small"]
    moods = ["happy"]
    adjectives = colors + sizes + moods

    stage0: set[tuple[str, ...]] = {
        ("Mina",),
    }
    stage1: set[tuple[str, ...]] = {
        ("I", "Mina"),
    }
    stage2: set[tuple[str, ...]] = {
        ("I", "am", "Mina"),
        ("Mina", "is", "happy", "."),
    }
    stage3: set[tuple[str, ...]] = set()
    stage4: set[tuple[str, ...]] = set()
    stage5: set[tuple[str, ...]] = set()

    for det in dets:
        for noun in nouns:
            for mood in moods:
                stage4.add((det, noun, "is", mood, "."))
            for verb in verbs:
                stage3.add((det, noun, verb, "."))
                stage4.add((det, noun, "is", "on", "mat", "."))
            for adj in adjectives:
                for verb in verbs:
                    stage4.add((det, adj, noun, verb, "."))
                    stage4.add((det, adj, noun, "is", mood, "."))
                    stage4.add((det, adj, noun, "is", "on", "mat", "."))
            for adj1 in sizes:
                for adj2 in colors:
                    for verb in verbs:
                        stage5.add((det, adj1, adj2, noun, verb, "."))
                        stage5.add((det, adj1, adj2, noun, "is", mood, "."))
                        stage5.add((det, adj1, adj2, noun, "is", "on", "mat", "."))

    # A few no-determiner variants.
    for noun in nouns:
        for color in colors:
            stage4.add((color, noun, "is", "happy", "."))
            stage4.add((color, noun, "runs", "."))
        for size in sizes:
            stage4.add((size, noun, "jumps", "."))
            stage4.add((size, noun, "is", "on", "mat", "."))

    return [
        [list(sentence) for sentence in sorted(stage0)],
        [list(sentence) for sentence in sorted(stage1)],
        [list(sentence) for sentence in sorted(stage2)],
        [list(sentence) for sentence in sorted(stage3)],
        [list(sentence) for sentence in sorted(stage4)],
        [list(sentence) for sentence in sorted(stage5)],
    ]


def build_sentences() -> list[list[str]]:
    stages = build_sentence_stages()
    all_sentences = []
    for stage in stages:
        all_sentences.extend(stage)
    return all_sentences


def build_episodes(sentences: list[list[str]], stage_index: int, include_eos: bool = True) -> list[Episode]:
    episodes: list[Episode] = []
    for sentence in sentences:
        full = ["<BOS>", *sentence]
        if include_eos:
            full.append("<EOS>")
        ids = encode_tokens(full)
        episodes.append(
            Episode(
                stage_index=stage_index,
                sentence_tokens=tuple(sentence),
                token_ids=tuple(ids),
            )
        )
    return episodes


def build_transition_episodes(sentences: list[list[str]], stage_index: int, include_eos: bool = True) -> list[Episode]:
    episodes: list[Episode] = []
    seen: set[tuple[int, int]] = set()
    for sentence in sentences:
        full = ["<BOS>", *sentence]
        if include_eos:
            full.append("<EOS>")
        ids = encode_tokens(full)
        for source_id, target_id in zip(ids, ids[1:]):
            pair = (source_id, target_id)
            if pair in seen:
                continue
            seen.add(pair)
            episodes.append(
                Episode(
                    stage_index=stage_index,
                    sentence_tokens=(f"{source_id}->{target_id}",),
                    token_ids=pair,
                )
            )
    return episodes


def train_eval_split(seed: int = 13, eval_fraction: float = 0.2) -> tuple[list[Episode], list[Episode]]:
    staged_sentences = build_sentence_stages()
    sentences: list[tuple[int, list[str]]] = []
    for stage_index, stage_sentences in enumerate(staged_sentences):
        for sentence in stage_sentences:
            sentences.append((stage_index, sentence))
    rng = random.Random(seed)
    rng.shuffle(sentences)
    eval_size = max(1, int(len(sentences) * eval_fraction))
    eval_sentences = sentences[:eval_size]
    train_sentences = sentences[eval_size:]
    train_episodes: list[Episode] = []
    eval_episodes: list[Episode] = []
    for stage_index, sentence in train_sentences:
        train_episodes.extend(build_episodes([sentence], stage_index=stage_index, include_eos=stage_index > 0))
    for stage_index, sentence in eval_sentences:
        eval_episodes.extend(build_episodes([sentence], stage_index=stage_index, include_eos=stage_index > 0))
    return train_episodes, eval_episodes


def train_eval_split_by_stage(
    seed: int = 13,
    eval_fraction: float = 0.2,
    curriculum_mode: str = "sentences",
) -> tuple[list[list[Episode]], list[list[Episode]]]:
    rng = random.Random(seed)
    train_stages: list[list[Episode]] = []
    eval_stages: list[list[Episode]] = []
    if curriculum_mode not in {"sentences", "transitions"}:
        raise ValueError(f"Unknown curriculum_mode: {curriculum_mode}")
    for stage_index, stage_sentences in enumerate(build_sentence_stages()):
        shuffled = list(stage_sentences)
        rng.shuffle(shuffled)
        if len(shuffled) <= 1:
            eval_size = 0
        else:
            eval_size = min(len(shuffled) - 1, max(1, int(len(shuffled) * eval_fraction)))
        include_eos = stage_index > 0
        episode_builder = build_transition_episodes if curriculum_mode == "transitions" else build_episodes
        train_stages.append(episode_builder(shuffled[eval_size:], stage_index=stage_index, include_eos=include_eos))
        eval_stages.append(episode_builder(shuffled[:eval_size], stage_index=stage_index, include_eos=include_eos))
    return train_stages, eval_stages


def sample_batch(episodes: list[Episode], batch_size: int, rng: random.Random) -> list[Episode]:
    indices = [rng.randrange(len(episodes)) for _ in range(batch_size)]
    return [episodes[index] for index in indices]


def describe_episode(episode: Episode) -> str:
    return f"tokens={list(episode.token_ids)}"
