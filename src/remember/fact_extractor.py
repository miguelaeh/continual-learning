"""Extract factual knowledge from conversations via self-distillation.

Takes a conversation log and uses an LLM to extract clean factual statements.
These facts are then formatted as training data for sparse memory finetuning.

Supports two backends:
- Local model via transformers (the memory-augmented Gemma itself)
- External API via OpenAI-compatible endpoint (Ollama, vLLM, etc.)
"""

import json
import logging
import re
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
You are a fact extraction system. Given a conversation, extract all factual \
statements, preferences, and important information that was shared.

Rules:
- Output one fact per line
- Write each fact as a clear, self-contained statement
- Include who the fact is about when relevant (e.g., "The user's name is Miguel")
- Skip greetings, pleasantries, and meta-conversation
- Skip questions that were asked but not answered
- If no facts are present, output "NO_FACTS"

Conversation:
{conversation}

Extracted facts:"""

FORMATTING_PROMPT = """\
Rewrite the following fact as a natural passage that a language model could \
learn from. Write 1-3 sentences that convey the information clearly. \
Do not add information that is not in the original fact.

Fact: {fact}

Passage:"""


@dataclass
class Message:
    role: str  # "user", "assistant", "system"
    content: str


def load_conversation(path: str) -> list[Message]:
    """Load a conversation from a JSON file.

    Supports two formats:
    1. List of {"role": str, "content": str} objects (OpenAI-style)
    2. List of {"sender": str, "text": str} objects (generic)
    """
    with open(path) as f:
        data = json.load(f)

    messages = []
    if isinstance(data, list):
        for msg in data:
            if "role" in msg and "content" in msg:
                messages.append(Message(role=msg["role"], content=msg["content"]))
            elif "sender" in msg and "text" in msg:
                messages.append(Message(role=msg["sender"], content=msg["text"]))
            else:
                raise ValueError(f"Unknown message format: {msg.keys()}")
    elif isinstance(data, dict) and "messages" in data:
        return load_conversation_from_list(data["messages"])
    else:
        raise ValueError(
            "Expected a JSON list of messages or a dict with 'messages' key"
        )

    return messages


def load_conversation_from_list(messages_list: list[dict]) -> list[Message]:
    """Load from a list of message dicts."""
    return [
        Message(
            role=m.get("role", m.get("sender", "unknown")),
            content=m.get("content", m.get("text", "")),
        )
        for m in messages_list
    ]


def format_conversation(messages: list[Message]) -> str:
    """Format a conversation into a readable string for the extraction prompt."""
    lines = []
    for msg in messages:
        role = msg.role.capitalize()
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)


def extract_facts_local(
    messages: list[Message],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str = "cuda",
    max_new_tokens: int = 1024,
) -> list[str]:
    """Extract facts using a local model (the memory-augmented Gemma itself).

    Args:
        messages: Conversation messages.
        model: The model to use for extraction.
        tokenizer: Tokenizer for the model.
        device: Device for inference.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        List of extracted factual statements.
    """
    conversation_text = format_conversation(messages)
    prompt = EXTRACTION_PROMPT.format(conversation=conversation_text)

    # Format as chat for instruction-tuned model
    chat_messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
        )

    # Decode only the generated part
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()

    return _parse_facts(response)


def extract_facts_api(
    messages: list[Message],
    api_base: str = "http://localhost:11434/v1",
    model_name: str = "gemma3:4b",
    api_key: str = "ollama",
) -> list[str]:
    """Extract facts using an OpenAI-compatible API (Ollama, vLLM, etc.).

    Args:
        messages: Conversation messages.
        api_base: API base URL.
        model_name: Model name for the API.
        api_key: API key (use "ollama" for Ollama).

    Returns:
        List of extracted factual statements.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package required for API extraction. "
            "Install with: pip install openai"
        )

    client = OpenAI(base_url=api_base, api_key=api_key)

    conversation_text = format_conversation(messages)
    prompt = EXTRACTION_PROMPT.format(conversation=conversation_text)

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024,
    )

    return _parse_facts(response.choices[0].message.content.strip())


def facts_to_training_text(
    facts: list[str],
    model: AutoModelForCausalLM | None = None,
    tokenizer: AutoTokenizer | None = None,
    api_base: str | None = None,
    model_name: str | None = None,
    api_key: str = "ollama",
    device: str = "cuda",
) -> list[str]:
    """Convert extracted facts into natural training passages.

    Each fact is expanded into a 1-3 sentence passage that reads naturally
    as training data for a language model.

    If no model/API is provided, facts are used as-is (still works, just less natural).
    """
    if not facts:
        return []

    # If no expansion model available, use facts directly
    if model is None and api_base is None:
        logger.info("No expansion model provided, using raw facts as training text")
        return facts

    passages = []
    for fact in facts:
        prompt = FORMATTING_PROMPT.format(fact=fact)

        if model is not None and tokenizer is not None:
            chat_messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(formatted, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=True,
                )

            generated = outputs[0][inputs["input_ids"].shape[1] :]
            passage = tokenizer.decode(generated, skip_special_tokens=True).strip()
        else:
            try:
                from openai import OpenAI
            except ImportError:
                logger.warning("openai not installed, using raw fact")
                passages.append(fact)
                continue

            client = OpenAI(base_url=api_base, api_key=api_key)
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=256,
            )
            passage = response.choices[0].message.content.strip()

        passages.append(passage)

    logger.info(f"Expanded {len(facts)} facts into {len(passages)} training passages")
    return passages


def _parse_facts(response: str) -> list[str]:
    """Parse the extraction model's response into individual facts."""
    if "NO_FACTS" in response:
        return []

    facts = []
    for line in response.strip().split("\n"):
        line = line.strip()
        # Remove common list prefixes
        line = re.sub(r"^[-*\d.)\]]+\s*", "", line)
        line = line.strip()
        if line and len(line) > 5:  # skip very short lines
            facts.append(line)

    return facts
