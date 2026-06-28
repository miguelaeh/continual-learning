from smf_retrofit.data import _oasst1_path_to_messages


def test_oasst1_path_to_text_formats_dialogue():
    path = [
        {"role": "prompter", "text": "What is 2 plus 2?"},
        {"role": "assistant", "text": "2 plus 2 is 4."},
        {"role": "prompter", "text": "Why?"},
        {"role": "assistant", "text": "Because addition combines two and two."},
    ]
    messages = _oasst1_path_to_messages(path)
    assert messages is not None
    assert messages[0] == {"role": "user", "content": "What is 2 plus 2?"}
    assert messages[1] == {"role": "assistant", "content": "2 plus 2 is 4."}


def test_oasst1_path_to_text_skips_paths_without_assistant():
    path = [{"role": "prompter", "text": "Hello"}]
    assert _oasst1_path_to_messages(path) is None
