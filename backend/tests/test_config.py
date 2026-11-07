"""The hosted key: read from .env only, and kept out of the environment and settings."""
import os

import pytest

from app.config import (ROOT, HOSTED_KEY, OPENROUTER_KEY, Settings, hosted_api_key,
                        load_settings, openrouter_api_key)

FAKE = "not-a-real-key"


def env_file(tmp_path, text: str):
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_key_comes_from_the_env_file(tmp_path):
    assert hosted_api_key(env_file(tmp_path, f"{HOSTED_KEY}={FAKE}\n")) == FAKE


def test_a_quoted_key_with_a_dollar_sign_is_read_as_written(tmp_path):
    assert hosted_api_key(env_file(tmp_path, f"{HOSTED_KEY}='a$b'\n")) == "a$b"


@pytest.mark.parametrize("name", [HOSTED_KEY, "ANTHROPIC_API_KEY"])
def test_the_process_environment_is_never_read(tmp_path, monkeypatch, name):
    monkeypatch.setenv(name, FAKE)
    assert hosted_api_key(tmp_path / "missing.env") is None


def test_another_variable_in_the_file_is_not_a_key(tmp_path):
    assert hosted_api_key(env_file(tmp_path, f"ANTHROPIC_API_KEY={FAKE}\n")) is None


def test_an_empty_value_counts_as_no_key(tmp_path):
    assert hosted_api_key(env_file(tmp_path, f"{HOSTED_KEY}=\n")) is None


def test_reading_the_key_sets_no_environment_variable(tmp_path, monkeypatch):
    monkeypatch.delenv(HOSTED_KEY, raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    before = dict(os.environ)
    hosted_api_key(env_file(tmp_path, f"{HOSTED_KEY}={FAKE}\nANTHROPIC_API_KEY={FAKE}\n"))
    assert dict(os.environ) == before


def test_settings_have_no_field_for_the_key():
    assert not any("key" in name for name in vars(Settings()))


def test_the_env_file_is_ignored_by_git():
    assert ".env" in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize("name", [OPENROUTER_KEY, "OPENAI_API_KEY", "OPENROUTER_API_KEY"])
def test_the_openrouter_key_never_comes_from_the_process_environment(tmp_path, monkeypatch, name):
    monkeypatch.setenv(name, FAKE)
    assert openrouter_api_key(tmp_path / "missing.env") is None


def test_the_openrouter_key_comes_from_the_env_file(tmp_path):
    assert openrouter_api_key(env_file(tmp_path, f"{OPENROUTER_KEY}={FAKE}\n")) == FAKE


def config(tmp_path, text: str):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


FAST = "  - {id: fast, label: Fast, provider: ollama, model: gpt-oss:20b, think: medium}\n"


def test_the_project_config_lists_the_four_options_in_order():
    settings = load_settings()
    assert [option.id for option in settings.answer_options] == ["fast", "balanced", "qwen", "deep"]
    assert settings.answer_options[0].model == "gpt-oss:20b"
    qwen = settings.option("qwen")
    assert (qwen.provider, qwen.model, qwen.route, qwen.quantization) == (
        "openrouter", "qwen/qwen3.8-27b", "DeepInfra", "bf16")
    deep = settings.option("deep")
    assert (deep.model, deep.fallback) == ("claude-opus-5-5", "claude-sonnet-5-5")


def test_web_search_is_offered_where_the_gate_run_says_it_works():
    settings = load_settings()
    assert {option.id: option.web for option in settings.answer_options} == {
        "fast": "off", "balanced": "off", "qwen": "tools", "deep": "tools"}


def test_a_bare_off_for_web_reads_as_off(tmp_path):
    settings = load_settings(config(tmp_path, "answer_options:\n" + FAST +
                                    "  - {id: g, label: G, provider: ollama, model: gemma, web: off}\n"))
    assert settings.option("g").web == "off" and settings.option("fast").web == "tools"


def test_balanced_may_think_to_16k_and_the_others_keep_the_default():
    settings = load_settings()
    balanced = settings.option("balanced").apply(settings)
    fast = settings.option("fast").apply(settings)
    assert balanced.chat_num_predict == 16384 and fast.chat_num_predict == settings.chat_num_predict == 8192


def test_an_option_sets_only_its_own_model(tmp_path):
    settings = load_settings(config(tmp_path, "answer_options:\n" + FAST +
                                    "  - {id: g, label: G, provider: ollama, model: gemma, think: true}\n"))
    gemma = settings.option("g").apply(settings)
    assert (gemma.chat_model, gemma.chat_think, gemma.hosted_model) == ("gemma", True, settings.hosted_model)


@pytest.mark.parametrize("entry, message", [
    ("  - {id: fast, label: Again, provider: ollama, model: x}\n", "repeat"),
    ("  - {id: x, label: X, provider: elsewhere, model: x}\n", "provider must be"),
    ("  - {id: x, label: X, provider: openrouter, model: qwen/qwen3.8-27b}\n", "needs a route"),
    ("  - {id: x, label: X, provider: openrouter, model: someone/new, route: Y}\n", "no price for someone/new"),
    ("  - {id: x, label: X, provider: anthropic, model: claude-opus-5-5, fallback: claude-new}\n", "no price for claude-new"),
    ("  - {id: x, label: X, provider: ollama, model: x, colour: red}\n", "Unknown answer option settings"),
    ("  - {id: x, label: X, provider: ollama, model: x, web: sometimes}\n", "web must be one of"),
    ("  - {id: x, label: X, provider: anthropic, model: claude-opus-5-5, num_predict: 9000}\n", "num_predict is for"),
    ("  - {id: x, label: X, provider: ollama, model: x, num_predict: 32768}\n", "must leave room"),
])
def test_a_mistake_in_the_options_stops_the_start(tmp_path, entry, message):
    with pytest.raises(ValueError, match=message):
        load_settings(config(tmp_path, "answer_options:\n" + FAST + entry))


def test_an_empty_list_of_options_stops_the_start(tmp_path):
    with pytest.raises(ValueError, match="no answer options"):
        load_settings(config(tmp_path, "answer_options: []\n"))


def test_another_config_file_can_be_named_in_the_environment(tmp_path, monkeypatch):
    from app.config import CONFIG_ENV, load_settings
    other = tmp_path / "other.yaml"
    other.write_text(f"port: 8001\ndata_dir: {tmp_path / 'data'}\n", encoding="utf-8")
    monkeypatch.setenv(CONFIG_ENV, str(other))
    settings = load_settings()
    assert settings.port == 8001 and settings.data_dir == tmp_path / "data"
    monkeypatch.delenv(CONFIG_ENV)
    assert load_settings().port == 8000
