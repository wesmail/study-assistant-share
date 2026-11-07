"""Paths, model names and limits. Defaults live here; config.yaml overrides them."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import yaml
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"
HOSTED_KEY = "STUDY_ASSISTANT_ANTHROPIC_KEY"
OPENROUTER_KEY = "STUDY_ASSISTANT_OPENROUTER_KEY"
PROVIDERS = ("ollama", "openrouter", "anthropic")


def hosted_api_key(path: Path = ENV_FILE) -> str | None:
    """The hosted model's key, from the .env file and nowhere else.

    It stays out of os.environ, and out of Settings so that no repr or log can show
    it. Pass it to the client directly. ANTHROPIC_API_KEY is never read or set.
    """
    return dotenv_values(path, interpolate=False).get(HOSTED_KEY) or None


def openrouter_api_key(path: Path = ENV_FILE) -> str | None:
    """The OpenRouter key, under the same rules as the hosted key: the .env file only,
    never os.environ or Settings, and never printed, logged or put in an error."""
    return dotenv_values(path, interpolate=False).get(OPENROUTER_KEY) or None


@dataclass(frozen=True)
class Price:
    """Dollars per million tokens, as the provider listed them on `checked`.

    Cache rates differ by model (0.05x input on Opus 5.5 and Sonnet 5.5, 0.1x on most
    others), so they're written out. Missing, they're charged at the input price and
    1.25x, which can only overstate a bill.
    """

    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None    # the 5-minute write, the only one the app uses
    checked: str | None = None          # the date the price was read from `source`
    source: str | None = None


def price_of(entry) -> Price:
    if isinstance(entry, Price):
        return entry
    if isinstance(entry, (list, tuple)):
        return Price(*entry)
    entry = dict(entry)
    if "checked" in entry:
        entry["checked"] = str(entry["checked"])     # YAML reads a bare date as a date
    return Price(**entry)


@dataclass(frozen=True)
class AnswerOption:
    """One choice in the answer selector. `apply` turns it into the settings for a question."""

    id: str
    label: str
    provider: str                   # ollama, openrouter or anthropic
    model: str
    think: str | bool | None = None         # ollama: as chat_think
    # ollama: the most tokens one reply may use, thinking included, instead of
    # chat_num_predict. Gemma's derivations ran out at 8K in tools mode (Phase 5d).
    num_predict: int | None = None
    fallback: str | None = None             # anthropic: restarts on this model if the first fails
    effort: str | None = None               # anthropic
    route: str | None = None                # openrouter: the one provider allowed to serve it
    quantization: str | None = None         # openrouter: what that provider must serve
    sampling: dict = field(default_factory=dict)        # openrouter: copied from the local model file
    reasoning: dict = field(default_factory=dict)       # openrouter: thinking on or off
    expected_seconds: float | None = None   # median per answer, measured; shown before asking
    expected_note: str | None = None        # what that median was measured on
    # Mean input and output tokens per question, measured; priced at today's price for
    # the expected cost, so a price change moves it without measuring again.
    expected_tokens: tuple[int, int] | None = None
    # Reads images: retrieved figures and pasted screenshots go to it as images. Without
    # it, a figure reaches the model only as its written description.
    vision: bool = False
    # Web search (Phase 5d): "tools", the model searches and opens pages itself; "app", the
    # app searches and hands it the top pages, so it only answers; "off", no Web toggle and
    # no proposed searches. Local models used the web tools poorly in the gate run.
    web: str = "tools"
    # With Check on (verification layer): how many answers it may write until one's claims
    # all pass (best-of-N; 1 means check badges only, no resampling), and the longest the
    # question may take before no new attempt starts and a running one is given up.
    check_samples: int = 1
    check_ceiling_seconds: float | None = None
    # How the badge reads: "full" ("N of M checked claims passed"); "limited", neutral, with a
    # note that most of the answer isn't checkable: Fast and Balanced, whose answers had a
    # checkable claim in only 2 to 6 of 10 gold derivations (round 19); or "unmeasured",
    # neutral, with a note that the option's checks haven't been measured (Qwen).
    check_badge: str = "full"

    @property
    def hosted(self) -> bool:
        return self.provider != "ollama"

    def apply(self, settings: Settings) -> Settings:
        settings = replace(settings, check_samples=self.check_samples, check_ceiling_seconds=self.check_ceiling_seconds)
        if self.provider == "ollama":
            changes = {"chat_model": self.model}
            if self.think is not None:
                changes["chat_think"] = self.think
            if self.num_predict is not None:
                changes["chat_num_predict"] = self.num_predict
        elif self.provider == "anthropic":
            changes = {"hosted_model": self.model, "hosted_fallback_model": self.fallback}
            if self.effort:
                changes["hosted_effort"] = self.effort
        else:
            changes = {"openrouter_model": self.model, "openrouter_route": self.route,
                       "openrouter_quantization": self.quantization,
                       "openrouter_sampling": dict(self.sampling), "openrouter_reasoning": dict(self.reasoning)}
        return replace(settings, **changes)


@dataclass(frozen=True)
class FigureDescriber:
    """Who writes the figure descriptions used for search (chosen blind on ten held-out figures)."""

    provider: str                   # anthropic or ollama
    model: str
    effort: str | None = None       # anthropic
    # Mean input and output tokens per figure, measured; priced for the estimate shown
    # before a hosted describer runs.
    expected_tokens: tuple[int, int] | None = None

    @property
    def hosted(self) -> bool:
        return self.provider != "ollama"

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"


WEB_MODES = ("tools", "app", "off")

DEFAULT_OPTIONS = (
    AnswerOption("fast", "Fast", "ollama", "gpt-oss:20b", think="medium", web="off"),
    AnswerOption("deep", "Deep", "anthropic", "claude-opus-5-5", fallback="claude-sonnet-5-5", effort="high",
                 vision=True),
)


@dataclass
class Settings:
    library_dir: Path = ROOT / "library"
    data_dir: Path = ROOT / "data"
    outputs_dir: Path = ROOT / "outputs"
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "bge-m3"
    embed_dimensions: int = 1024
    embed_batch: int = 16
    embed_timeout_seconds: int = 300
    max_attempts: int = 3
    settle_seconds: float = 5.0
    port: int = 8000
    chat_model: str = "gpt-oss:20b"
    # gpt-oss takes "low", "medium" or "high"; Qwen takes true or false. "high"
    # loops on gpt-oss and returns nothing (measured in the first benchmark).
    chat_think: str | bool = "medium"
    # Sent with every request, because Ollama's default is far too small and a
    # silently truncated prompt looks like a bad answer. 8K answer + 14K sources
    # (search_source_tokens) + 3K carried by a follow-up + 6K history needs 31K; 32K
    # costs only 576 MiB over 8K. An option's larger num_predict (Balanced: 16K) fits a
    # first question (about 9K of prompt in tools mode); on a long follow-up the context
    # ends it first, and a reply cut off that way is shown as one, with Ask again in Deep.
    chat_num_ctx: int = 32768
    chat_num_predict: int = 8192
    chat_keep_alive: str = "30m"
    chat_timeout_seconds: int = 600
    chat_history_tokens: int = 6000
    search_depth: int = 30      # how far down each list RRF looks
    search_top_k: int = 6
    search_neighbour_hits: int = 3      # how many top text hits bring in the chunks around them
    search_figure_slots: int = 2        # figure hits from the top_k, on top of top_k text hits
    search_summary_slots: int = 2       # summary-tree hits from the top_k, likewise
    search_note_slots: int = 2          # the reader's notes from the top_k, likewise (roadmap item 4)
    search_note_candidates: int = 5     # notes put among the candidates, from the notes' own index
    search_notes: str = "with"          # "with" the books, "only" (the Notes scope), or "off"
    # The answer options that may suggest highlights (roadmap item 4, step 5): Deep, and Balanced
    # only once it meets the gate's quote check (decision 2). The first is the default.
    suggest_options: list = field(default_factory=lambda: ["deep"])
    # The answer options Explain may use, per kind; the reader's mode if listed, else the first
    # (gate of roadmap item 4, Nov 6: Balanced answered 5 of 5 code blocks but only 2 of 5 equations).
    explain_options: dict = field(default_factory=lambda: {"equation": ["deep"], "code": ["deep", "balanced"]})
    search_summary_tokens: int = 3000   # a summary hit and the summaries of its parts, at most
    search_neighbour_radius: int = 2    # how many chunks each way
    search_source_tokens: int = 13600   # hits are always read; neighbours fill up to this
    rrf_k: int = 60
    rerank_enabled: bool = False            # config.yaml turns it on; tests never load the model
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_revision: str = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    rerank_device: str = "cuda"             # 0.2 s for 30 passages; 26 s on the CPU
    rerank_candidates: int = 30
    rerank_batch: int = 8
    rerank_max_length: int = 512
    helper_model: str = "qwen3:1.7b"        # about 0.2 s a rewrite once loaded
    helper_keep_alive: int = -1             # stays resident: its first load took 15 s
    helper_timeout_seconds: int = 30
    rewrite_num_ctx: int = 2048
    rewrite_num_predict: int = 60
    rewrite_answer_chars: int = 1500        # of the previous answer, to fit that context
    followup_carry_tokens: int = 3000       # of what the previous answer cited
    # Tools. tool_rounds counts model replies in one turn; a file takes three (the call,
    # the document, the reply about it), so this allows a chain or two with a repair.
    tools_enabled: bool = True
    tool_rounds: int = 10
    tool_output_chars: int = 6000           # of a run's output, to the model
    tool_search_tokens: int = 4000          # of passages, per search_library call
    run_seconds: int = 30
    run_max_files: int = 8
    build_seconds: int = 120                # a deck compiles once, however many equations
    sympy_seconds: int = 20
    # Verification layer (roadmap item 3): the claims block an answer ends with.
    verify_seconds: int = 120               # the whole block, in one container; each claim gets 10 s
    verify_block_chars: int = 6000          # longer blocks are cut before they go in
    check_samples: int = 1                  # set per answer option (AnswerOption.check_samples)
    check_ceiling_seconds: float | None = None
    sandbox_image: str = "study-assistant-sandbox:1"
    sandbox_memory: str = "2g"
    sandbox_cpus: float = 2.0
    sandbox_pids: int = 256
    sandbox_file_bytes: int = 64 * 1024 * 1024
    sandbox_work_bytes: int = 512 * 1024 * 1024     # the whole work folder, which is on the host (review S6)
    sandbox_work_entries: int = 20000                # files and folders in it, against a flood of empty files
    sandbox_output_bytes: int = 256 * 1024  # kept of stdout and of stderr, per run
    # LibreOffice (roadmap item 7): Word and PowerPoint to PDF, and reading decks, in its own
    # image under the sandbox's limits (app/office.py). The fonts folder is mounted read-only
    # so decks render with their own fonts; None renders with the image's stand-ins.
    office_image: str = "study-assistant-office:1"
    office_fonts_dir: Path | None = None
    office_seconds: int = 180               # one render; a big deck takes tens of seconds
    office_memory: str = "3g"
    office_max_bytes: int = 200 * 1024 * 1024   # a larger file isn't opened at all
    upload_max_bytes: int = 200 * 1024 * 1024   # the library pane's Add button (app/upload.py)
    office_max_unpacked_bytes: int = 1024 * 1024 * 1024     # by the zip's own directory
    office_max_slides: int = 600
    hosted_model: str = "claude-opus-5-5"
    hosted_fallback_model: str | None = "claude-sonnet-5-5"     # None means no fallback
    # Effort, not a thinking budget, sets the depth. Models differ in their default, so
    # it's always sent. Answers run to about 3K tokens; the rest of this is for thinking.
    hosted_effort: str = "high"
    hosted_max_tokens: int = 16000
    # Dollars per million input and output tokens: a model without a price isn't called.
    # config.yaml holds the checked prices; these are only for tests and a bare start.
    hosted_prices: dict[str, Price] = field(default_factory=lambda: {
        "claude-opus-5-5": Price(4.0, 20.0, cache_read=0.20, cache_write=5.0),
        "claude-sonnet-5-5": Price(2.0, 10.0, cache_read=0.10, cache_write=2.5),
        "qwen/qwen3.8-27b": Price(0.15, 1.875),
    })
    # Monthly limits by provider, for the selector's spend line. Anthropic's is the
    # study-assistant workspace limit; OpenRouter has none set here.
    hosted_monthly_limits: dict[str, float] = field(default_factory=lambda: {"anthropic": 80.0})
    # Set per question, so every hosted call it makes is logged under it.
    turn_id: str | None = None
    figure_describer: FigureDescriber = field(default_factory=lambda: FigureDescriber(
        "anthropic", "claude-sonnet-5-5", effort="high", expected_tokens=(1302, 770)))
    # The summary tree's model, chosen in Phase 5b; the worker's summarize stage uses it.
    # A change misses the cache, so every document's tree would be written again.
    summarizer_model: str = "qwen3.8:27b"
    summarizer_think: bool | str = False
    # The self-hosted SearXNG behind web_search (scripts/searxng.sh).
    searxng_url: str = "http://127.0.0.1:8888"
    # With web search off, the model may propose a search, which the reader confirms with a click.
    web_proposals: bool = True
    # Collections whole-library search leaves out. Set per request by the server from the
    # reader's library setting; empty everywhere else, so the evaluation suites see every book.
    hidden_collections: tuple[str, ...] = ()
    figure_max_tokens: int = 16000
    figure_page_text_chars: int = 1800      # of the page's text, given with the caption
    figure_busy_seconds: int = 900          # a busy marker older than this is left over, not a chat
    figure_images_max: int = 4              # figure images sent with one question, to a vision option
    # OpenRouter, for the option that uses it. One provider, no fallbacks, and none that
    # keeps or trains on prompts. Sampling copies the local Ollama model file.
    openrouter_model: str = "qwen/qwen3.8-27b"
    openrouter_route: str | None = "DeepInfra"
    openrouter_quantization: str | None = "bf16"
    openrouter_sampling: dict = field(default_factory=lambda: {
        "temperature": 1, "top_k": 20, "top_p": 0.95, "min_p": 0, "presence_penalty": 0})
    openrouter_reasoning: dict = field(default_factory=lambda: {"enabled": True})
    openrouter_max_tokens: int = 16000      # as hosted_max_tokens; thinking counts against it
    # Measured from one local model to the other: only one fits in the GPU.
    local_switch_seconds: list[int] = field(default_factory=lambda: [20, 60])
    # The answer selector, in order. The first is the default.
    answer_options: list[AnswerOption] = field(default_factory=lambda: list(DEFAULT_OPTIONS))
    # Backups (app/backup.py, scripts/backup.sh); None turns them off. Archives go to
    # backup_dir; the .env copies, the log and the lock stay in backup_local_dir, under /home.
    backup_dir: Path | None = None
    backup_local_dir: Path = Path.home() / "backups" / "study-assistant"
    backup_keep_daily: int = 7
    backup_keep_weekly: int = 4
    # Below 24 so a daily timer firing a few seconds early doesn't skip a day.
    backup_max_age_hours: float = 20

    def option(self, option_id: str) -> AnswerOption:
        for option in self.answer_options:
            if option.id == option_id:
                return option
        raise KeyError(option_id)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.sqlite"

    @property
    def parsed_dir(self) -> Path:
        return self.data_dir / "parsed"

    @property
    def renders_dir(self) -> Path:
        """LibreOffice's PDFs of decks and of Word and PowerPoint outputs, named by the
        source's SHA-256, so a changed file gets a new one (roadmap item 7)."""
        return self.data_dir / "renders"


CONFIG_ENV = "STUDY_ASSISTANT_CONFIG"     # another config file: a second instance, with its own library and data


def load_settings(path: Path | None = None) -> Settings:
    path = path or Path(os.environ.get(CONFIG_ENV) or ROOT / "config.yaml")
    values = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    values = values or {}
    known = {f.name for f in fields(Settings)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown settings in {path}: {sorted(unknown)}")
    for key in ("library_dir", "data_dir", "outputs_dir"):
        if key in values:
            values[key] = (ROOT / values[key]).resolve()
    for key in ("backup_dir", "backup_local_dir", "office_fonts_dir"):
        if values.get(key) is not None:
            values[key] = (ROOT / Path(values[key]).expanduser()).resolve()
    if "answer_options" in values:
        values["answer_options"] = answer_options(values["answer_options"], path)
    if "figure_describer" in values:
        entry = dict(values["figure_describer"])
        if entry.get("expected_tokens") is not None:
            entry["expected_tokens"] = tuple(entry["expected_tokens"])
        values["figure_describer"] = FigureDescriber(**entry)
    if "hosted_prices" in values:
        values["hosted_prices"] = {model: price_of(entry) for model, entry in values["hosted_prices"].items()}
    settings = Settings(**values)
    check_options(settings, path)
    return settings


def answer_options(raw: list[dict], path: Path) -> list[AnswerOption]:
    known = {f.name for f in fields(AnswerOption)}
    options = []
    for entry in raw or []:
        unknown = set(entry) - known
        if unknown:
            raise ValueError(f"Unknown answer option settings in {path}: {sorted(unknown)}")
        if entry.get("expected_tokens") is not None:
            entry = {**entry, "expected_tokens": tuple(entry["expected_tokens"])}
        if entry.get("web") is False:       # YAML reads a bare off as false
            entry = {**entry, "web": "off"}
        options.append(AnswerOption(**entry))
    return options


def check_options(settings: Settings, path: Path) -> None:
    """Mistakes here would only show up as a failed question, so they stop the start instead."""
    describer = settings.figure_describer
    if describer.provider not in ("ollama", "anthropic"):
        raise ValueError(f"figure_describer: provider must be ollama or anthropic, not {describer.provider}")
    if describer.hosted and describer.model not in settings.hosted_prices:
        raise ValueError(f"figure_describer: no price for {describer.model}, so its cost couldn't be logged")
    options = settings.answer_options
    if not options:
        raise ValueError(f"{path} lists no answer options")
    ids = [option.id for option in options]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Answer option ids repeat in {path}: {ids}")
    for option in options:
        if option.provider not in PROVIDERS:
            raise ValueError(f"Answer option {option.id}: provider must be one of {PROVIDERS}")
        if option.web not in WEB_MODES:
            raise ValueError(f"Answer option {option.id}: web must be one of {WEB_MODES}, not {option.web!r}")
        if option.num_predict is not None and (option.provider != "ollama" or option.num_predict <= 0):
            raise ValueError(f"Answer option {option.id}: num_predict is for a local (ollama) option, above 0")
        if option.num_predict is not None and option.num_predict >= settings.chat_num_ctx:
            raise ValueError(f"Answer option {option.id}: num_predict must leave room in chat_num_ctx "
                             f"({settings.chat_num_ctx}) for the prompt")
        if option.provider == "openrouter" and not option.route:
            raise ValueError(f"Answer option {option.id}: an OpenRouter option needs a route to pin")
        for model in (option.model, option.fallback):
            if option.hosted and model and model not in settings.hosted_prices:
                raise ValueError(f"Answer option {option.id}: no price for {model}, so its cost couldn't be logged")
