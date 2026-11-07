import { useCallback, useEffect, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import type {
  AnswerOption, AnswerSpend, CheckReport, MonthSpend, OutputFile, Rating, ReaderImage, ReplacedAttempt, Source, Verdict,
} from "../api";
import { Markdown } from "../render/Markdown";
import { FeedbackPane } from "./FeedbackPane";
import { NotesPane } from "./NotesPane";
import { badgeNote, limitedBadge, modelBadge, offersDeepResend, useStore } from "../store/store";
import type { Message, ScopeMode } from "../store/store";
import { pageLabel } from "../viewer/highlight";

function SourceList({ sources, cited }: { sources: Source[]; cited: number[] }) {
  const openSource = useStore((state) => state.openSource);
  const showNote = useStore((state) => state.showNote);
  const used = cited.flatMap((id) => sources.filter((source) => source.id === id));
  if (used.length === 0) return null;
  return (
    <div className="mt-4 border-t border-stone-200 pt-3">
      <h3 className="text-xs font-medium text-stone-500">Sources</h3>
      <ul className="mt-1.5 space-y-1">
        {used.map((source) => (
          <li key={source.id}>
            <button
              type="button"
              className="flex w-full gap-2 rounded px-1 py-0.5 text-left text-sm hover:bg-stone-100"
              data-testid="source"
              data-kind={source.kind ?? "text"}
              onClick={() => (isNote(source) && (!source.doc_id || source.page_start === null)
                ? showNote(source.note_id as number) : openSource(source))}
            >
              <span className="shrink-0 font-medium text-teal-800">S{source.id}</span>
              <span className={`min-w-0 truncate ${isNote(source) ? "text-teal-900" : "text-stone-700"}`}>
                {source.doc_title}{source.page_start !== null && ` · ${pageLabel(source)}`}
                {source.heading_path.length > 0 && ` · ${source.heading_path[source.heading_path.length - 1]}`}
              </span>
            </button>
            {isNote(source) && source.doc_id > 0 && source.page_start !== null && (
              <button type="button" className="ml-7 text-xs text-teal-800 hover:underline" data-testid="source-open-note"
                onClick={() => showNote(source.note_id as number)}>
                Open the note
              </button>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

/** The explanations asked for in this session, each reopening its two-column panel over the viewer. */
function ExplanationLinks() {
  const explanations = useStore((state) => state.explanations);
  const show = useStore((state) => state.showExplanation);
  if (explanations.length === 0) return null;
  return (
    <ul className="space-y-1 border-t border-stone-200 pt-2 text-xs" data-testid="explanation-links">
      {explanations.map((entry) => (
        <li key={entry.id}>
          <button type="button" className="text-left text-teal-800 hover:underline" onClick={() => show(entry.id)}>
            Explanation ({entry.depth}): {entry.label}
            {entry.error ? " (failed)" : entry.result ? "" : " (working…)"}
          </button>
        </li>
      ))}
    </ul>
  );
}

const isNote = (source: Source) => source.kind === "note" && typeof source.note_id === "number";

/** The figures of the sources an answer cites: each source's nearest one, the rest on request.
 * A passage merged from neighbours can hold many; the server lists the one it was found by first. */
function hostOf(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

/** The web pages an answer read, apart from the library's sources, each with its link. */
function WebSourceList({ message }: { message: Message }) {
  if (message.webSources.length === 0) return null;
  const cited = new Set(message.webCitations);
  return (
    <div className="mt-4 border-t border-stone-200 pt-3" data-testid="web-sources">
      <h3 className="text-xs font-medium text-stone-500">Web sources</h3>
      <ul className="mt-1.5 space-y-1">
        {message.webSources.map((page) => (
          <li key={page.number} className="flex gap-2 px-1 py-0.5 text-sm">
            <span className={`shrink-0 font-medium ${cited.has(page.number) ? "text-sky-800" : "text-stone-400"}`}>
              Web {page.number}
            </span>
            <a href={page.url} target="_blank" rel="noreferrer noopener" className="min-w-0 truncate text-stone-700 hover:underline"
               title={page.url}>
              {page.title} · {hostOf(page.url)}
            </a>
            <span className="shrink-0 text-xs text-stone-400">read {page.fetched}</span>
          </li>
        ))}
      </ul>
      {message.finished && message.webSources.length > cited.size && (
        <p className="mt-1 text-xs text-stone-500">Grey: read but not cited.</p>
      )}
    </div>
  );
}

/** With web search off, the model may ask; nothing goes to the web until the reader clicks. */
function WebProposalCard({ message }: { message: Message }) {
  const searchWeb = useStore((state) => state.searchWeb);
  const sending = useStore((state) => state.sending);
  const proposal = message.webProposal;
  if (!proposal || message.streaming) return null;
  return (
    <div className="mt-3 rounded border border-sky-200 bg-sky-50 p-2 text-sm" data-testid="web-proposal">
      <p className="text-stone-700">{proposal.reason}</p>
      <div className="mt-1.5 flex flex-wrap items-center gap-2">
        <button
          type="button"
          disabled={sending}
          onClick={() => void searchWeb(message.id)}
          className="rounded bg-sky-800 px-2 py-1 text-xs font-medium text-white disabled:opacity-40"
        >
          Search the web for “{proposal.query}”
        </button>
        <span className="text-xs text-stone-500">
          Asks this question again with web search, through your SearXNG.
        </span>
      </div>
    </div>
  );
}

function CitedFigures({ message }: { message: Message }) {
  const openPage = useStore((state) => state.openPage);
  const [all, setAll] = useState(false);
  const cited = message.citations.flatMap((id) => message.sources.filter((source) => source.id === id));
  const every = cited.flatMap((source) => (source.figures ?? []).map((figure, place) => ({ source, figure, place })));
  const shown = all ? every : every.filter((item) => item.place === 0);
  if (every.length === 0) return null;
  return (
    <section className="mt-4" data-testid="cited-figures">
      <h3 className="text-xs font-medium text-stone-500">
        Figures in the cited sources
        {every.length > shown.length && (
          <button type="button" onClick={() => setAll(true)} data-testid="all-figures"
            className="ml-2 font-normal text-teal-800 hover:underline">
            Show all {every.length}
          </button>
        )}
      </h3>
      <ul className="mt-1.5 grid grid-cols-2 gap-2">
        {shown.map(({ source, figure }) => (
          <li key={figure.url}>
            <button
              type="button"
              className="block w-full rounded border border-stone-200 bg-white p-1 text-left hover:border-teal-700"
              onClick={() => openPage(source.doc_id, figure.page)}
            >
              <img src={figure.url} alt={figure.caption || `Figure on page ${figure.page}`} className="mx-auto max-h-56" />
              <span className="mt-1 block truncate text-xs text-stone-600">
                S{source.id} · page {figure.page}{figure.caption ? ` · ${figure.caption}` : ""}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

function UnverifiedCrops({ message }: { message: Message }) {
  const openPage = useStore((state) => state.openPage);
  if (message.unverified.length === 0) return null;
  return (
    <section className="mt-4 rounded border border-amber-300 bg-amber-50 p-3" data-testid="unverified">
      <h3 className="text-sm font-medium text-amber-900">Formulas shown from the page</h3>
      <p className="mt-0.5 text-xs text-amber-800">
        The sources for this answer contain formulas that couldn't be verified. If the answer uses
        one, trust the image from the page, not the LaTeX.
      </p>
      <ul className="mt-2 space-y-2">
        {message.unverified.map((crop) => (
          <li key={crop.url}>
            <button
              type="button"
              className="block w-full text-left"
              onClick={() => openPage(crop.doc_id, crop.page)}
            >
              <img
                src={crop.url}
                alt={`Formula on page ${crop.page}`}
                className="max-w-full rounded border border-amber-200 bg-white"
              />
              <span className="mt-0.5 block text-xs text-amber-800">
                S{crop.source} · page {crop.page}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

const TOOL_NAMES: Record<string, string> = {
  search_library: "Searching the library",
  make_markdown: "Making a Markdown file",
  make_pdf: "Making a PDF",
  make_docx: "Making a Word document",
  make_pptx: "Making a deck",
  run_python: "Running code in the sandbox",
  sympy_check: "Checking with SymPy",
  annotate_pdf: "Suggesting highlights",
};

function ToolActivity({ message }: { message: Message }) {
  const openPage = useStore((state) => state.openPage);
  if (message.tools.length === 0) return null;
  return (
    <ul className="mt-3 space-y-0.5 text-xs text-stone-500" data-testid="tool-activity">
      {message.tools.map((line) => (
        <li key={line.id} className={line.ok ? "" : "text-amber-800"}>
          {line.running ? `${TOOL_NAMES[line.name] ?? line.name}…` : line.summary}
          {!line.running && line.suggested && line.suggested.count > 0 && (
            <button type="button" className="ml-2 text-teal-800 underline" data-testid="open-suggestions"
              onClick={() => openPage(line.suggested!.doc_id, line.suggested!.page ?? 1)}>
              Accept or reject them in the viewer
            </button>
          )}
        </li>
      ))}
    </ul>
  );
}

function sizeLabel(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function fileFacts(file: OutputFile): string {
  const facts = [file.kind.toUpperCase(), sizeLabel(file.size)];
  if (file.pages) facts.push(`${file.pages} page${file.pages === 1 ? "" : "s"}`);
  if (file.slides) facts.push(`${file.slides} slide${file.slides === 1 ? "" : "s"}`);
  return facts.join(" · ");
}

const RENDERED_KINDS = ["docx", "pptx"];

/** Word and PowerPoint files go through LibreOffice on the server (roadmap item 7). Files
 *  from before it have no render_url, so it's derived from the file's own address. */
function renderUrl(file: OutputFile): string | null {
  if (!RENDERED_KINDS.includes(file.kind)) return null;
  return file.render_url ?? `${file.url}/preview`;
}

function SourceText({ file, note }: { file: OutputFile; note: string }) {
  return (
    <div className="max-h-96 overflow-y-auto rounded border border-stone-200 bg-white p-3">
      {file.kind !== "md" && <p className="mb-2 text-xs text-stone-500">{note}</p>}
      <Markdown text={file.preview ?? ""} streaming={false} onCite={() => undefined} />
    </div>
  );
}

type Rendered = { state: "loading" } | { state: "ready"; url: string } | { state: "failed"; reason: string };

function RenderedPreview({ file, url }: { file: OutputFile; url: string }) {
  const [rendered, setRendered] = useState<Rendered>({ state: "loading" });
  useEffect(() => {
    let objectUrl: string | null = null;
    let live = true;
    fetch(url)
      .then(async (response) => {
        if (!response.ok) {
          const body = await response.json().catch(() => null);
          throw new Error(typeof body?.detail === "string" ? body.detail : `HTTP ${response.status}`);
        }
        const blob = await response.blob();
        if (!live) return;      // closed while it loaded: a URL made now would never be revoked (R2-C10)
        objectUrl = URL.createObjectURL(blob);
        setRendered({ state: "ready", url: objectUrl });
      })
      .catch((error: Error) => {
        if (live) setRendered({ state: "failed", reason: error.message });
      });
    return () => {
      live = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [url]);
  if (rendered.state === "loading") {
    return (
      <p className="text-xs text-stone-500" role="status">
        Rendering with LibreOffice…
      </p>
    );
  }
  if (rendered.state === "ready") {
    return (
      <iframe
        src={rendered.url}
        title={`${file.name} as rendered`}
        data-testid="rendered-preview"
        className="h-96 w-full rounded border border-stone-200"
      />
    );
  }
  const note = `LibreOffice couldn't render it (${rendered.reason}). The content of the file instead; the layout in the file itself differs.`;
  if (file.preview) return <SourceText file={file} note={note} />;
  return <p className="text-xs text-stone-500">{note.split(" The content")[0]} No other preview.</p>;
}

function FilePreview({ file }: { file: OutputFile }) {
  if (file.kind === "pdf") {
    return <iframe src={file.url} title={file.name} className="h-96 w-full rounded border border-stone-200" />;
  }
  if (["png", "jpg", "jpeg"].includes(file.kind)) {
    return <img src={file.url} alt={file.name} className="max-w-full rounded border border-stone-200 bg-white" />;
  }
  const rendered = renderUrl(file);
  if (rendered) return <RenderedPreview file={file} url={rendered} />;
  if (file.preview) {
    return <SourceText file={file} note="The content of the file. The layout in the file itself differs." />;
  }
  return <p className="text-xs text-stone-500">No preview for this kind of file.</p>;
}

function FileCard({ file }: { file: OutputFile }) {
  const isPicture = ["png", "jpg", "jpeg"].includes(file.kind);
  const [open, setOpen] = useState(isPicture);
  return (
    <li className="rounded border border-stone-200 bg-stone-50 p-2" data-testid="file-card">
      <div className="flex items-center gap-2">
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium text-stone-800">{file.name}</p>
          <p className="text-xs text-stone-500">{fileFacts(file)}</p>
        </div>
        <button
          type="button"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          className="rounded px-2 py-1 text-xs text-stone-700 hover:bg-stone-200"
        >
          {open ? "Hide" : "Preview"}
        </button>
        <a
          href={`${file.url}?download=true`}
          download={file.name}
          className="rounded bg-teal-800 px-2 py-1 text-xs font-medium text-white"
        >
          Download
        </a>
      </div>
      {open && (
        <div className="mt-2" data-testid="file-preview">
          <FilePreview file={file} />
        </div>
      )}
    </li>
  );
}

function FileCards({ files }: { files: OutputFile[] }) {
  if (files.length === 0) return null;
  return (
    <ul className="mt-3 space-y-2">
      {files.map((file) => (
        <FileCard key={file.url} file={file} />
      ))}
    </ul>
  );
}

function ModelBadge({ message }: { message: Message }) {
  const text = modelBadge(message);
  if (!text) return null;
  const colours = message.fallbackFrom
    ? "bg-amber-200 text-amber-900"
    : message.hosted
      ? "bg-indigo-100 text-indigo-900"
      : "bg-stone-200 text-stone-700";
  return (
    <div className="mb-2">
      <span className={`rounded px-1.5 py-0.5 text-xs ${colours}`} data-testid="model-badge">
        {text}
      </span>
      {message.fallbackFrom && (
        <p className="mt-1 text-xs text-amber-800" data-testid="fallback-note">
          {message.fallbackFrom} failed ({message.fallbackReason}), so {message.model} wrote this answer.
        </p>
      )}
    </div>
  );
}

/** "17 s", or "about 3 min" once it's long enough that seconds read as false precision. */
export function duration(seconds: number): string {
  if (seconds < 90) return `${Math.round(seconds)} s`;
  return `about ${Math.round(seconds / 60)} min`;
}

/** Dollars with as many places as a small amount needs: $0.0014, $0.090, $15.84. */
export function dollars(usd: number): string {
  if (usd >= 1) return `$${usd.toFixed(2)}`;
  if (usd >= 0.01) return `$${usd.toFixed(3)}`;
  return `$${usd.toFixed(4)}`;
}

const PROVIDER_NAMES: Record<string, string> = { anthropic: "Anthropic", openrouter: "OpenRouter" };

function spendLine(spend: MonthSpend[]): string {
  return "Hosted this month: " + spend.map((item) =>
    `${PROVIDER_NAMES[item.provider] ?? item.provider} $${item.usd.toFixed(2)}` +
    (item.limit_usd ? ` of $${item.limit_usd.toFixed(0)}` : "")).join(" · ");
}

export function costLine(spend: AnswerSpend): string {
  const tokens = `${spend.input_tokens.toLocaleString("en-US")} in, ${spend.output_tokens.toLocaleString("en-US")} out tokens`;
  const calls = spend.calls > 1 ? ` · ${spend.calls} calls` : "";
  const failed = spend.failed_calls
    ? `, ${spend.failed_calls} of them failed, which ${spend.failed_calls === 1 ? "is" : "are"} included`
    : "";
  return `${dollars(spend.cost_usd)} for this answer · ${tokens}${calls}${failed}`;
}

/** A running clock: "42 s", "1:23". */
export function clock(seconds: number): string {
  const whole = Math.max(0, Math.floor(seconds));
  if (whole < 60) return `${whole} s`;
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

const SENT = "this question, the passages found for it, your previous question and answer with the " +
  "passages it cited, and what tools return: more passages, code output and SymPy results. Older turns " +
  "don't. Code and files still run and stay on this machine.";
const SENT_WITH_IMAGES = SENT.replace("this question, the passages found for it,", "this question and any image " +
  "you attach, the passages found for it with the images of any figures among them,");

const IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp", "image/gif"];
const IMAGE_BYTES = 5 * 1024 * 1024;
const IMAGES_MAX = 3;

/** An image file as the API takes it, or why it can't go. */
export function readImage(file: File): Promise<ReaderImage> {
  if (!IMAGE_TYPES.includes(file.type)) return Promise.reject(new Error("Only PNG, JPEG, WebP or GIF images can be sent."));
  if (file.size > IMAGE_BYTES) return Promise.reject(new Error(`That image is ${(file.size / 1e6).toFixed(1)} MB; the limit is 5 MB.`));
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve({ media_type: file.type, data: String(reader.result).split(",", 2)[1] ?? "" });
    reader.onerror = () => reject(new Error("The image couldn't be read."));
    reader.readAsDataURL(file);
  });
}

/** With an image attached and an option that can't read it: say so, and offer the ones that can. */
function ImageNotice({ count }: { count: number }) {
  const mode = useStore((state) => state.mode);
  const options = useStore((state) => state.options);
  const setMode = useStore((state) => state.setMode);
  const chosen = options.find((option) => option.id === mode);
  if (count === 0 || !chosen || chosen.vision) return null;
  const readers = options.filter((option) => option.vision);
  return (
    <div className="mb-2 rounded border border-amber-300 bg-amber-50 p-2 text-xs text-amber-900" data-testid="image-notice">
      {chosen.label} can't read images. Ask with
      {readers.map((option, index) => (
        <span key={option.id}>
          {index > 0 ? " or" : ""}{" "}
          <button type="button" onClick={() => setMode(option.id)} data-testid={`image-use-${option.id}`}
            className="rounded bg-white px-1.5 py-0.5 font-medium text-amber-900 ring-1 ring-amber-300 hover:bg-amber-100">
            {option.label}
          </button>
          {option.hosted ? " (goes to Anthropic)" : " (runs here)"}
        </span>
      ))}
      .
    </div>
  );
}

/** What leaves the machine, where to, and how long it usually takes. */
/** The mode note's one line: where it runs, how long, what it costs (item 9). optionNote is the detail. */
function optionLine(option: AnswerOption, switchSeconds: [number, number]): string {
  const usual = option.expected_seconds ? ` · usually ${duration(option.expected_seconds)}` : "";
  const cost = option.expected_usd !== null ? `, about ${dollars(option.expected_usd)}` : "";
  if (option.provider === "anthropic") return `Sent to Anthropic${usual}${cost}`;
  if (option.provider === "openrouter") return `Sent to OpenRouter, ${option.route} only${usual}${cost}`;
  const load = option.loaded === false ? ` · first answer +${switchSeconds[0]}–${switchSeconds[1]} s to load` : "";
  return `Runs on this machine${usual}${load}`;
}

function optionNote(option: AnswerOption, switchSeconds: [number, number]): string {
  const wait = option.expected_seconds
    ? ` Usually ${duration(option.expected_seconds)} an answer` +
      (option.expected_note ? ` (median, ${option.expected_note})` : "") +
      (option.expected_usd !== null ? ` and about ${dollars(option.expected_usd)} a question` : "") + "."
    : "";
  if (option.provider === "anthropic") {
    return `Goes to Anthropic: ${option.vision ? SENT_WITH_IMAGES : SENT}${wait}`;
  }
  if (option.provider === "openrouter") {
    return `Goes to OpenRouter, which may pass it only to ${option.route}, and to no provider that keeps ` +
      `or trains on prompts: ${SENT}${wait} If it stops without an answer, you can resend it in Deep; ` +
      "it never switches by itself.";
  }
  const switching = option.loaded === false
    ? ` ${option.model} isn't loaded, so the first answer waits ${switchSeconds[0]}–${switchSeconds[1]} s ` +
      "longer: only one local model fits in the GPU."
    : "";
  return `Runs on this machine.${wait}${switching}`;
}

/** A question that asks for code. Not "function" alone: in these books that's usually maths. */
const ASKS_FOR_CODE = /```|\b(code|python|numpy|scipy|pytorch|implement|implementation|script|program|snippet)\b/i;

export function asksForCode(text: string): boolean {
  return ASKS_FOR_CODE.test(text);
}

/** A question that asks for a file, read as the server reads it (prompts.TOOL_REQUESTS). "pdf"
 * alone isn't enough: in these books it usually means a probability density. */
const MAKE = String.raw`\b(?:make|create|write|generate|export|save|produce|build|erstelle|schreibe)\b`;
const ASKS_FOR_FILE = [
  /\b(?:deck|slides?|powerpoint|pptx|folien|präsentation)\b/i,
  /\b(?:word (?:document|doc|file)|docx)\b/i,
  new RegExp(String.raw`\b(?:as|in|into|to) (?:an? )?(?:\w+ )?pdf\b|${MAKE}.{0,60}\bpdf\b`, "i"),
  /\bmarkdown\b|\.md\b/i,
];

export function asksForFile(text: string): boolean {
  return ASKS_FOR_FILE.some((pattern) => pattern.test(text));
}

/** For a local option asked for code or a file. Measured on the 22 tool tasks. */
export function toolHint(text: string): string | null {
  const code = asksForCode(text);
  const file = asksForFile(text);
  if (!code && !file) return null;
  const subject = code && file ? "For code and files" : code ? "For code" : "For files";
  return `${subject}, Qwen and Deep are more reliable: they passed all 22 tool tasks, against 16 for Fast and 17 for Balanced.`;
}

function ModeSelector({ draft }: { draft: string }) {
  const mode = useStore((state) => state.mode);
  const setMode = useStore((state) => state.setMode);
  const options = useStore((state) => state.options);
  const optionsError = useStore((state) => state.optionsError);
  const switchSeconds = useStore((state) => state.switchSeconds);
  const monthSpend = useStore((state) => state.monthSpend);
  if (options.length === 0) {
    return optionsError ? (
      <p className="mb-2 text-xs text-amber-800">Couldn't load the answer options: {optionsError}</p>
    ) : null;
  }
  const chosen = options.find((option) => option.id === mode) ?? options[0];
  return (
    <div className="mb-2">
      <div className="flex items-center gap-2">
        <span className="text-xs text-stone-500">Answer</span>
        <div className="flex overflow-hidden rounded border border-stone-300" role="group" aria-label="Answer model">
          {options.map((option) => (
            <button
              key={option.id}
              type="button"
              onClick={() => setMode(option.id)}
              aria-pressed={chosen.id === option.id}
              title={`${option.model}${option.hosted ? " (hosted)" : " (local)"}`}
              data-testid={`mode-${option.id}`}
              className={`px-2.5 py-1 text-xs font-medium ${
                chosen.id === option.id ? "bg-teal-800 text-white" : "bg-white text-stone-700 hover:bg-stone-100"
              }`}
            >
              {option.label}
              {option.expected_usd !== null && (
                <span className="ml-1 font-normal opacity-75" data-testid={`cost-${option.id}`}>
                  ~{dollars(option.expected_usd)}
                </span>
              )}
            </button>
          ))}
        </div>
        <span className="min-w-0 truncate text-xs text-stone-500">{chosen.model}</span>
      </div>
      <div className="mt-1 flex items-center gap-1 text-xs text-stone-500" data-testid="mode-note">
        <span className="min-w-0 truncate" data-testid="mode-line">{optionLine(chosen, switchSeconds)}</span>
        {/* The detail on hover and on keyboard focus; Escape hides it again. */}
        <span className="group relative shrink-0">
          <button type="button" aria-describedby="mode-details" data-testid="note-details-button"
            className="rounded px-1 underline decoration-dotted hover:text-stone-800 focus:text-stone-800"
            onKeyDown={(event) => { if (event.key === "Escape") event.currentTarget.blur(); }}>
            Details
          </button>
          <span role="tooltip" id="mode-details" data-testid="note-details"
            className="invisible absolute bottom-full right-0 z-20 mb-1 w-80 rounded border border-stone-300 bg-white p-2 text-stone-700 shadow-lg group-focus-within:visible group-hover:visible">
            {optionNote(chosen, switchSeconds)}
          </span>
        </span>
      </div>
      {!chosen.hosted && toolHint(draft) && (
        <p className="mt-0.5 text-xs text-indigo-800" data-testid="tool-hint">{toolHint(draft)}</p>
      )}
      {monthSpend.length > 0 && (
        <p className="mt-0.5 text-xs text-stone-500" data-testid="month-spend">
          {spendLine(monthSpend)}
          {chosen.price?.checked && ` · ${chosen.label} prices checked ${chosen.price.checked}`}
        </p>
      )}
    </div>
  );
}

/** Web search for the next question only. It turns itself off after sending, so the web is
 * never searched without a choice made for that question. */
function WebToggle() {
  const on = useStore((state) => state.webNext);
  const setWebNext = useStore((state) => state.setWebNext);
  const web = useStore((state) => state.options.find((option) => option.id === state.mode)?.web ?? "tools");
  if (web === "off") return null;
  return (
    <div className="mb-2 flex items-center gap-2 text-xs">
      <button
        type="button"
        aria-pressed={on}
        onClick={() => setWebNext(!on)}
        data-testid="web-toggle"
        className={`rounded border px-2 py-0.5 font-medium ${
          on ? "border-sky-700 bg-sky-700 text-white" : "border-stone-300 text-stone-600 hover:bg-stone-100"
        }`}
      >
        Web
      </button>
      <span className="text-stone-500">
        {on
          ? "On for the next question: it may search the web through your SearXNG and read a few pages."
          : "Off: the library only. The model may propose a search, which you confirm."}
        {on && web === "app" && " The app runs the search and gives the model the top pages."}
      </span>
    </div>
  );
}

/** Check for the next question only, like Web: the answer's claims are checked with SymPy in
 * the sandbox, and an option that may try again does so when a claim fails. */
function CheckToggle() {
  const on = useStore((state) => state.checkNext);
  const setCheckNext = useStore((state) => state.setCheckNext);
  const option = useStore((state) => state.options.find((candidate) => candidate.id === state.mode));
  const samples = option?.check_samples ?? 1;
  const ceiling = option?.check_ceiling_seconds;
  return (
    <div className="mb-2 flex items-center gap-2 text-xs">
      <button
        type="button"
        aria-pressed={on}
        onClick={() => setCheckNext(!on)}
        data-testid="check-toggle"
        className={`rounded border px-2 py-0.5 font-medium ${
          on ? "border-teal-800 bg-teal-800 text-white" : "border-stone-300 text-stone-600 hover:bg-stone-100"
        }`}
      >
        Check
      </button>
      <span className="text-stone-500" data-testid="check-hint">
        {!on
          ? "Off: the answer's claims aren't checked."
          : samples > 1
            ? `On for the next question: the answer's claims are checked with SymPy, and if one fails it answers again (up to ${samples} attempts${ceiling ? `, at most ${Math.round(ceiling / 60)} minutes` : ""}).`
            : option?.check_badge === "limited"
              ? `On for the next question: the claims the answer writes down are checked with SymPy. Most of a ${option.label} answer isn't checkable.`
              : option?.check_badge === "unmeasured"
                ? `On for the next question: the claims the answer writes down are checked with SymPy. How well ${option.label}'s claims can be checked hasn't been measured yet.`
                : "On for the next question: the answer's claims are checked with SymPy and shown with it."}
      </span>
    </div>
  );
}

const VERDICT_STYLE: Record<Verdict, string> = {
  passed: "text-teal-800",
  failed: "text-red-800",
  numerically_equal: "text-stone-600",
  undecided: "text-stone-600",
  error: "text-amber-800",
};

/** "N of M checked claims passed", never "correct", with each claim and how it was judged. */
function CheckBadge({ checks, note = null }: { checks: CheckReport; note?: string | null }) {
  const failed = checks.claims.some((claim) => claim.verdict === "failed");
  const allPassed = checks.claims.length > 0 && checks.claims.every((claim) => claim.verdict === "passed");
  // Fast and Balanced: neutral, whatever the result, since most of their answers can't be checked
  // (round 19); Qwen: neutral until it's measured.
  const tone = note ? "border-stone-200 bg-stone-50 text-stone-700"
    : failed ? "border-red-200 bg-red-50 text-red-900"
    : allPassed ? "border-teal-200 bg-teal-50 text-teal-900" : "border-stone-200 bg-stone-50 text-stone-700";
  const attempts = checks.attempts ?? 1;
  return (
    <div className={`mt-3 rounded border p-2 text-sm ${tone}`} data-testid="check-badge">
      <details>
        <summary className="cursor-pointer select-none font-medium">
          <span data-testid="check-badge-text">
            {note && !checks.failure ? `${limitedBadge(checks.claims)} ${note}` : checks.badge}
          </span>
          {attempts > 1 && <span className="font-normal"> · attempt {checks.attempt} of {attempts}</span>}
        </summary>
        {checks.claims.length > 0 && (
          <ul className="mt-2 space-y-1.5" data-testid="check-claims">
            {checks.claims.map((claim, index) => (
              <li key={index}>
                <code className="break-words text-xs text-stone-800">{claim.line}</code>
                <span className={`ml-2 text-xs font-medium ${VERDICT_STYLE[claim.verdict]}`}>{claim.shown}</span>
                {claim.note && claim.verdict !== "passed" && (
                  <p className="text-xs text-stone-500">{claim.note}</p>
                )}
              </li>
            ))}
          </ul>
        )}
        <p className="mt-2 text-xs text-stone-500">
          The checks cover the claims the model wrote down, not the reasoning around them.
          {checks.source === "extracted" && " The model listed these claims after the answer."}
        </p>
      </details>
      {checks.none_passed && (
        <p className="mt-1 text-xs" data-testid="check-none-passed">
          None of the {attempts} attempts passed every check; this one had the fewest failed claims.
        </p>
      )}
      {checks.stopped === "time" && (
        <p className="mt-1 text-xs" data-testid="check-stopped">
          Stopped after {attempts} of {checks.samples} attempts: the time limit.
        </p>
      )}
      {checks.stopped === "error" && (
        <p className="mt-1 text-xs" data-testid="check-stopped">A later attempt failed; this is the best before it.</p>
      )}
      {checks.failure && <p className="mt-1 text-xs">{checks.failure}</p>}
    </div>
  );
}

/** The attempts another one replaced, collapsed, each with the claims that failed. */
function ReplacedAttempts({ replaced, onCite }: { replaced: ReplacedAttempt[]; onCite: (id: number) => void }) {
  if (replaced.length === 0) return null;
  return (
    <div className="mt-3 space-y-2" data-testid="replaced-attempts">
      {replaced.map((attempt) => (
        <details key={attempt.attempt} className="rounded border border-stone-200 p-2 text-sm text-stone-600">
          <summary className="cursor-pointer select-none">
            Attempt {attempt.attempt}, replaced: {attempt.checks.badge}
          </summary>
          <div className="mt-2 opacity-80">
            <Markdown text={attempt.answer} streaming={false} onCite={onCite} />
          </div>
        </details>
      ))}
      {replaced.map((attempt) =>
        attempt.failing.length > 0 && (
          <p key={`failing-${attempt.attempt}`} className="text-xs text-stone-500" data-testid="replaced-failing">
            Attempt {attempt.attempt} failed on:{" "}
            {attempt.failing.map((line, index) => (
              <code key={index} className="mr-2 text-red-800">{line}</code>
            ))}
          </p>
        ),
      )}
    </div>
  );
}

function ScopeSelector() {
  const scopeMode = useStore((state) => state.scopeMode);
  const setScopeMode = useStore((state) => state.setScopeMode);
  const ticked = useStore((state) => state.scope.length);
  const openDocument = useStore((state) => state.viewer?.docId ?? null);
  const title = useStore((state) => state.documents.find((d) => d.id === openDocument)?.title);
  const options: { value: ScopeMode; label: string; enabled: boolean; hint: string }[] = [
    { value: "library", label: "Library", enabled: true, hint: "Search every document" },
    {
      value: "document",
      label: "This document",
      enabled: openDocument !== null,
      hint: title ?? "Open a document in the viewer first",
    },
    {
      value: "ticked",
      label: `Ticked${ticked ? ` (${ticked})` : ""}`,
      enabled: ticked > 0,
      hint: ticked ? "Search the ticked documents" : "Tick documents in the library first",
    },
    { value: "notes", label: "Notes", enabled: true, hint: "Search only your notes, not the books" },
  ];
  // A choice that can no longer apply must not look active while the library is searched.
  const active = options.find((option) => option.value === scopeMode)?.enabled ? scopeMode : "library";
  return (
    <div className="mb-2 flex items-center gap-2">
      <span className="text-xs text-stone-500">Search</span>
      <div className="flex overflow-hidden rounded border border-stone-300" role="group" aria-label="Search scope">
        {options.map(({ value, label, enabled, hint }) => (
          <button
            key={value}
            type="button"
            onClick={() => setScopeMode(value)}
            disabled={!enabled}
            title={hint}
            aria-pressed={active === value}
            data-testid={`scope-${value}`}
            className={`px-2.5 py-1 text-xs font-medium disabled:cursor-not-allowed disabled:text-stone-400 ${
              active === value ? "bg-teal-800 text-white" : "bg-white text-stone-700 hover:bg-stone-100"
            }`}
          >
            {label}
          </button>
        ))}
      </div>
      {active === "document" && title && (
        <span className="min-w-0 truncate text-xs text-stone-500" data-testid="scope-note">{title}</span>
      )}
    </div>
  );
}

/** While no answer text has come: what's happening, with a clock. */
function Waiting({ message }: { message: Message }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);
  const elapsed = clock((now - message.startedAt) / 1000);
  const doing = !message.searched
    ? "Searching the library"
    : message.thinking
      ? "Thinking"
      : `Waiting for ${message.model ?? "the model"}`;
  return (
    <p className="text-sm text-stone-500" data-testid="waiting">
      {doing}… <span className="tabular-nums">{elapsed}</span>
    </p>
  );
}

function DeepResend({ message }: { message: Message }) {
  const resendDeep = useStore((state) => state.resendDeep);
  const sending = useStore((state) => state.sending);
  const label = useStore((state) => state.options.find((option) => option.id === message.option)?.label);
  if (!offersDeepResend(message)) return null;
  const name = label ?? message.option;
  return (
    <div className="mt-2 flex items-center gap-2 text-sm">
      <span className="text-stone-600">
        {message.cutOff
          ? `${name} ran out of room while thinking.`
          : message.error
            ? `${name} didn't answer.`
            : `${name}'s answer was cut off.`}
      </span>
      <button
        type="button"
        onClick={() => void resendDeep(message.id)}
        disabled={sending}
        data-testid="resend-deep"
        className="rounded bg-indigo-800 px-2 py-1 text-xs font-medium text-white disabled:opacity-40"
      >
        Ask again in Deep
      </button>
      <span className="text-xs text-stone-500">Sends it to Anthropic.</span>
    </div>
  );
}

/** Feedback can be given once the answer has finished or failed, and the question is stored. */
export function canRate(answer: Pick<Message, "streaming" | "finished" | "error" | "questionId">): boolean {
  return !answer.streaming && answer.questionId !== null && (answer.finished || answer.error !== null);
}

/** Thumbs up or down under an answer, with an optional note. A click saves at once; the
 * note is saved with its own button, and clicking the other thumb changes the rating. */
function FeedbackBar({ message }: { message: Message }) {
  const rate = useStore((state) => state.rate);
  const saved = message.feedback;
  const [note, setNote] = useState(saved?.note ?? "");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  if (!canRate(message)) return null;

  const save = async (rating: Rating) => {
    setBusy(true);
    setProblem(null);
    try {
      await rate(message.id, rating, note);
    } catch (error) {
      setProblem(`Not saved: ${(error as Error).message}`);
    } finally {
      setBusy(false);
    }
  };
  const thumb = (rating: Rating, symbol: string, label: string) => (
    <button
      type="button"
      aria-label={label}
      title={label}
      aria-pressed={saved?.rating === rating}
      disabled={busy}
      data-testid={`rate-${rating}`}
      onClick={() => void save(rating)}
      className={`rounded px-1.5 py-0.5 text-base leading-none disabled:opacity-40 ${
        saved?.rating === rating ? (rating === "up" ? "bg-teal-100" : "bg-red-100") : "opacity-60 hover:bg-stone-100 hover:opacity-100"
      }`}
    >
      {symbol}
    </button>
  );
  const noteChanged = saved !== null && note.trim() !== saved.note;
  return (
    <div className="mt-3 border-t border-stone-100 pt-2" data-testid="feedback-bar">
      <div className="flex items-center gap-1.5 text-xs text-stone-500">
        {thumb("up", "👍", "Good answer")}
        {thumb("down", "👎", "Bad answer")}
        {saved && !noteChanged && <span data-testid="feedback-saved">Saved to the feedback list.</span>}
      </div>
      {saved && (
        <div className="mt-1.5 flex items-start gap-2">
          <textarea
            value={note}
            onChange={(event) => setNote(event.target.value)}
            rows={1}
            maxLength={4000}
            aria-label="Feedback note"
            placeholder={saved.rating === "down" ? "What was wrong? (optional)" : "What was good? (optional)"}
            className="min-w-0 flex-1 resize-y rounded border border-stone-300 px-2 py-1 text-sm outline-none focus:border-teal-700"
          />
          <button
            type="button"
            disabled={busy || !noteChanged}
            onClick={() => void save(saved.rating)}
            data-testid="save-note"
            className="rounded border border-stone-300 px-2 py-1 text-xs text-stone-700 hover:bg-stone-100 disabled:opacity-40"
          >
            Save note
          </button>
        </div>
      )}
      {problem && <p className="mt-1 text-xs text-red-800" role="alert">{problem}</p>}
    </div>
  );
}

function Answer({ message }: { message: Message }) {
  const openSource = useStore((state) => state.openSource);
  const sources = message.sources;
  const onCite = useCallback(
    (id: number) => {
      const source = sources.find((candidate) => candidate.id === id);
      if (source) openSource(source);
    },
    [sources, openSource],
  );
  const note = useStore((state) => badgeNote(state.options.find((option) => option.id === message.option)));
  const explain = useStore((state) => state.explain);
  const explainSnippet = useCallback((text: string, kind: "equation" | "code") => {
    void explain({ text, kind }, `${kind} from an answer: ${text.split("\n")[0].slice(0, 50)}`);
  }, [explain]);
  const webSources = message.webSources;
  const webUrl = useCallback(
    (number: number) => webSources.find((page) => page.number === number)?.url,
    [webSources],
  );

  return (
    <article className="rounded-lg bg-white p-4 shadow-sm ring-1 ring-stone-200" data-testid="answer">
      <ModelBadge message={message} />
      {message.searchQuery && (
        <p className="mb-2 text-xs text-stone-500" data-testid="search-query">
          Searched for: {message.searchQuery}
        </p>
      )}
      {message.thinking && (
        <details className="mb-2 text-xs text-stone-500">
          <summary className="cursor-pointer select-none">
            {message.streaming && !message.text ? "Show the thinking so far" : "Reasoning"}
          </summary>
          <p className="mt-1 max-h-48 overflow-y-auto whitespace-pre-wrap">{message.thinking}</p>
        </details>
      )}
      {message.streaming && !message.text && !message.checkStatus && <Waiting message={message} />}
      {message.streaming && message.checkStatus && (
        <p className="mb-2 text-sm text-teal-800" data-testid="check-status">{message.checkStatus}</p>
      )}
      {message.web && (
        <p className="mb-2 text-xs text-sky-800" data-testid="web-on">Web search was on for this question.</p>
      )}
      {message.text && <Markdown text={message.text} streaming={message.streaming} onCite={onCite} webUrl={webUrl}
        onExplain={explainSnippet} />}
      {message.checks && <CheckBadge checks={message.checks} note={note} />}
      <ReplacedAttempts replaced={message.replaced ?? []} onCite={onCite} />
      <ToolActivity message={message} />
      <FileCards files={message.files} />
      {message.finished && message.sources.length > 0 && message.citations.length === 0 && (
        <p className="mt-3 text-sm text-amber-800" data-testid="uncited">
          This answer cites no sources, so check it against the book.
        </p>
      )}
      {message.truncated && (
        <p className="mt-3 text-sm text-amber-800">The answer hit the length limit and was cut off.</p>
      )}
      {message.error && (
        <p className="mt-2 rounded bg-red-50 p-2 text-sm text-red-800" role="alert">{message.error}</p>
      )}
      <DeepResend message={message} />
      {message.spend && message.spend.calls > 0 && (
        <p className="mt-2 text-xs text-stone-500" data-testid="answer-cost">{costLine(message.spend)}</p>
      )}
      <CitedFigures message={message} />
      <UnverifiedCrops message={message} />
      <WebProposalCard message={message} />
      <SourceList sources={message.sources} cited={message.citations} />
      <WebSourceList message={message} />
      <FeedbackBar message={message} />
    </article>
  );
}

export function ChatPane() {
  const messages = useStore((state) => state.messages);
  const sending = useStore((state) => state.sending);
  const send = useStore((state) => state.send);
  const stop = useStore((state) => state.stop);
  const newChat = useStore((state) => state.newChat);
  const view = useStore((state) => state.view);
  const setView = useStore((state) => state.setView);
  const [draft, setDraft] = useState("");
  const [images, setImages] = useState<ReaderImage[]>([]);
  const [imageError, setImageError] = useState<string | null>(null);
  const picker = useRef<HTMLInputElement>(null);
  const mode = useStore((state) => state.mode);
  const canSeeImages = useStore((state) => state.options.find((option) => option.id === state.mode)?.vision ?? false);
  const bottom = useRef<HTMLDivElement>(null);
  const log = useRef<HTMLDivElement>(null);
  const following = useRef(true);
  const questionBox = useRef<HTMLTextAreaElement>(null);
  const chatDraft = useStore((state) => state.chatDraft);

  // "Ask about this" in the viewer: the passage goes before whatever is typed already.
  useEffect(() => {
    if (!chatDraft) return;
    setDraft((current) => chatDraft.text + current);
    // After the box shows the new text: the caret goes to its end, where the question starts.
    const timer = window.setTimeout(() => {
      const box = questionBox.current;
      box?.focus();
      box?.setSelectionRange(box.value.length, box.value.length);
    }, 0);
    return () => window.clearTimeout(timer);
  }, [chatDraft]);

  // Follow the answer as it grows, unless the reader has scrolled up to read.
  useEffect(() => {
    if (following.current) bottom.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  const addImages = async (files: File[]) => {
    setImageError(null);
    for (const file of files) {
      if (images.length >= IMAGES_MAX) {
        setImageError(`At most ${IMAGES_MAX} images go with one question.`);
        return;
      }
      try {
        const image = await readImage(file);
        setImages((current) => (current.length < IMAGES_MAX ? [...current, image] : current));
      } catch (error) {
        setImageError((error as Error).message);
      }
    }
  };

  const blocked = images.length > 0 && !canSeeImages;
  const submit = () => {
    if (!draft.trim() || sending || blocked) return;
    following.current = true;
    void send(draft, mode, images);
    setDraft("");
    setImages([]);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  // The chat stays mounted behind the review list, so a half-typed question survives the visit.
  return (
    <>
    {view === "feedback" && <FeedbackPane />}
    {view === "notes" && <NotesPane />}
    <main className={`${view === "chat" ? "flex" : "hidden"} min-h-0 min-w-0 flex-col`} data-testid="chat">
      <header className="flex items-center justify-between border-b border-stone-200 px-4 py-3">
        <h2 className="text-sm font-semibold">Chat</h2>
        <div className="flex gap-4">
          <button type="button" onClick={() => setView("notes")} data-testid="open-notes"
            className="text-sm text-stone-600 hover:text-stone-900">
            Notes
          </button>
          <button type="button" onClick={() => setView("feedback")} data-testid="open-feedback"
            className="text-sm text-stone-600 hover:text-stone-900">
            Feedback
          </button>
          <button type="button" onClick={newChat} className="text-sm text-stone-600 hover:text-stone-900">
            New chat
          </button>
        </div>
      </header>
      <div
        ref={log}
        className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4"
        onScroll={() => {
          const element = log.current;
          if (element) {
            following.current = element.scrollHeight - element.scrollTop - element.clientHeight < 80;
          }
        }}
      >
        {messages.length === 0 && (
          <p className="pt-16 text-center text-sm text-stone-500">Ask something about your library.</p>
        )}
        {messages.map((message) =>
          message.role === "user" ? (
            <div key={message.id} className="ml-8 rounded-lg bg-teal-800 px-3 py-2 text-[15px] text-white">
              {message.images?.length > 0 && (
                <div className="mb-2 flex flex-wrap gap-2" data-testid="question-images">
                  {message.images.map((image, index) => (
                    <img key={index} src={`data:${image.media_type};base64,${image.data}`} alt="Your image"
                      className="max-h-40 rounded bg-white" />
                  ))}
                </div>
              )}
              <p className="whitespace-pre-wrap">{message.text}</p>
            </div>
          ) : (
            <Answer key={message.id} message={message} />
          ),
        )}
        <ExplanationLinks />
        <div ref={bottom} />
      </div>
      <footer className="border-t border-stone-200 p-3">
        <ScopeSelector />
        <ModeSelector draft={draft} />
        <WebToggle />
        <CheckToggle />
        <ImageNotice count={images.length} />
        {images.length > 0 && (
          <ul className="mb-2 flex flex-wrap gap-2" data-testid="attached-images">
            {images.map((image, index) => (
              <li key={index} className="relative">
                <img src={`data:${image.media_type};base64,${image.data}`} alt={`Attached image ${index + 1}`}
                  className="h-16 rounded border border-stone-300 bg-white" />
                <button type="button" aria-label={`Remove image ${index + 1}`}
                  onClick={() => setImages((current) => current.filter((_, i) => i !== index))}
                  className="absolute -right-1.5 -top-1.5 rounded-full bg-stone-700 px-1.5 text-xs text-white">×</button>
              </li>
            ))}
          </ul>
        )}
        {imageError && <p className="mb-2 text-xs text-amber-800" role="status">{imageError}</p>}
        <div className="flex items-end gap-2">
          <input ref={picker} type="file" accept={IMAGE_TYPES.join(",")} multiple hidden data-testid="image-picker"
            onChange={(event) => { void addImages([...(event.target.files ?? [])]); event.target.value = ""; }} />
          <button type="button" onClick={() => picker.current?.click()} title="Attach an image or screenshot"
            aria-label="Attach an image" className="rounded border border-stone-300 px-2 py-2 text-sm text-stone-600 hover:bg-stone-100">
            Image
          </button>
          <textarea
            ref={questionBox}
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onPaste={(event) => {
              const files = [...event.clipboardData.files].filter((file) => file.type.startsWith("image/"));
              if (files.length > 0) {
                event.preventDefault();
                void addImages(files);
              }
            }}
            onKeyDown={onKeyDown}
            rows={2}
            aria-label="Your question"
            placeholder="Ask a question, or ask for a file, a plot or a check. Enter sends, Shift+Enter starts a new line."
            className="min-w-0 flex-1 resize-none rounded border border-stone-300 bg-white px-3 py-2 text-[15px] outline-none focus:border-teal-700"
          />
          {sending ? (
            <button type="button" onClick={stop} className="rounded bg-stone-700 px-4 py-2 text-sm font-medium text-white">
              Stop
            </button>
          ) : (
            <button
              type="button"
              onClick={submit}
              disabled={!draft.trim() || blocked}
              className="rounded bg-teal-800 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
            >
              Send
            </button>
          )}
        </div>
      </footer>
    </main>
    </>
  );
}
