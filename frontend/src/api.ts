export interface Region {
  page: number;
  bbox: [number, number, number, number];
}

export interface Source {
  id: number;
  chunk_id: number;
  doc_id: number;
  doc_title: string;
  page_start: number | null;
  page_end: number | null;
  focus_page: number | null;
  heading_path: string[];
  regions: Region[];
  /** The figures inside the passage, for showing next to an answer that cites it. */
  figures?: SourceFigure[];
  /** "note" for one of the reader's notes (roadmap item 4): doc_id is 0 when it has no book. */
  kind?: string;
  note_id?: number | null;
}

interface SourceFigure {
  url: string;
  page: number;
  caption: string;
}

export interface UnverifiedCrop {
  source: number;
  doc_id: number;
  page: number;
  url: string;
  cited: boolean;
}

export interface OutputFile {
  name: string;
  kind: string;
  size: number;
  url: string;
  /** The source text, for files a browser can't show itself. */
  preview: string | null;
  /** Word and PowerPoint: the file as LibreOffice renders it, a PDF (roadmap item 7). */
  render_url?: string | null;
  pages?: number;
  slides?: number;
}

/** Where a document is in the queued flow (app/library.py). */
type DocumentState =
  | "queued" | "parsing" | "indexing" | "describing" | "summarizing" | "done" | "failed" | "removed";

/** A stage that waits for the reader's OK before it spends money. */
interface Waiting {
  stage: "describe";
  figures: number;
  usd: number | null;
  model: string;
  provider: string;
}

export interface DocumentRow {
  id: number;
  path: string;
  /** From the document's metadata, never its file name. */
  title: string;
  author: string | null;
  year: number | null;
  publisher?: string | null;
  kind: string;
  pages: number | null;
  status: string;
  chunks: number;
  state?: DocumentState;
  searchable?: boolean;
  waiting?: Waiting | null;
  progress?: { done: number; total: number } | null;
  error?: string | null;
  failed_stage?: string | null;
  /** "test": the books the evaluation suites use, which a library setting can hide. */
  collection?: string | null;
  /** No metadata.yaml entry yet: the pane asks for one. */
  needs_details?: boolean;
}

/** What removing a gone document deletes (GET /api/documents/{id}/removal). */
export interface RemovalPreview {
  title: string;
  chunks: { text: number; figure: number; summary: number };
  blocks: number;
  figures: number;
  jobs: number;
  annotations: number;
  notes_kept: number;
  figure_descriptions: number;
  figure_descriptions_usd: number;
  summary_cache: number;
  summary_cache_seconds: number;
  folders: string[];
  folder_bytes: number;
}

export interface DocumentDetails {
  title: string | null;
  author: string | null;
  year: number | null;
  publisher: string | null;
}

export interface SseEvent {
  event: string;
  data: any;
}

/** Splits a stream into events; a network read can end anywhere, even mid-event. */
export class SseParser {
  private buffer = "";

  feed(chunk: string): SseEvent[] {
    this.buffer += chunk;
    const blocks = this.buffer.split("\n\n");
    this.buffer = blocks.pop() ?? "";
    return blocks.flatMap((block) => {
      let event = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      return data ? [{ event, data: JSON.parse(data) }] : [];
    });
  }
}

/** An answer option's id from config.yaml: fast, balanced, qwen or deep there. */
export type Mode = string;

/** One choice in the answer selector, as GET /api/options gives it. */
export interface AnswerOption {
  id: string;
  label: string;
  provider: "ollama" | "openrouter" | "anthropic";
  model: string;
  fallback: string | null;
  /** The one provider OpenRouter may send it to. */
  route: string | null;
  hosted: boolean;
  /** The measured median per answer, and what it was measured on. */
  expected_seconds: number | null;
  expected_note: string | null;
  /** For a local option: whether its model is in memory now. Null if unknown or hosted. */
  loaded: boolean | null;
  /** Reads images: figures and screenshots go to it as images. */
  vision: boolean;
  /** Web search: the model searches itself ("tools"), the app searches for it ("app"), or none ("off"). */
  web?: "tools" | "app" | "off";
  /** With Check on: how many answers it may write until one's claims all pass (1: badges only),
   * and the longest the question may take. */
  check_samples?: number;
  check_ceiling_seconds?: number | null;
  /** "limited": few of its claims can be checked, so the badge is neutral and says so (Fast, Balanced).
   * "unmeasured": its checks haven't been measured, so the badge is neutral and says so (Qwen). */
  check_badge?: "full" | "limited" | "unmeasured";
  /** For a hosted option: the measured mean tokens per question at today's price. */
  expected_usd: number | null;
  price: { input: number; output: number; checked: string | null } | null;
}

export interface MonthSpend {
  provider: "anthropic" | "openrouter";
  usd: number;
  limit_usd: number | null;
}

/** What the log says one hosted answer cost: every call it made, failed ones included. */
export interface AnswerSpend {
  calls: number;
  failed_calls: number;
  cost_usd: number;
  input_tokens: number;
  output_tokens: number;
}

export interface OptionList {
  options: AnswerOption[];
  default: string;
  /** Measured range for switching between local models. */
  switch_seconds: [number, number];
  month_spend: MonthSpend[];
  /** The options that may suggest highlights; the first is the default (roadmap item 4). */
  suggest_options?: string[];
}

/** An image the reader pasted or uploaded, for one question. */
export interface ReaderImage {
  media_type: string;
  data: string;     // base64, without the data: prefix
}

export interface ChatBody {
  message: string;
  conversation_id: number | null;
  doc_ids: number[] | null;
  mode: Mode;
  images?: ReaderImage[];
  /** Web search on for this question only (Phase 5d). */
  web?: boolean;
  /** A search the model proposed and the reader agreed to. */
  web_query?: string;
  /** Check on for this question: its claims are checked with SymPy in the sandbox. */
  check?: boolean;
  /** The Notes scope: the reader's notes are searched, not the books. */
  notes_only?: boolean;
}

export type Verdict = "passed" | "failed" | "numerically_equal" | "undecided" | "error";

/** One claim from an answer's claims block, as the checker judged it. */
interface CheckedClaim {
  line: string;
  label: string;
  verdict: Verdict;
  /** The verdict as the reader reads it: "passed (to stated digits)", "≈ (no proof)" ... */
  shown: string;
  note: string;
}

/** The checks on an answer: "N of M checked claims passed", never "correct". */
export interface CheckReport {
  badge: string;
  /** Who wrote the claims: the answer itself, or the model again afterwards ("extracted"). */
  source: "answer" | "extracted" | "none";
  failure: string | null;
  claims: CheckedClaim[];
  /** The claims block as the model wrote it, shown only as text. */
  block: string | null;
  /** Which attempt is shown, how many were made, and of how many allowed. */
  attempt?: number;
  attempts?: number;
  samples?: number;
  /** Why no further attempt was made: "time" (the ceiling) or "error". */
  stopped?: string | null;
  /** Every attempt had a failed claim; the one with the fewest is shown. */
  none_passed?: boolean;
}

/** An attempt another one replaced, kept for the reader with the claims that failed. */
export interface ReplacedAttempt {
  attempt: number;
  answer: string;
  checks: CheckReport;
  failing: string[];
}

/** A web page an answer read: cited as [Web n]. */
export interface WebSource {
  number: number;
  title: string;
  url: string;
  fetched: string;
}

/** A search the model asks the reader to allow, with web search off. */
export interface WebProposal {
  query: string;
  reason: string;
}

/** POST, so it can't use EventSource. Aborting the fetch stops the model too. */
export async function streamChat(
  body: ChatBody,
  onEvent: (event: SseEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) {
    throw new Error(`The server answered ${response.status}`);
  }
  const parser = new SseParser();
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) return;
    parser.feed(value).forEach(onEvent);
  }
}

export interface LibrarySettings {
  /** Leave the test collection out of whole-library search and the library pane. */
  hide_test_collection: boolean;
}

export async function fetchLibrary(): Promise<{ documents: DocumentRow[]; settings: LibrarySettings; metadataProblem: string | null }> {
  const response = await fetch("/api/documents");
  if (!response.ok) throw new Error(`The server answered ${response.status}`);
  const body = await response.json();
  return { documents: body.documents, settings: body.settings ?? { hide_test_collection: false },
    metadataProblem: body.metadata_problem ?? null };
}

export async function saveLibrarySettings(settings: Partial<LibrarySettings>): Promise<LibrarySettings> {
  const response = await fetch("/api/library/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings),
  });
  if (!response.ok) throw new Error(`The server answered ${response.status}`);
  return response.json();
}

export async function fetchOptions(): Promise<OptionList> {
  const response = await fetch("/api/options");
  if (!response.ok) throw new Error(`The server answered ${response.status}`);
  return response.json();
}

/** The server's reason, from FastAPI's {"detail": ...}, or the status. */
async function failure(response: Response): Promise<Error> {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return new Error(body.detail);
  } catch {
    // not JSON
  }
  return new Error(`The server answered ${response.status}`);
}

async function post(url: string, body?: unknown): Promise<any> {
  const response = await fetch(url, {
    method: "POST",
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) throw await failure(response);
  return response.json();
}

/** Agree to describe a document's figures for the estimate shown. */
export function approveFigures(docId: number, usd: number): Promise<unknown> {
  return post(`/api/documents/${docId}/figures/approve`, { usd });
}

/** Leave this version's figures undescribed: they're found by their captions only. */
export function skipFigures(docId: number): Promise<unknown> {
  return post(`/api/documents/${docId}/figures/skip`);
}

export async function fetchDetails(docId: number): Promise<DocumentDetails> {
  const response = await fetch(`/api/documents/${docId}/details`);
  if (!response.ok) throw await failure(response);
  return response.json();
}

export async function fetchRemoval(docId: number): Promise<RemovalPreview> {
  const response = await fetch(`/api/documents/${docId}/removal`);
  if (!response.ok) throw await failure(response);
  return response.json();
}

/** Delete everything made from a document whose file has left library/. */
export function removeDocument(docId: number): Promise<{ removed: RemovalPreview }> {
  return post(`/api/documents/${docId}/remove`);
}

/** Add a PDF or a deck to library/, as it is and under its own name. The server checks it and
 * never overwrites a file; the worker then reads it as if it had been copied in. */
export async function uploadDocument(file: File): Promise<{ name: string; size: number }> {
  const response = await fetch(`/api/library/files/${encodeURIComponent(file.name)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/octet-stream" },
    body: file,
  });
  if (!response.ok) throw await failure(response);
  return response.json();
}

/** Write the document's metadata.yaml entry. */
export function saveDetails(docId: number, details: DocumentDetails): Promise<DocumentDetails> {
  return post(`/api/documents/${docId}/details`, details);
}

/** A source as the feedback list keeps it: enough to show it and open its page. */
export interface FeedbackSource extends Pick<Source, "id" | "doc_id" | "doc_title" | "page_start" | "page_end" |
  "focus_page" | "heading_path" | "regions"> {
  cited: boolean;
}

interface FeedbackWebSource {
  number: number;
  title: string;
  url: string;
  fetched: string | null;
  cited: boolean;
}

export type Rating = "up" | "down";

/** The reader's rating of one answer, with a copy of the turn taken when it was rated. */
export interface FeedbackEntry {
  id: number;
  question_id: number | null;
  answer_id: number | null;
  rating: Rating;
  note: string;
  status: "open" | "done";
  /** A thumbs-down marked as a gold-question candidate (data/gold_candidates.yaml). */
  candidate: boolean;
  question: string;
  answer: string;
  /** Why the answer failed, if it did. */
  error: string | null;
  option: string | null;
  model: string | null;
  hosted: boolean;
  served_by: string | null;
  sources: FeedbackSource[];
  web_sources: FeedbackWebSource[];
  /** UTC, "YYYY-MM-DD HH:MM:SS", as SQLite writes it. */
  asked_at: string;
  answered_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface FeedbackCounts {
  total: number;
  up: number;
  down: number;
  open: number;
  candidates: number;
}

async function send(method: string, url: string, body: unknown): Promise<any> {
  const response = await fetch(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw await failure(response);
  return response.json();
}

/** Thumbs up or down on the answer to a question, with an optional note. Rating again changes it. */
export function rateAnswer(questionId: number, rating: Rating, note = ""): Promise<FeedbackEntry> {
  return send("PUT", `/api/questions/${questionId}/feedback`, { rating, note });
}

export async function fetchFeedback(
  filter: { rating?: Rating; status?: "open" | "done" } = {},
): Promise<{ entries: FeedbackEntry[]; counts: FeedbackCounts }> {
  const query = new URLSearchParams(Object.entries(filter).filter(([, value]) => value) as [string, string][]);
  const response = await fetch(`/api/feedback${query.size ? `?${query}` : ""}`);
  if (!response.ok) throw await failure(response);
  return response.json();
}

/** Mark an entry done or open again, or a thumbs-down a gold candidate or not. */
export function changeFeedback(
  id: number, change: { status?: "open" | "done"; candidate?: boolean },
): Promise<FeedbackEntry> {
  return send("PATCH", `/api/feedback/${id}`, change);
}

// ---- Highlights in the viewer (roadmap item 4) ----

export type HighlightColor = "yellow" | "green" | "blue" | "pink";
export const HIGHLIGHT_COLORS: HighlightColor[] = ["yellow", "green", "blue", "pink"];

/** A highlight: one box per line, in PDF points from the page's top left, and its quote. */
export interface Annotation {
  id: number;
  doc_id: number;
  page: number;
  rects: [number, number, number, number][];
  bbox: [number, number, number, number];
  quote: string;
  prefix: string;
  suffix: string;
  kind: "highlight";
  color: HighlightColor;
  body: string | null;
  origin: "user" | "ai";
  status: "suggested" | "accepted";
  created_at: string;
  /** Only on a new one: whether the server matched the selection to the page's words. */
  snapped?: boolean;
  /** After the book's file changed: found again ("ok", "moved") or not ("unanchored"). */
  anchor_status?: "ok" | "moved" | "unanchored";
  /** Only in the list of lost highlights. */
  doc_title?: string;
  /** AI suggestions made together, and what the reader asked for (roadmap item 4, step 5). */
  batch?: string | null;
  instruction?: string | null;
}

export interface SuggestionResult {
  batch: string;
  suggestions: Annotation[];
  not_found: { page: number; quote: string; why: string }[];
  asked: number;
  cost_usd: number;
  option: string;
}

/** Ask a model for suggested highlights on pages first..last, for any instruction. */
export function suggestHighlights(body: {
  doc_id: number; first_page: number; last_page: number; instruction: string; option?: string;
}): Promise<SuggestionResult> {
  return post("/api/suggestions", body);
}

/** Accept or reject every suggestion still waiting in a batch. */
export function batchAction(batch: string, action: "accept" | "reject"): Promise<{ changed: number }> {
  return post(`/api/annotations/batch/${batch}`, { action });
}

export interface AnnotatedCopy {
  name: string;
  url: string;
  highlights: number;
  notes: number;
  suggested_left_out: number;
  lost_left_out: number;
}

/** Write the annotated copy (outputs/annotated/); the original is only read. */
export function exportAnnotated(docId: number): Promise<AnnotatedCopy> {
  return post(`/api/annotations/export/${docId}`);
}

export async function fetchAnnotations(docId: number): Promise<Annotation[]> {
  const response = await fetch(`/api/annotations?doc_id=${docId}`);
  if (!response.ok) throw await failure(response);
  return (await response.json()).annotations;
}

/** Save a selection as a highlight; the server snaps its boxes to the page's words. */
export function createAnnotation(body: {
  doc_id: number; page: number; rects: number[][]; quote: string; color?: HighlightColor;
}): Promise<Annotation> {
  return post("/api/annotations", body);
}

export function changeAnnotation(
  id: number, change: { color?: HighlightColor; body?: string; status?: "suggested" | "accepted" },
): Promise<Annotation> {
  return send("PATCH", `/api/annotations/${id}`, change);
}

/** Highlights that lost their passage when their book's file changed. */
export async function fetchLostAnnotations(): Promise<Annotation[]> {
  const response = await fetch("/api/annotations/lost");
  if (!response.ok) throw await failure(response);
  return (await response.json()).annotations;
}

export async function deleteAnnotation(id: number): Promise<void> {
  const response = await fetch(`/api/annotations/${id}`, { method: "DELETE" });
  if (!response.ok) throw await failure(response);
}

// ---- Notes (roadmap item 4) ----

/** A passage selected in the viewer: what a note (or a highlight) is anchored to. */
export interface Passage {
  doc_id: number;
  page: number;
  rects: number[][];
  quote: string;
}

export interface Note {
  id: number;
  /** null for a note written without a passage, or whose book was removed. */
  doc_id: number | null;
  /** The book's title when the note was written; kept if the book is removed. */
  doc_title: string;
  page: number | null;
  rects: [number, number, number, number][];
  bbox: [number, number, number, number] | null;
  quote: string;
  prefix: string;
  suffix: string;
  body_md: string;
  tags: string[];
  created_at: string;
  updated_at: string;
  snapped?: boolean | null;
  /** null for a note with no passage; "unanchored" when its passage wasn't found after the book changed. */
  anchor_status?: "ok" | "moved" | "unanchored" | null;
}

export async function fetchNotes(): Promise<Note[]> {
  const response = await fetch("/api/notes");
  if (!response.ok) throw await failure(response);
  return (await response.json()).notes;
}

export function createNote(body: { body_md: string; tags: string[]; passage: Passage | null }): Promise<Note> {
  return post("/api/notes", body);
}

export function changeNote(id: number, change: { body_md?: string; tags?: string[]; passage?: Passage }): Promise<Note> {
  return send("PATCH", `/api/notes/${id}`, change);
}

export async function deleteNote(id: number): Promise<void> {
  const response = await fetch(`/api/notes/${id}`, { method: "DELETE" });
  if (!response.ok) throw await failure(response);
}

/** The Obsidian export: a zip of Markdown files, one per document. */
export function notesExportUrl(docId: number | null): string {
  return docId === null ? "/api/notes/export" : `/api/notes/export?doc_id=${docId}`;
}

// ---- Explain (roadmap item 4, step 6) ----

export interface ExplainItem {
  span: string;
  explanation: string;
}

export interface ExplainResult {
  kind: "equation" | "code";
  depth: "intro" | "intermediate" | "expert";
  text: string;
  option: string;
  cost_usd: number;
  items: ExplainItem[];
  /** From the viewer: the book's own LaTeX was used, not the flattened selected text. */
  from_book?: boolean;
}

export interface ExplainBody {
  text?: string;
  kind?: "equation" | "code";
  depth?: string;
  option?: string;
  passage?: Passage;
}

/** A line-by-line explanation: the text cut into spans, each with its explanation. */
export function explainText(body: ExplainBody): Promise<ExplainResult> {
  return post("/api/explain", body);
}
