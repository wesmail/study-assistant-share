import { create } from "zustand";
import { explainText, fetchLibrary, fetchNotes, fetchOptions, rateAnswer, saveLibrarySettings, streamChat } from "../api";
import type {
  ExplainBody,
  ExplainResult,
  Note,
  Passage,
  AnswerOption, AnswerSpend, CheckReport, DocumentRow, FeedbackEntry, Mode, MonthSpend, OutputFile, Rating,
  ReaderImage, Region, ReplacedAttempt, Source, SseEvent,
  UnverifiedCrop, WebProposal, WebSource,
} from "../api";
import { pageOf } from "../viewer/highlight";

/** One tool call, as the chat shows it. */
interface ToolLine {
  id: string;
  name: string;
  running: boolean;
  ok: boolean;
  summary: string;
  /** annotate_pdf: where its suggestions are, to open them in the viewer. */
  suggested?: { doc_id: number; page: number | null; count: number } | null;
}

export interface Message {
  id: number;
  role: "user" | "assistant";
  text: string;
  thinking: string;
  sources: Source[];
  citations: number[];
  unverified: UnverifiedCrop[];
  error: string | null;
  truncated: boolean;
  streaming: boolean;
  finished: boolean;
  /** The model that actually answered, which a fallback can change mid-answer. */
  model: string | null;
  hosted: boolean;
  fallbackFrom: string | null;
  fallbackReason: string | null;
  /** What was searched for, when a follow-up was rewritten. */
  searchQuery: string | null;
  tools: ToolLine[];
  files: OutputFile[];
  /** The answer option it was asked with, and who served a routed model. */
  option: string | null;
  servedBy: string | null;
  /** The question it answers, so it can be asked again with another option. */
  question: string;
  /** When it was asked, for the elapsed time while it thinks. */
  startedAt: number;
  /** The search has returned, so a wait after this is the model. */
  searched: boolean;
  /** It stopped at its token limit while still thinking, with no answer. */
  cutOff: boolean;
  /** The model replied, but with no answer: cut off, empty, or only tool lines. */
  noAnswer: boolean;
  /** A hosted answer's cost, from the log. */
  spend: AnswerSpend | null;
  /** The reader's images sent with a question, shown in its bubble. */
  images: ReaderImage[];
  /** Asked with web search on (Phase 5d), and the pages it read and cited. */
  web: boolean;
  webSources: WebSource[];
  webCitations: number[];
  /** With web search off: a search the model asks the reader to allow. */
  webProposal: WebProposal | null;
  /** The stored question this answers, which feedback is about; from the "meta" event. */
  questionId: number | null;
  /** The reader's rating of this answer, once saved. */
  feedback: FeedbackEntry | null;
  /** Asked with Check on (verification layer): its claims checked with SymPy in the sandbox. */
  check: boolean;
  /** What the check is doing now: "Checking the claims…", "A claim failed: trying again (2/3)". */
  checkStatus: string | null;
  /** The checks on the answer shown, once done. */
  checks: CheckReport | null;
  /** Earlier attempts this one replaced, each with the claims that failed. */
  replaced: ReplacedAttempt[];
}

/** The neutral badge for an option whose answers are mostly not checkable (Fast, Balanced):
 * how many claims were checked and how they came out, never "correct". */
export function limitedBadge(claims: { verdict: string }[]): string {
  const total = claims.length;
  const passed = claims.filter((claim) => claim.verdict === "passed").length;
  if (total === 0) return "No claims could be checked.";
  const checked = `${total} claim${total === 1 ? "" : "s"} checked`;
  if (passed === total) {
    return `${checked}, ${total === 1 ? "it passed" : total === 2 ? "both passed" : "all passed"}.`;
  }
  return `${checked}, ${passed} passed.`;
}

const NOT_CHECKABLE = "Most of this answer isn't checkable.";

/** The note after a neutral badge, or null for the normal badge (Deep). */
export function badgeNote(option: { label: string; check_badge?: string } | undefined): string | null {
  if (option?.check_badge === "limited") return NOT_CHECKABLE;
  if (option?.check_badge === "unmeasured") return `How well ${option.label}'s claims can be checked hasn't been measured yet.`;
  return null;
}

/** The status line while Check runs, from the "check_status" and "attempt_replaced" events. */
export function checkingLine(attempt: number, of: number): string {
  return of > 1 ? `Checking the claims (attempt ${attempt} of ${of})…` : "Checking the claims…";
}

export function retryLine(next: number, of: number, failing: number): string {
  return `${failing === 1 ? "A claim" : `${failing} claims`} failed: trying again (${next}/${of})`;
}

/** How a question uses the web: off, on, or on for a search the reader agreed to. */
interface WebChoice {
  on: boolean;
  query?: string;
}

/** Where a question searches: everything, the document in the viewer, or the ticked ones. */
export type ScopeMode = "library" | "document" | "ticked" | "notes";

interface ViewerTarget {
  docId: number;
  page: number;
  regions: Region[];
  /** Changes on every open, so clicking the same citation twice still scrolls to it. */
  opened: number;
}

interface State {
  documents: DocumentRow[];
  libraryError: string | null;
  /** The reader's library setting: the test collection is left out of whole-library search and the pane. */
  hideTest: boolean;
  /** The ticked documents. */
  scope: number[];
  scopeMode: ScopeMode;
  messages: Message[];
  conversationId: number | null;
  /** The answer selector, in config order. */
  options: AnswerOption[];
  optionsError: string | null;
  switchSeconds: [number, number];
  /** The options that may suggest highlights, the default first. */
  suggestOptions: string[];
  monthSpend: MonthSpend[];
  /** A hosted choice applies to the next question only, so a hosted call is always a
   * choice; then the selector goes back to the last local choice, not always the first,
   * because switching local models reloads one. */
  mode: Mode;
  localMode: Mode;
  sending: boolean;
  /** Web search for the next question only: it turns itself off once the question is sent. */
  webNext: boolean;
  /** Check for the next question only, like Web. */
  checkNext: boolean;
  viewer: ViewerTarget | null;
  /** The middle pane: the chat, the feedback review list, or the notes. */
  view: "chat" | "feedback" | "notes";
  loadDocuments: () => Promise<void>;
  setHideTest: (hide: boolean) => Promise<void>;
  loadOptions: () => Promise<void>;
  toggleScope: (docId: number) => void;
  setScopeMode: (scopeMode: ScopeMode) => void;
  setMode: (mode: Mode) => void;
  send: (text: string, mode?: Mode, images?: ReaderImage[], web?: WebChoice) => Promise<void>;
  setWebNext: (on: boolean) => void;
  setCheckNext: (on: boolean) => void;
  /** Ask an answer's question again with the web search its model proposed. */
  searchWeb: (answerId: number) => Promise<void>;
  /** Ask an answer's question again in Deep, after Qwen stopped or failed. */
  resendDeep: (answerId: number) => Promise<void>;
  /** Rate an answer, or change its rating or note. Throws if the server refuses. */
  rate: (answerId: number, rating: Rating, note?: string) => Promise<void>;
  setView: (view: State["view"]) => void;
  stop: () => void;
  newChat: () => void;
  /** Open a source's page in the viewer, its regions highlighted. */
  openSource: (source: Pick<Source, "doc_id" | "focus_page" | "regions" | "page_start">) => void;
  openPage: (docId: number, page: number) => void;
  /** The viewer's close button: back to the narrow strip. */
  closeViewer: () => void;
  /** Text for the question box, put there from elsewhere ("Ask about this" in the viewer). */
  chatDraft: { text: string; nonce: number } | null;
  prefillChat: (text: string) => void;
  /** Every note, newest first (null until loaded). */
  notes: Note[] | null;
  notesError: string | null;
  loadNotes: () => Promise<void>;
  /** A note being written for a passage selected in the viewer. */
  noteDraft: (Passage & { docTitle: string; nonce: number }) | null;
  /** Open the notes with a new note for this passage. */
  startNote: (passage: Passage, docTitle: string) => void;
  cancelNoteDraft: () => void;
  /** The note to show, scrolled to, in the notes panel. */
  shownNote: { id: number; nonce: number } | null;
  showNote: (id: number) => void;
  /** After the server saved or changed a note. */
  putNote: (note: Note) => void;
  dropNote: (id: number) => void;
  /** A note being attached to a new passage: the viewer's toolbar offers "Attach note here". */
  reattaching: { noteId: number; docId: number } | null;
  startReattach: (note: Note) => void;
  cancelReattach: () => void;
  /** Explanations asked for in this session, newest last; the chat links to each. */
  explanations: Explanation[];
  shownExplanation: number | null;
  /** Explain a selection, a code block or an equation, with the chosen answer option. */
  explain: (body: ExplainBody, label: string, depth?: string) => Promise<void>;
  showExplanation: (id: number | null) => void;
}

export interface Explanation {
  id: number;
  /** What was explained, short: "p. 60, equation" or the code's first line. */
  label: string;
  body: ExplainBody;
  depth: string;
  result: ExplainResult | null;
  error: string | null;
}

let nextId = 1;
let opened = 0;
let controller: AbortController | null = null;

function message(role: Message["role"], text = "", streaming = false): Message {
  return {
    id: nextId++, role, text, thinking: "", sources: [], citations: [], unverified: [],
    error: null, truncated: false, streaming, finished: false,
    model: null, hosted: false, fallbackFrom: null, fallbackReason: null, searchQuery: null,
    tools: [], files: [], option: null, servedBy: null, question: "", startedAt: Date.now(),
    searched: false, cutOff: false, noAnswer: false, spend: null, images: [],
    web: false, webSources: [], webCitations: [], webProposal: null, questionId: null, feedback: null,
    check: false, checkStatus: null, checks: null, replaced: [],
  };
}

/** Deep is offered, and never taken without a click, when a Qwen answer stops or fails, or a
 * Balanced one comes back without an answer (thinking used up its limit, an empty reply, or
 * only tool lines) or cut off. A Balanced failure for any other reason, such as Ollama being
 * down, isn't one Deep fixes. */
export function offersDeepResend(
  answer: Pick<Message, "option" | "error" | "truncated" | "streaming" | "cutOff" | "noAnswer">,
): boolean {
  if (answer.streaming) return false;
  if (answer.option === "qwen") return answer.error !== null || answer.truncated;
  if (answer.option === "balanced") return answer.cutOff || answer.noAnswer || answer.truncated;
  return false;
}

/** The documents a question searches, or null for the whole library. */
export function searchedDocuments(
  state: Pick<State, "scope" | "scopeMode" | "viewer">,
): number[] | null {
  if (state.scopeMode === "document" && state.viewer) return [state.viewer.docId];
  if (state.scopeMode === "ticked" && state.scope.length > 0) return state.scope;
  return null;
}

/** What the badge on an answer says: where it ran, and which model it really was. */
export function modelBadge(answer: Pick<Message, "model" | "hosted"> & { servedBy?: string | null }): string | null {
  if (!answer.model) return null;
  const via = answer.servedBy ? ` · ${answer.servedBy}` : "";
  return `${answer.hosted ? "hosted" : "local"} · ${answer.model}${via}`;
}

export const useStore = create<State>((set, get) => {
  // An answer is changed by its id: after New chat the list no longer holds it, and a late
  // event or the stream's own finally must change nothing (review R2-C2, Nov 2026).
  const updateAnswer = (id: number, change: (answer: Message) => Partial<Message>) =>
    set((state) => {
      const index = state.messages.findIndex((item) => item.id === id);
      if (index < 0) return {};
      const messages = [...state.messages];
      messages[index] = { ...messages[index], ...change(messages[index]) };
      return { messages };
    });

  return {
    documents: [],
    libraryError: null,
    hideTest: false,
    scope: [],
    scopeMode: "library",
    messages: [],
    conversationId: null,
    options: [],
    optionsError: null,
    switchSeconds: [20, 60],
    suggestOptions: ["deep"],
    monthSpend: [],
    mode: "fast",
    localMode: "fast",
    sending: false,
    webNext: false,
    checkNext: false,
    viewer: null,
    view: "chat",

    loadDocuments: async () => {
      try {
        const library = await fetchLibrary();
        // A broken metadata.yaml: ingestion waits and hosted Explain is refused until it is fixed (R2-C7).
        const problem = library.metadataProblem
          ? `${library.metadataProblem}. New files wait until it is fixed.` : null;
        set({ documents: library.documents, hideTest: library.settings.hide_test_collection, libraryError: problem });
      } catch (error) {
        set({ libraryError: String(error) });
      }
    },

    setHideTest: async (hide) => {
      try {
        const settings = await saveLibrarySettings({ hide_test_collection: hide });
        set((state) => ({
          hideTest: settings.hide_test_collection,
          // A hidden book can't stay ticked: the reader couldn't see or untick it.
          scope: hide
            ? state.scope.filter((id) => state.documents.find((doc) => doc.id === id)?.collection !== "test")
            : state.scope,
          libraryError: null,
        }));
      } catch (error) {
        set({ libraryError: String(error) });
      }
    },

    loadOptions: async () => {
      try {
        const list = await fetchOptions();
        const known = (id: string) => list.options.some((option) => option.id === id);
        set((state) => ({
          options: list.options,
          optionsError: null,
          switchSeconds: list.switch_seconds,
          suggestOptions: list.suggest_options ?? ["deep"],
          monthSpend: list.month_spend ?? [],
          mode: known(state.mode) ? state.mode : list.default,
          localMode: known(state.localMode) ? state.localMode : list.default,
        }));
      } catch (error) {
        set({ optionsError: String(error) });
      }
    },

    toggleScope: (docId) =>
      set((state) => {
        const scope = state.scope.includes(docId)
          ? state.scope.filter((id) => id !== docId)
          : [...state.scope, docId];
        return { scope, scopeMode: scope.length > 0 ? "ticked" : "library" };
      }),

    setScopeMode: (scopeMode) => set({ scopeMode }),

    setMode: (mode) => set({ mode }),

    send: async (text, chosen, images = [], webChoice) => {
      const { conversationId, sending, options, localMode } = get();
      const mode = chosen ?? get().mode;
      const docIds = searchedDocuments(get());
      if (sending || !text.trim()) return;
      const chosenOption = options.find((option) => option.id === mode);
      const hosted = chosenOption?.hosted ?? mode === "deep";
      // An option without web search never sends one, even if the toggle was on for another.
      const web = chosenOption?.web === "off" ? { on: false } : (webChoice ?? { on: get().webNext });
      const check = get().checkNext;
      const pending = { ...message("assistant", "", true), option: mode, question: text, web: web.on, check };
      const update = (change: (answer: Message) => Partial<Message>) => updateAnswer(pending.id, change);
      const shown = () => get().messages.some((item) => item.id === pending.id);
      set((state) => ({
        sending: true,
        webNext: false,
        checkNext: false,
        mode: hosted ? localMode : mode,
        localMode: hosted ? localMode : mode,
        messages: [...state.messages, { ...message("user", text), images }, pending],
      }));

      // Tokens can arrive faster than the screen redraws, and each redraw parses
      // the whole answer, so they're applied once per frame.
      let textWaiting = "";
      let thinkingWaiting = "";
      let frame: number | null = null;
      const flush = () => {
        frame = null;
        if (!textWaiting && !thinkingWaiting) return;
        const [addText, addThinking] = [textWaiting, thinkingWaiting];
        textWaiting = thinkingWaiting = "";
        update((answer) => ({
          text: answer.text + addText,
          thinking: answer.thinking + addThinking,
        }));
      };
      const schedule = () => {
        frame ??= requestAnimationFrame(flush);
      };

      const onEvent = ({ event, data }: SseEvent) => {
        if (!shown()) return;     // New chat came first: this answer and its conversation are gone
        if (event === "meta") {
          set({ conversationId: data.conversation_id });
          update(() => ({
            model: data.model, hosted: data.hosted, option: data.option ?? mode, questionId: data.question_id ?? null,
          }));
        } else if (event === "fallback") {
          // Another model has started again, so nothing from the first one may stay.
          textWaiting = thinkingWaiting = "";
          update(() => ({
            text: "",
            thinking: "",
            tools: [],
            files: [],
            model: data.model,
            fallbackFrom: data.from,
            fallbackReason: data.reason,
          }));
        } else if (event === "sources") {
          update(() => ({ sources: data.sources, searchQuery: data.query ?? null, searched: true }));
        } else if (event === "web_source") {
          update((answer) => ({
            webSources: [...answer.webSources.filter((page) => page.number !== data.number), data],
          }));
        } else if (event === "web_proposal") {
          update(() => ({ webProposal: data }));
        } else if (event === "more_sources") {
          update((answer) => ({ sources: [...answer.sources, ...data.sources] }));
        } else if (event === "tool_start") {
          flush();
          update((answer) => ({
            tools: [...answer.tools, { id: data.id, name: data.name, running: true, ok: true, summary: "" }],
          }));
        } else if (event === "tool_end") {
          update((answer) => ({
            tools: answer.tools.map((line) =>
              line.id === data.id
                ? { ...line, running: false, ok: data.ok, summary: data.summary,
                  ...(data.detail?.suggested ? { suggested: data.detail.suggested } : {}) }
                : line,
            ),
          }));
        } else if (event === "file") {
          // A retry replaces its own file, so it mustn't leave a stale card.
          update((answer) => ({
            files: [...answer.files.filter((file) => file.name !== data.name), data],
          }));
        } else if (event === "check_status") {
          flush();
          update(() => ({ checkStatus: checkingLine(data.attempt, data.of) }));
        } else if (event === "attempt_replaced") {
          // The next attempt streams in its place; this one stays below it, collapsed.
          textWaiting = thinkingWaiting = "";
          update((answer) => ({
            replaced: [...answer.replaced, { attempt: data.attempt, answer: data.answer, checks: data.checks,
              failing: data.failing }],
            text: "",
            thinking: "",
            tools: [],
            files: [],
            sources: data.sources ?? answer.sources,
            checkStatus: retryLine(data.next, data.of, data.failing.length),
          }));
        } else if (event === "thinking") {
          thinkingWaiting += data.text;
          schedule();
        } else if (event === "token") {
          textWaiting += data.text;
          schedule();
        } else if (event === "withdraw") {
          // A failed or rebuilt file's document leaves the answer, so a retry isn't shown twice:
          // everything from `keep` on, or only up to `end`.
          flush();
          update((answer) => ({
            text: answer.text.slice(0, data.keep) + (data.end == null ? "" : answer.text.slice(data.end)),
          }));
        } else if (event === "done") {
          flush();
          update(() => ({
            citations: data.citations,
            unverified: data.unverified,
            truncated: data.done_reason === "length" || data.done_reason === "max_tokens",
            finished: true,
            model: data.model,
            hosted: data.hosted,
            servedBy: data.served_by ?? null,
            spend: data.spend ?? null,
            // Only what "done" carries replaces what came during the answer.
            ...(data.web_sources ? { webSources: data.web_sources } : {}),
            ...(data.web_citations ? { webCitations: data.web_citations } : {}),
            ...(data.web_proposal !== undefined ? { webProposal: data.web_proposal } : {}),
            // With Check on, the attempt shown may be an earlier one than the last to stream.
            ...(data.checks ? { text: data.answer, checks: data.checks, replaced: data.replaced ?? [] } : {}),
            checkStatus: null,
          }));
        } else if (event === "error") {
          flush();
          update(() => ({
            error: data.message,
            cutOff: Boolean(data.cut_off),
            noAnswer: Boolean(data.no_answer),
            spend: data.spend ?? null,
            checkStatus: null,
          }));
        }
      };

      const mine = new AbortController();
      controller = mine;
      try {
        await streamChat(
          {
            message: text, conversation_id: conversationId, doc_ids: docIds, mode,
            ...(images.length ? { images } : {}),
            ...(web.on ? { web: true, ...(web.query ? { web_query: web.query } : {}) } : {}),
            ...(check ? { check: true } : {}),
            ...(get().scopeMode === "notes" ? { notes_only: true } : {}),
          },
          onEvent,
          mine.signal,
        );
      } catch (error) {
        if ((error as Error).name !== "AbortError") {
          update(() => ({ error: `The connection failed: ${error}` }));
        }
      } finally {
        if (frame !== null) cancelAnimationFrame(frame);
        flush();
        update(() => ({ streaming: false, checkStatus: null }));
        if (controller === mine) {    // not already let go by New chat, maybe for a newer question
          controller = null;
          set({ sending: false });
        }
        // A local model may have been loaded, or the month's spend gone up: the selector shows both.
        void get().loadOptions();
      }
    },

    setWebNext: (on) => set({ webNext: on }),
    setCheckNext: (on) => set({ checkNext: on }),

    searchWeb: async (answerId) => {
      const answer = get().messages.find((candidate) => candidate.id === answerId);
      if (!answer?.question || !answer.webProposal) return;
      await get().send(answer.question, answer.option ?? undefined, [], { on: true, query: answer.webProposal.query });
    },

    resendDeep: async (answerId) => {
      const answer = get().messages.find((candidate) => candidate.id === answerId);
      if (answer?.question) await get().send(answer.question, "deep");
    },

    rate: async (answerId, rating, note = "") => {
      const answer = get().messages.find((candidate) => candidate.id === answerId);
      if (answer?.questionId == null) return;
      const saved = await rateAnswer(answer.questionId, rating, note);
      set((state) => ({
        messages: state.messages.map((item) => (item.id === answerId ? { ...item, feedback: saved } : item)),
      }));
    },

    setView: (view) => set({ view }),

    stop: () => controller?.abort(),

    newChat: () => {
      controller?.abort();
      controller = null;
      set({ messages: [], conversationId: null, view: "chat", sending: false });
    },

    openSource: (source) =>
      set({
        viewer: { docId: source.doc_id, page: pageOf(source), regions: source.regions, opened: ++opened },
      }),

    openPage: (docId, page) => set({ viewer: { docId, page, regions: [], opened: ++opened } }),
    closeViewer: () => set({ viewer: null }),
    chatDraft: null,
    prefillChat: (text) => set({ chatDraft: { text, nonce: ++opened }, view: "chat" }),
    notes: null,
    notesError: null,
    loadNotes: async () => {
      try {
        set({ notes: await fetchNotes(), notesError: null });
      } catch (error) {
        set({ notesError: `Couldn't load the notes: ${(error as Error).message}` });
      }
    },
    noteDraft: null,
    startNote: (passage, docTitle) => set({ noteDraft: { ...passage, docTitle, nonce: ++opened }, view: "notes" }),
    cancelNoteDraft: () => set({ noteDraft: null }),
    shownNote: null,
    showNote: (id) => set({ shownNote: { id, nonce: ++opened }, view: "notes" }),
    putNote: (note) => set((state) => ({
      notes: [note, ...(state.notes ?? []).filter((item) => item.id !== note.id)],
    })),
    dropNote: (id) => set((state) => ({ notes: (state.notes ?? []).filter((item) => item.id !== id) })),
    reattaching: null,
    startReattach: (note) => {
      if (note.doc_id === null) return;
      set({ reattaching: { noteId: note.id, docId: note.doc_id },
        viewer: { docId: note.doc_id, page: note.page ?? 1, regions: [], opened: ++opened } });
    },
    cancelReattach: () => set({ reattaching: null }),
    explanations: [],
    shownExplanation: null,
    explain: async (body, label, depth = "intermediate") => {
      const id = ++opened;
      const entry: Explanation = { id, label, body, depth, result: null, error: null };
      set((state) => ({ explanations: [...state.explanations, entry], shownExplanation: id }));
      const update = (change: Partial<Explanation>) => set((state) => ({
        explanations: state.explanations.map((item) => (item.id === id ? { ...item, ...change } : item)),
      }));
      try {
        update({ result: await explainText({ ...body, depth, option: get().mode }) });
      } catch (error) {
        update({ error: (error as Error).message });
      }
    },
    showExplanation: (id) => set({ shownExplanation: id }),
  };
});
