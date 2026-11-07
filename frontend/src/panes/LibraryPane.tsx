import { useState } from "react";
import {
  approveFigures,
  fetchDetails,
  fetchRemoval,
  removeDocument,
  saveDetails,
  skipFigures,
  type DocumentDetails,
  type DocumentRow,
  type RemovalPreview,
} from "../api";
import { searchedDocuments, useStore } from "../store/store";
import { THEMES, useTheme } from "../theme";
import type { Theme } from "../theme";

const STATE_STYLE: Record<string, string> = {
  done: "text-teal-700",
  failed: "text-red-700",
  removed: "text-amber-700",
};

/** "Deisenroth et al.": the first author's surname, for a line under the title. */
export function shortAuthors(author: string | null): string | null {
  if (!author) return null;
  const names = author.split(/,|;|\band\b/).map((name) => name.trim()).filter(Boolean);
  const surname = (name: string) => name.split(/\s+/).at(-1) ?? name;
  if (names.length === 1) return surname(names[0]);
  if (names.length === 2) return `${surname(names[0])} and ${surname(names[1])}`;
  return `${surname(names[0])} et al.`;
}

/** "parsing 3/10", or the state alone. Older servers send no state: their status stands in. */
export function stateLabel(document: DocumentRow): string {
  const state = document.state ?? document.status;
  const progress = document.progress;
  return progress && progress.total ? `${state} ${progress.done}/${progress.total}` : state;
}

export function dollars(usd: number | null | undefined): string {
  if (usd === null || usd === undefined) return "an unknown amount";
  return usd < 0.01 ? "under $0.01" : `about $${usd.toFixed(2)}`;
}

const PROVIDER_NAME: Record<string, string> = { anthropic: "Anthropic", openrouter: "OpenRouter" };

function FiguresOk({ document, onDone }: { document: DocumentRow; onDone: () => void }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const waiting = document.waiting!;
  const act = async (action: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      await action();
      onDone();
    } catch (failure) {
      setError(String(failure instanceof Error ? failure.message : failure));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="mt-2 rounded border border-amber-300 bg-amber-50 p-2 text-xs" data-testid="figures-ok">
      <p>
        Describe {waiting.figures} figure{waiting.figures === 1 ? "" : "s"} with {waiting.model} for{" "}
        <strong>{dollars(waiting.usd)}</strong>? Each image, its caption and its page's text go to{" "}
        {PROVIDER_NAME[waiting.provider] ?? waiting.provider}.
      </p>
      <div className="mt-1.5 flex gap-2">
        <button
          type="button"
          disabled={busy}
          className="rounded bg-teal-700 px-2 py-0.5 text-white disabled:opacity-50"
          onClick={() => act(() => approveFigures(document.id, waiting.usd ?? 0))}
        >
          Describe
        </button>
        <button
          type="button"
          disabled={busy}
          className="rounded border border-stone-300 px-2 py-0.5 disabled:opacity-50"
          onClick={() => act(() => skipFigures(document.id))}
        >
          Skip
        </button>
      </div>
      {error && <p className="mt-1 text-red-700">{error}</p>}
    </div>
  );
}

const FIELDS: { key: keyof DocumentDetails; label: string }[] = [
  { key: "title", label: "Title" },
  { key: "author", label: "Authors" },
  { key: "year", label: "Year" },
  { key: "publisher", label: "Publisher" },
];

function DetailsForm({ document, onDone }: { document: DocumentRow; onDone: () => void }) {
  const [values, setValues] = useState<DocumentDetails | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const open = async () => {
    setError(null);
    try {
      setValues(await fetchDetails(document.id));
    } catch (failure) {
      setError(String(failure instanceof Error ? failure.message : failure));
    }
  };
  const save = async () => {
    if (!values) return;
    setBusy(true);
    setError(null);
    try {
      await saveDetails(document.id, values);
      setValues(null);
      onDone();
    } catch (failure) {
      setError(String(failure instanceof Error ? failure.message : failure));
    } finally {
      setBusy(false);
    }
  };

  if (!values) {
    return (
      <div className="mt-2 text-xs">
        <button type="button" className="text-teal-700 underline" onClick={open}>
          Add title, authors and year
        </button>
        {error && <p className="mt-1 text-red-700">{error}</p>}
      </div>
    );
  }
  return (
    <form
      className="mt-2 space-y-1 rounded border border-stone-300 bg-white p-2 text-xs"
      data-testid="details-form"
      onSubmit={(event) => {
        event.preventDefault();
        void save();
      }}
    >
      {FIELDS.map(({ key, label }) => (
        <label key={key} className="block">
          <span className="text-stone-500">{label}</span>
          <input
            className="mt-0.5 block w-full rounded border border-stone-300 px-1 py-0.5"
            inputMode={key === "year" ? "numeric" : undefined}
            value={values[key] ?? ""}
            onChange={(event) => setValues({ ...values, [key]: event.target.value })}
          />
        </label>
      ))}
      <p className="text-stone-500">Saved in library/metadata.yaml.</p>
      <div className="flex gap-2">
        <button type="submit" disabled={busy} className="rounded bg-teal-700 px-2 py-0.5 text-white disabled:opacity-50">
          Save
        </button>
        <button type="button" className="rounded border border-stone-300 px-2 py-0.5" onClick={() => setValues(null)}>
          Cancel
        </button>
      </div>
      {error && <p className="text-red-700">{error}</p>}
    </form>
  );
}

function megabytes(bytes: number): string {
  return bytes < 1e6 ? `${Math.max(1, Math.round(bytes / 1e3))} KB` : `${(bytes / 1e6).toFixed(1)} MB`;
}

/** The lines of the confirmation: only what there is to delete. */
export function removalLines(preview: RemovalPreview): string[] {
  const { text, figure, summary } = preview.chunks;
  const lines = [
    `${text} text passages, ${figure} figure passages and ${summary} summaries, from search`,
    `${preview.blocks} blocks, ${preview.figures} figures and ${preview.jobs} jobs`,
  ];
  if (preview.folders.length) lines.push(`${megabytes(preview.folder_bytes)} of parsed files (${preview.folders.join(", ")})`);
  if (preview.figure_descriptions)
    lines.push(
      `${preview.figure_descriptions} cached figure descriptions: describing them again would cost about $${preview.figure_descriptions_usd.toFixed(2)}`,
    );
  if (preview.summary_cache)
    lines.push(
      `${preview.summary_cache} cached summary replies: writing them again would take about ${Math.ceil(preview.summary_cache_seconds / 60)} min`,
    );
  if (preview.annotations) lines.push(`${preview.annotations} annotations`);
  if (preview.notes_kept) lines.push(`Kept: ${preview.notes_kept} notes, no longer tied to the document`);
  return lines;
}

function RemoveDocument({ document, onDone }: { document: DocumentRow; onDone: () => void }) {
  const [preview, setPreview] = useState<RemovalPreview | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const run = async (action: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (failure) {
      setError(String(failure instanceof Error ? failure.message : failure));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="mt-2 rounded border border-amber-300 bg-amber-50 p-2 text-xs" data-testid="removal">
      <p>The file has left library/, so this book is left out of search. Its index is still here.</p>
      {!preview ? (
        <button
          type="button"
          disabled={busy}
          className="mt-1.5 rounded border border-stone-300 bg-white px-2 py-0.5 disabled:opacity-50"
          onClick={() => run(async () => setPreview(await fetchRemoval(document.id)))}
        >
          Remove from the index…
        </button>
      ) : (
        <>
          <p className="mt-1.5 font-semibold">This deletes:</p>
          <ul className="ml-4 list-disc">
            {removalLines(preview).map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
          <p className="mt-1">If you add the file again, it's indexed from the start.</p>
          <div className="mt-1.5 flex gap-2">
            <button
              type="button"
              disabled={busy}
              className="rounded bg-red-700 px-2 py-0.5 text-white disabled:opacity-50"
              onClick={() =>
                run(async () => {
                  await removeDocument(document.id);
                  onDone();
                })
              }
            >
              Remove
            </button>
            <button type="button" className="rounded border border-stone-300 bg-white px-2 py-0.5" onClick={() => setPreview(null)}>
              Keep
            </button>
          </div>
        </>
      )}
      {error && <p className="mt-1 text-red-700">{error}</p>}
    </div>
  );
}

function DocumentItem({ document }: { document: DocumentRow }) {
  const scope = useStore((state) => state.scope);
  const toggleScope = useStore((state) => state.toggleScope);
  const openPage = useStore((state) => state.openPage);
  const loadDocuments = useStore((state) => state.loadDocuments);
  const refresh = () => void loadDocuments();
  const state = document.state ?? document.status;

  return (
    <li className="flex gap-2 rounded px-2 py-2 hover:bg-stone-200/60" data-testid="document">
      <input
        type="checkbox"
        className="mt-1 accent-teal-700"
        aria-label={`Search only in ${document.title}`}
        checked={scope.includes(document.id)}
        onChange={() => toggleScope(document.id)}
      />
      <div className="min-w-0 flex-1">
        <button
          type="button"
          className="block w-full text-left text-sm leading-snug hover:underline"
          title={document.title}
          onClick={() => openPage(document.id, 1)}
        >
          <span className="line-clamp-3">{document.title}</span>
        </button>
        {(document.author || document.year) && (
          <p className="mt-0.5 truncate text-xs text-stone-600" data-testid="byline">
            {[shortAuthors(document.author), document.year].filter(Boolean).join(" · ")}
          </p>
        )}
        <p className="mt-1 text-xs text-stone-500">
          {document.pages ?? "?"} pages · {document.chunks} chunks ·{" "}
          <span className={STATE_STYLE[state] ?? ""} data-testid="state">
            {stateLabel(document)}
          </span>
        </p>
        {document.error && (
          <p className="mt-1 text-xs text-red-700" title={document.error}>
            {document.failed_stage ? `${document.failed_stage} failed: ` : ""}
            <span className="line-clamp-2">{document.error}</span>
            {document.searchable && state === "failed" && <span className="text-stone-500"> (still searchable)</span>}
          </p>
        )}
        {document.waiting?.stage === "describe" && <FiguresOk document={document} onDone={refresh} />}
        {document.needs_details && <DetailsForm document={document} onDone={refresh} />}
        {state === "removed" && <RemoveDocument document={document} onDone={refresh} />}
      </div>
    </li>
  );
}

export function LibraryPane() {
  const documents = useStore((state) => state.documents);
  const error = useStore((state) => state.libraryError);
  const hideTest = useStore((state) => state.hideTest);
  const setHideTest = useStore((state) => state.setHideTest);
  // A count, because searchedDocuments can return a new list each time, and a selector
  // that always returns something new makes the page render forever.
  const searched = useStore((state) => searchedDocuments(state)?.length ?? 0);
  const testBooks = documents.filter((document) => document.collection === "test").length;
  const shown = hideTest ? documents.filter((document) => document.collection !== "test") : documents;

  return (
    <aside className="flex min-h-0 flex-col border-r border-stone-200 bg-stone-100" data-testid="library">
      <header className="border-b border-stone-200 px-4 py-3">
        <div className="flex items-center justify-between gap-2">
          <h1 className="text-sm font-semibold">Library</h1>
          <ThemeSwitch />
        </div>
        <p className="mt-0.5 text-xs text-stone-500" data-testid="searching">
          {searched > 0
            ? `Searching ${searched} of ${shown.length}`
            : hideTest && testBooks > 0
              ? "Searching everything but the test collection"
              : "Searching everything"}
        </p>
        {testBooks > 0 && (
          <label className="mt-1.5 flex items-center gap-1.5 text-xs text-stone-600">
            <input
              type="checkbox"
              className="accent-teal-700"
              checked={hideTest}
              onChange={(event) => void setHideTest(event.target.checked)}
            />
            Hide the test collection ({testBooks} {testBooks === 1 ? "book" : "books"})
          </label>
        )}
      </header>
      {error && <p className="p-4 text-sm text-red-700">{error}</p>}
      <ul className="min-h-0 flex-1 overflow-y-auto p-2">
        {shown.map((document) => (
          <DocumentItem key={document.id} document={document} />
        ))}
      </ul>
    </aside>
  );
}


const THEME_LABELS: Record<Theme, string> = { light: "Light", dark: "Dark", system: "System" };

/** Light, dark, or the system's mode (theme.ts); kept in this browser. */
function ThemeSwitch() {
  const [theme, setTheme] = useTheme();
  return (
    <div className="flex rounded border border-stone-300 text-[11px]" role="group" aria-label="Colour theme"
      data-testid="theme-switch">
      {THEMES.map((option) => (
        <button key={option} type="button" aria-pressed={theme === option} data-testid={`theme-${option}`}
          onClick={() => setTheme(option)}
          className={`px-1.5 py-0.5 first:rounded-l last:rounded-r ${
            theme === option ? "bg-teal-800 text-white" : "text-stone-600 hover:bg-stone-200"}`}>
          {THEME_LABELS[option]}
        </button>
      ))}
    </div>
  );
}
