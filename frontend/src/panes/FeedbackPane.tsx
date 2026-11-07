import { useCallback, useEffect, useState } from "react";
import { changeFeedback, fetchFeedback } from "../api";
import type { FeedbackCounts, FeedbackEntry, FeedbackSource, Rating } from "../api";
import { Markdown } from "../render/Markdown";
import { useStore } from "../store/store";
import { pageLabel } from "../viewer/highlight";

type Filter = { rating?: Rating; status?: "open" | "done" };

/** SQLite's UTC "YYYY-MM-DD HH:MM:SS", in the reader's own time. */
export function localTime(stamp: string | null): string {
  if (!stamp) return "";
  const when = new Date(`${stamp.replace(" ", "T")}Z`);
  return Number.isNaN(when.getTime())
    ? stamp
    : when.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

/** Cited sources first, in their order; then the ones the model was given but didn't cite. */
export function sourcesInOrder(sources: FeedbackSource[]): FeedbackSource[] {
  return [...sources.filter((source) => source.cited), ...sources.filter((source) => !source.cited)];
}

function SourceLines({ entry }: { entry: FeedbackEntry }) {
  const openSource = useStore((state) => state.openSource);
  if (entry.sources.length === 0 && entry.web_sources.length === 0) {
    return <p className="mt-2 text-xs text-stone-500">No sources.</p>;
  }
  return (
    <div className="mt-3 border-t border-stone-200 pt-2" data-testid="feedback-sources">
      <h4 className="text-xs font-medium text-stone-500">Sources (grey: given to the model but not cited)</h4>
      <ul className="mt-1 space-y-0.5">
        {sourcesInOrder(entry.sources).map((source) => (
          <li key={source.id}>
            <button
              type="button"
              onClick={() => openSource(source)}
              className="flex w-full gap-2 rounded px-1 py-0.5 text-left text-sm hover:bg-stone-100"
            >
              <span className={`shrink-0 font-medium ${source.cited ? "text-teal-800" : "text-stone-400"}`}>
                S{source.id}
              </span>
              <span className={`min-w-0 truncate ${source.cited ? "text-stone-700" : "text-stone-400"}`}>
                {source.doc_title} · {pageLabel(source)}
                {source.heading_path.length > 0 && ` · ${source.heading_path[source.heading_path.length - 1]}`}
              </span>
            </button>
          </li>
        ))}
        {entry.web_sources.map((page) => (
          <li key={`web-${page.number}`} className="flex gap-2 px-1 py-0.5 text-sm">
            <span className={`shrink-0 font-medium ${page.cited ? "text-sky-800" : "text-stone-400"}`}>
              Web {page.number}
            </span>
            <a href={page.url} target="_blank" rel="noreferrer noopener" title={page.url}
              className={`min-w-0 truncate hover:underline ${page.cited ? "text-stone-700" : "text-stone-400"}`}>
              {page.title || page.url}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Entry({ entry, expanded, onToggle, onChange }: {
  entry: FeedbackEntry;
  expanded: boolean;
  onToggle: () => void;
  onChange: (change: { status?: "open" | "done"; candidate?: boolean }) => void;
}) {
  const openSource = useStore((state) => state.openSource);
  const label = useStore((state) => state.options.find((option) => option.id === entry.option)?.label);
  const onCite = useCallback(
    (id: number) => {
      const source = entry.sources.find((candidate) => candidate.id === id);
      if (source) openSource(source);
    },
    [entry.sources, openSource],
  );
  const webUrl = useCallback(
    (number: number) => entry.web_sources.find((page) => page.number === number)?.url,
    [entry.web_sources],
  );
  return (
    <li className={`rounded-lg bg-white p-3 shadow-sm ring-1 ${entry.rating === "down" ? "ring-red-200" : "ring-stone-200"}`}
      data-testid="feedback-entry">
      <button type="button" onClick={onToggle} aria-expanded={expanded} className="block w-full text-left">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs text-stone-500">
          <span className="text-base leading-none" aria-label={entry.rating === "up" ? "Good answer" : "Bad answer"}>
            {entry.rating === "up" ? "👍" : "👎"}
          </span>
          <span>{label ?? entry.option ?? "unknown option"}</span>
          {entry.model && <span>· {entry.hosted ? "hosted" : "local"} · {entry.model}{entry.served_by ? ` · ${entry.served_by}` : ""}</span>}
          <span>· {localTime(entry.asked_at)}</span>
          {entry.status === "done" && <span className="rounded bg-stone-100 px-1.5 text-stone-600">done</span>}
          {entry.candidate && (
            <span className="rounded bg-amber-100 px-1.5 text-amber-900" data-testid="candidate-badge">gold candidate</span>
          )}
          {entry.error && <span className="rounded bg-red-50 px-1.5 text-red-800">failed</span>}
        </div>
        <p className={`mt-1 text-sm font-medium text-stone-800 ${expanded ? "whitespace-pre-wrap" : "line-clamp-2"}`}>
          {entry.question}
        </p>
        {entry.note && <p className="mt-1 text-sm italic text-stone-600">“{entry.note}”</p>}
      </button>
      {expanded && (
        <div className="mt-2" data-testid="feedback-detail">
          {entry.error && <p className="rounded bg-red-50 p-2 text-sm text-red-800">{entry.error}</p>}
          {entry.answer && (
            <div className="max-h-[60vh] overflow-y-auto border-l-2 border-stone-200 pl-3">
              <Markdown text={entry.answer} streaming={false} onCite={onCite} webUrl={webUrl} />
            </div>
          )}
          <SourceLines entry={entry} />
          <p className="mt-2 text-xs text-stone-500">
            Asked {localTime(entry.asked_at)}
            {entry.answered_at && `, answered ${localTime(entry.answered_at)}`}; rated {localTime(entry.updated_at)}.
          </p>
          <div className="mt-2 flex flex-wrap gap-2">
            <button type="button" data-testid="toggle-done"
              onClick={() => onChange({ status: entry.status === "open" ? "done" : "open" })}
              className="rounded border border-stone-300 px-2 py-1 text-xs text-stone-700 hover:bg-stone-100">
              {entry.status === "open" ? "Mark done" : "Open again"}
            </button>
            {entry.rating === "down" && (
              <button type="button" data-testid="toggle-candidate"
                onClick={() => onChange({ candidate: !entry.candidate })}
                className="rounded border border-amber-300 px-2 py-1 text-xs text-amber-900 hover:bg-amber-50">
                {entry.candidate ? "Not a gold candidate" : "Make a gold candidate"}
              </button>
            )}
          </div>
          {entry.candidate && (
            <p className="mt-1 text-xs text-stone-500">
              Written to data/gold_candidates.yaml. Copy it into eval/questions.yaml by hand to use it.
            </p>
          )}
        </div>
      )}
    </li>
  );
}

const FILTERS: { label: string; filter: Filter; count: (counts: FeedbackCounts) => number }[] = [
  { label: "Open", filter: { status: "open" }, count: (counts) => counts.open },
  { label: "👎", filter: { rating: "down" }, count: (counts) => counts.down },
  { label: "👍", filter: { rating: "up" }, count: (counts) => counts.up },
  { label: "All", filter: {}, count: (counts) => counts.total },
];

/** The review list: every rated answer, newest first. */
export function FeedbackPane() {
  const setView = useStore((state) => state.setView);
  const [chosen, setChosen] = useState(0);
  const [entries, setEntries] = useState<FeedbackEntry[] | null>(null);
  const [counts, setCounts] = useState<FeedbackCounts | null>(null);
  const [expanded, setExpanded] = useState<number | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const list = await fetchFeedback(FILTERS[chosen].filter);
      setEntries(list.entries);
      setCounts(list.counts);
      setProblem(null);
    } catch (error) {
      setProblem(`Couldn't load the feedback list: ${(error as Error).message}`);
    }
  }, [chosen]);
  useEffect(() => {
    void load();
  }, [load]);

  const change = async (id: number, what: { status?: "open" | "done"; candidate?: boolean }) => {
    try {
      await changeFeedback(id, what);
      await load();
    } catch (error) {
      setProblem(`Not changed: ${(error as Error).message}`);
    }
  };

  return (
    <main className="flex min-h-0 min-w-0 flex-col" data-testid="feedback-pane">
      <header className="flex items-center justify-between border-b border-stone-200 px-4 py-3">
        <h2 className="text-sm font-semibold">Feedback</h2>
        <button type="button" onClick={() => setView("chat")} data-testid="back-to-chat"
          className="text-sm text-stone-600 hover:text-stone-900">
          Back to chat
        </button>
      </header>
      <div className="flex flex-wrap items-center gap-1.5 border-b border-stone-200 px-4 py-2">
        {FILTERS.map((item, index) => (
          <button key={item.label} type="button" aria-pressed={chosen === index} data-testid={`filter-${index}`}
            onClick={() => setChosen(index)}
            className={`rounded px-2 py-1 text-xs ${chosen === index ? "bg-teal-800 text-white" : "text-stone-600 hover:bg-stone-100"}`}>
            {item.label}{counts ? ` ${item.count(counts)}` : ""}
          </button>
        ))}
        {counts && counts.candidates > 0 && (
          <span className="ml-auto text-xs text-stone-500">{counts.candidates} gold candidate{counts.candidates === 1 ? "" : "s"}</span>
        )}
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        {problem && <p className="mb-3 rounded bg-red-50 p-2 text-sm text-red-800" role="alert">{problem}</p>}
        {entries?.length === 0 && (
          <p className="pt-16 text-center text-sm text-stone-500">
            {counts?.total ? "Nothing here with this filter." : "No feedback yet. Rate an answer with 👍 or 👎 under it."}
          </p>
        )}
        <ul className="space-y-3">
          {entries?.map((entry) => (
            <Entry key={entry.id} entry={entry} expanded={expanded === entry.id}
              onToggle={() => setExpanded(expanded === entry.id ? null : entry.id)}
              onChange={(what) => void change(entry.id, what)} />
          ))}
        </ul>
      </div>
    </main>
  );
}
