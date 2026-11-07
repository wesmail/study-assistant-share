import { useEffect, useRef, useState } from "react";
import { batchAction, changeAnnotation, deleteAnnotation, suggestHighlights } from "../api";
import type { Annotation } from "../api";
import { useStore } from "../store/store";

const EXAMPLES = [
  "Highlight the key definitions and results to remember",
  "Mark grammar mistakes and suggest corrections",
  "Highlight claims without a citation",
];

/** Suggested highlights from any instruction, on a few pages, for the reader to accept or reject. */
export function SuggestPanel({ docId, page, pages, annotations, onChange, onGo, onClose }: {
  docId: number; page: number; pages: number; annotations: Annotation[];
  onChange: () => Promise<void>; onGo: (page: number) => void; onClose: () => void;
}) {
  const options = useStore((state) => state.options);
  const suggestOptions = useStore((state) => state.suggestOptions);
  const [instruction, setInstruction] = useState("");
  const [first, setFirst] = useState(page);
  const [last, setLast] = useState(page);
  const [option, setOption] = useState(suggestOptions[0] ?? "deep");
  const [running, setRunning] = useState(false);
  const [report, setReport] = useState<string | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const waiting = annotations.filter((annotation) => annotation.status === "suggested");
  const batches = [...new Set(waiting.map((annotation) => annotation.batch).filter(Boolean))] as string[];
  const label = (id: string) => options.find((item) => item.id === id)?.label ?? id;
  const hosted = options.find((item) => item.id === option)?.hosted ?? true;
  // The panel is remade for each document; a reply for the one before must change nothing (R2-C5).
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const run = async () => {
    if (!instruction.trim() || running) return;
    setRunning(true);
    setProblem(null);
    setReport(null);
    try {
      const result = await suggestHighlights({ doc_id: docId, first_page: Math.min(first, last), last_page: Math.max(first, last),
        instruction, option });
      if (!mounted.current) return;
      await onChange();
      const missing = result.not_found.length;
      setReport(`${result.suggestions.length} suggestion${result.suggestions.length === 1 ? "" : "s"}`
        + (missing ? `; ${missing} quote${missing === 1 ? "" : "s"} the model gave weren't in the text and were dropped` : "")
        + (result.cost_usd ? ` · $${result.cost_usd.toFixed(3)}` : "") + ".");
      const firstPage = result.suggestions[0]?.page;
      if (firstPage) onGo(firstPage);
    } catch (error) {
      if (mounted.current) setProblem((error as Error).message);
    } finally {
      if (mounted.current) setRunning(false);
    }
  };

  const decide = async (work: Promise<unknown>) => {
    try {
      await work;
      await onChange();
    } catch (error) {
      setProblem((error as Error).message);
    }
  };

  return (
    <section className="border-b border-stone-200 bg-white px-3 py-2 text-sm" data-testid="suggest-panel">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-stone-500">Suggest highlights</h3>
        <button type="button" aria-label="Close" className="px-1 text-stone-500" onClick={onClose}>×</button>
      </div>
      <textarea value={instruction} onChange={(event) => setInstruction(event.target.value)} rows={2}
        aria-label="Instruction" data-testid="suggest-instruction" maxLength={500}
        placeholder={`What to look for, e.g. "${EXAMPLES[0]}"`}
        className="mt-1 w-full resize-y rounded border border-stone-300 px-2 py-1 text-sm outline-none focus:border-teal-700" />
      <div className="mt-1 flex flex-wrap gap-1">
        {EXAMPLES.map((example) => (
          <button key={example} type="button" onClick={() => setInstruction(example)}
            className="rounded bg-stone-100 px-1.5 py-0.5 text-xs text-stone-600 hover:bg-stone-200">{example}</button>
        ))}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-stone-600">
        <label>Pages
          <input type="number" min={1} max={pages} value={first} data-testid="suggest-first"
            onChange={(event) => setFirst(Number(event.target.value))} className="ml-1 w-14 rounded border border-stone-300 px-1" />
        </label>
        <label>to
          <input type="number" min={1} max={pages} value={last} data-testid="suggest-last"
            onChange={(event) => setLast(Number(event.target.value))} className="ml-1 w-14 rounded border border-stone-300 px-1" />
        </label>
        <span>(at most 10)</span>
        {suggestOptions.length > 1 ? (
          <select value={option} onChange={(event) => setOption(event.target.value)} aria-label="Model"
            className="rounded border border-stone-300 px-1">
            {suggestOptions.map((id) => <option key={id} value={id}>{label(id)}</option>)}
          </select>
        ) : (
          <span>{label(option)}{hosted ? ", hosted: costs a few cents" : ""}</span>
        )}
        <button type="button" onClick={() => void run()} disabled={!instruction.trim() || running} data-testid="suggest-run"
          className="ml-auto rounded bg-teal-800 px-3 py-1 font-medium text-white disabled:opacity-50">
          {running ? "Reading the pages…" : "Suggest"}
        </button>
      </div>
      {problem && <p className="mt-2 text-xs text-red-800" role="alert">{problem}</p>}
      {report && <p className="mt-2 text-xs text-stone-600" data-testid="suggest-report">{report}</p>}
      {waiting.length > 0 && (
        <div className="mt-2 border-t border-stone-100 pt-2" data-testid="suggestions">
          <div className="flex items-center gap-2 text-xs">
            <span className="font-medium text-stone-700">{waiting.length} waiting for you</span>
            <button type="button" className="ml-auto text-teal-800 underline" data-testid="accept-all"
              onClick={() => void decide(Promise.all(batches.map((batch) => batchAction(batch, "accept"))))}>Accept all</button>
            <button type="button" className="text-stone-600 underline" data-testid="reject-all"
              onClick={() => void decide(Promise.all(batches.map((batch) => batchAction(batch, "reject"))))}>Reject all</button>
          </div>
          <ul className="mt-1 max-h-48 space-y-1.5 overflow-y-auto">
            {waiting.map((annotation) => (
              <li key={annotation.id} className="rounded bg-sky-50 p-1.5 text-xs" data-testid="suggestion">
                <button type="button" className="block w-full text-left" onClick={() => onGo(annotation.page)}>
                  <span className="text-stone-500">p. {annotation.page} · </span>
                  <span className="italic text-stone-800">{annotation.quote}</span>
                  {annotation.body && <span className="block text-stone-600">{annotation.body}</span>}
                </button>
                <span className="mt-1 flex gap-3">
                  <button type="button" className="text-teal-800 underline" data-testid="suggestion-accept"
                    onClick={() => void decide(changeAnnotation(annotation.id, { status: "accepted" }))}>Accept</button>
                  <button type="button" className="text-stone-600 underline" data-testid="suggestion-reject"
                    onClick={() => void decide(deleteAnnotation(annotation.id))}>Reject</button>
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
