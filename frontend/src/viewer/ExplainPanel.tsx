import { useState } from "react";
import type { ExplainItem } from "../api";
import { Markdown } from "../render/Markdown";
import { useStore } from "../store/store";

const DEPTHS = [
  { value: "intro", label: "Intro" },
  { value: "intermediate", label: "Intermediate" },
  { value: "expert", label: "Expert" },
];
const ignore = () => {};

/** A row of an aligned equation as KaTeX takes it alone: no alignment marks, no row break. */
export function displayable(span: string): string {
  return span
    .replace(/\\\\\s*$/, "")
    .replace(/(^|[^\\])&/g, "$1")
    .replace(/\\(?:begin|end)\{(?:aligned|align\*?|array|split)\}(\{[^}]*\})?/g, "")
    .trim();
}

/** Explain, over the viewer: the equation or code on the left, line by line, and on the right
 * what each piece means; the pair under the pointer is marked on both sides. */
export function ExplainPanel() {
  const shown = useStore((state) => state.shownExplanation);
  const entry = useStore((state) => state.explanations.find((item) => item.id === state.shownExplanation));
  const explain = useStore((state) => state.explain);
  const close = useStore((state) => state.showExplanation);
  const options = useStore((state) => state.options);
  const [hovered, setHovered] = useState<number | null>(null);
  if (shown === null || !entry) return null;
  const result = entry.result;
  const label = (id: string) => options.find((option) => option.id === id)?.label ?? id;
  return (
    <div className="absolute inset-0 z-20 flex flex-col bg-white" data-testid="explain-panel">
      <header className="flex items-center gap-2 border-b border-stone-200 px-3 py-2 text-sm">
        <h3 className="min-w-0 flex-1 truncate font-medium">Explain: {entry.label}</h3>
        <div className="flex overflow-hidden rounded border border-stone-300 text-xs" role="group" aria-label="Depth">
          {DEPTHS.map((depth) => (
            <button
              key={depth.value}
              type="button"
              aria-pressed={entry.depth === depth.value}
              data-testid={`depth-${depth.value}`}
              disabled={!result && !entry.error}
              onClick={() => entry.depth !== depth.value && void explain(entry.body, entry.label, depth.value)}
              className={`px-2 py-0.5 ${entry.depth === depth.value ? "bg-teal-800 text-white" : "hover:bg-stone-100"}`}
            >
              {depth.label}
            </button>
          ))}
        </div>
        <button type="button" aria-label="Close" className="px-1 text-stone-500" data-testid="explain-close"
          onClick={() => close(null)}>×</button>
      </header>
      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        {!result && !entry.error && (
          <p className="pt-10 text-center text-sm text-stone-500" data-testid="explain-waiting">
            Explaining… (equations with Deep; code with Balanced or Deep)
          </p>
        )}
        {entry.error && <p className="rounded bg-red-50 p-2 text-sm text-red-800" role="alert">{entry.error}</p>}
        {result && (
          <>
            <p className="mb-2 text-xs text-stone-500">
              {result.kind === "equation" ? "Equation" : "Code"}
              {result.from_book ? ", the book's own LaTeX" : ""} · {label(result.option)}
              {result.cost_usd ? ` · $${result.cost_usd.toFixed(3)}` : ""}
            </p>
            <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)] gap-x-4" data-testid="explain-rows">
              {result.items.map((item, index) => (
                <Row key={index} item={item} code={result.kind === "code"} active={hovered === index}
                  onHover={(on) => setHovered(on ? index : null)} />
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function Row({ item, code, active, onHover }: {
  item: ExplainItem; code: boolean; active: boolean; onHover: (on: boolean) => void;
}) {
  const tone = active ? "bg-teal-50" : "";
  return (
    <>
      <div className={`border-t border-stone-100 py-2 pr-2 ${tone}`} data-testid="explain-span"
        onMouseEnter={() => onHover(true)} onMouseLeave={() => onHover(false)}>
        {code ? (
          <pre className="overflow-x-auto font-mono text-[13px] whitespace-pre-wrap">{item.span}</pre>
        ) : (
          <div className="overflow-x-auto text-sm">
            <Markdown text={`$$${displayable(item.span)}$$`} streaming={false} onCite={ignore} />
          </div>
        )}
      </div>
      <div className={`border-t border-stone-100 py-2 text-sm ${tone}`} data-testid="explain-text"
        onMouseEnter={() => onHover(true)} onMouseLeave={() => onHover(false)}>
        <Markdown text={item.explanation} streaming={false} onCite={ignore} />
      </div>
    </>
  );
}
