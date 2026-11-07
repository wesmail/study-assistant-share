import * as pdfjs from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent } from "react";
import type { PDFDocumentProxy, RenderTask } from "pdfjs-dist";
import {
  changeAnnotation, changeNote, createAnnotation, deleteAnnotation, exportAnnotated, fetchAnnotations, HIGHLIGHT_COLORS,
} from "../api";
import type { Annotation, HighlightColor } from "../api";
import { useStore } from "../store/store";
import { highlightBoxes } from "../viewer/highlight";
import { annotationAt, annotationBoxes, askAboutText, selectedText, selectionRects } from "../viewer/selection";
import type { Rect } from "../viewer/selection";
import { SuggestPanel } from "../viewer/SuggestPanel";
import { ExplainPanel } from "../viewer/ExplainPanel";

pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;

const ZOOM_STEP = 1.25;

const HIGHLIGHT_STYLE: Record<HighlightColor, string> = {
  yellow: "bg-yellow-300/45",
  green: "bg-lime-300/45",
  blue: "bg-sky-300/45",
  pink: "bg-pink-300/45",
};

const SWATCH: Record<HighlightColor, string> = {
  yellow: "bg-yellow-300",
  green: "bg-lime-300",
  blue: "bg-sky-300",
  pink: "bg-pink-300",
};

/** What the reader has selected on the page: boxes in PDF points and the text. */
interface Selected {
  rects: Rect[];
  quote: string;
}

export function ViewerPane() {
  const target = useStore((state) => state.viewer);
  const documents = useStore((state) => state.documents);
  const prefillChat = useStore((state) => state.prefillChat);
  const startNote = useStore((state) => state.startNote);
  const showNote = useStore((state) => state.showNote);
  const allNotes = useStore((state) => state.notes);
  const loadNotes = useStore((state) => state.loadNotes);
  const reattaching = useStore((state) => state.reattaching);
  const cancelReattach = useStore((state) => state.cancelReattach);
  const closeViewer = useStore((state) => state.closeViewer);
  const putNote = useStore((state) => state.putNote);
  const explain = useStore((state) => state.explain);
  const [pdf, setPdf] = useState<PDFDocumentProxy | null>(null);
  const [page, setPage] = useState(1);
  const [zoom, setZoom] = useState(1);
  const [width, setWidth] = useState(0);
  const [scale, setScale] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [selected, setSelected] = useState<Selected | null>(null);
  const [active, setActive] = useState<number | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [color, setColor] = useState<HighlightColor>("yellow");
  const [suggesting, setSuggesting] = useState(false);
  const [copy, setCopy] = useState<{ url: string; text: string } | null>(null);
  const area = useRef<HTMLDivElement>(null);
  const sheet = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const textLayer = useRef<HTMLDivElement>(null);
  const firstHighlight = useRef<HTMLDivElement>(null);
  const docId = target?.docId ?? null;
  // The document open now, for replies that come back after another one was opened (review R2-C5).
  const openDoc = useRef(docId);
  openDoc.current = docId;

  useEffect(() => {
    if (docId === null) return;
    const task = pdfjs.getDocument({ url: `/api/documents/${docId}/file` });
    setPdf(null);
    setError(null);
    task.promise.then(setPdf, (reason) => {
      if (reason?.name !== "AbortException") setError(`Couldn't open the PDF: ${reason}`);
    });
    return () => {
      void task.destroy();
    };
  }, [docId]);

  useEffect(() => {
    setAnnotations([]);
    setActive(null);
    setCopy(null);
    if (docId === null) return;
    let current = true;
    fetchAnnotations(docId).then(
      (found) => current && setAnnotations(found),
      (reason) => current && setNotice(`Couldn't load the highlights: ${(reason as Error).message}`),
    );
    return () => {
      current = false;
    };
  }, [docId]);

  const reload = useCallback(async () => {
    if (docId === null) return;
    const found = await fetchAnnotations(docId);
    if (openDoc.current === docId) setAnnotations(found);
  }, [docId]);

  const makeCopy = async () => {
    if (docId === null) return;
    try {
      const made = await exportAnnotated(docId);
      const left = made.suggested_left_out ? ` ${made.suggested_left_out} suggestion${made.suggested_left_out === 1 ? "" : "s"} not yet accepted left out.` : "";
      const lost = made.lost_left_out ? ` ${made.lost_left_out} that lost their passage left out.` : "";
      setCopy({ url: `${made.url}?download=true`, text: `Annotated copy: ${made.highlights} highlight${made.highlights === 1 ? "" : "s"}, ${made.notes} note${made.notes === 1 ? "" : "s"}.${left}${lost} The original is unchanged.` });
    } catch (reason) {
      setNotice(`Couldn't make the annotated copy: ${(reason as Error).message}`);
    }
  };

  useEffect(() => {
    if (target) setPage(target.page);
  }, [target]);

  useEffect(() => {
    if (allNotes === null) void loadNotes();
  }, [allNotes, loadNotes]);

  useEffect(() => {
    const element = area.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  // A new page or zoom: what was selected or open belongs to the old one.
  useEffect(() => {
    setSelected(null);
    setActive(null);
  }, [page, zoom, docId]);

  useEffect(() => {
    if (!pdf || !canvas.current || width === 0) return;
    let task: RenderTask | null = null;
    let text: pdfjs.TextLayer | null = null;
    let current = true;
    pdf.getPage(page).then((pdfPage) => {
      if (!current || !canvas.current) return;
      const fitted = ((width - 24) / pdfPage.getViewport({ scale: 1 }).width) * zoom;
      const ratio = window.devicePixelRatio || 1;
      const viewport = pdfPage.getViewport({ scale: fitted * ratio });
      canvas.current.width = viewport.width;
      canvas.current.height = viewport.height;
      canvas.current.style.width = `${viewport.width / ratio}px`;
      canvas.current.style.height = `${viewport.height / ratio}px`;
      setScale(fitted);
      task = pdfPage.render({ canvas: canvas.current, viewport });
      task.promise.catch((reason) => {
        if (reason?.name !== "RenderingCancelledException") setError(`Couldn't draw the page: ${reason}`);
      });
      // The page's text, transparent over the canvas, so it can be selected.
      const layer = textLayer.current;
      if (!layer) return;
      layer.replaceChildren();
      layer.style.setProperty("--total-scale-factor", String(fitted));
      layer.style.setProperty("--scale-round-x", "1px");
      layer.style.setProperty("--scale-round-y", "1px");
      text = new pdfjs.TextLayer({
        textContentSource: pdfPage.streamTextContent(),
        container: layer,
        viewport: pdfPage.getViewport({ scale: fitted }),
      });
      text.render().then(() => {
        const end = document.createElement("div");
        end.className = "endOfContent";
        layer.append(end);
      }, () => {});
    });
    return () => {
      current = false;
      task?.cancel();
      text?.cancel();
    };
  }, [pdf, page, width, zoom]);

  const boxes = useMemo(
    () => (target && scale ? highlightBoxes(target.regions, page, scale) : []),
    [target, page, scale],
  );

  useEffect(() => {
    firstHighlight.current?.scrollIntoView({ block: "center" });
  }, [boxes]);

  // A highlight or note whose passage wasn't found after the book changed isn't drawn where it was.
  const onPage = useMemo(
    () => annotations.filter((annotation) => annotation.page === page && annotation.anchor_status !== "unanchored"),
    [annotations, page],
  );
  const notesOnPage = useMemo(
    () => (allNotes ?? []).filter((note) => note.doc_id === docId && note.page === page && note.rects.length > 0
      && note.anchor_status !== "unanchored"),
    [allNotes, docId, page],
  );
  const title = documents.find((document) => document.id === docId)?.title;
  const pages = pdf?.numPages ?? 0;
  const go = (next: number) => setPage(Math.min(Math.max(next, 1), pages));

  const clearSelection = () => {
    window.getSelection()?.removeAllRanges();
    setSelected(null);
  };

  /** After a drag or a click on the page: a selection opens the toolbar, a click on a highlight its menu. */
  const readSelection = useCallback((event?: ReactMouseEvent) => {
    const layer = textLayer.current;
    const page_ = sheet.current;
    layer?.classList.remove("selecting");
    if (!layer || !page_ || !scale) return;
    const selection = window.getSelection();
    const origin = page_.getBoundingClientRect();
    if (!selection || selection.isCollapsed || selection.rangeCount === 0) {
      setSelected(null);
      if (event) {
        const hit = annotationAt(onPage, (event.clientX - origin.left) / scale, (event.clientY - origin.top) / scale);
        setActive(hit?.id ?? null);
      }
      return;
    }
    const range = selection.getRangeAt(0);
    if (!layer.contains(range.commonAncestorContainer)) return;
    const quote = selectedText(selection.toString());
    const rects = selectionRects([...range.getClientRects()], origin, scale);
    setActive(null);
    setSelected(quote && rects.length ? { rects, quote } : null);
  }, [scale, onPage]);

  const highlight = async () => {
    if (!selected || docId === null) return;
    const chosen = selected;
    clearSelection();
    try {
      const saved = await createAnnotation({ doc_id: docId, page, rects: chosen.rects, quote: chosen.quote, color });
      if (saved.doc_id !== openDoc.current) return;
      setAnnotations((current) => [...current, saved]);
      setNotice(saved.snapped ? null : "Highlighted as selected: it didn't match the page's words exactly.");
    } catch (reason) {
      setNotice(`Couldn't save the highlight: ${(reason as Error).message}`);
    }
  };

  const addNote = (rects: number[][], quote: string) => {
    if (docId === null) return;
    startNote({ doc_id: docId, page, rects, quote }, title ?? "");
    clearSelection();
    setActive(null);
  };

  const attachHere = async (rects: number[][], quote: string) => {
    if (!reattaching || docId === null) return;
    const { noteId } = reattaching;
    clearSelection();
    try {
      putNote(await changeNote(noteId, { passage: { doc_id: docId, page, rects, quote } }));
      cancelReattach();
      showNote(noteId);
    } catch (reason) {
      setNotice(`Couldn't attach the note: ${(reason as Error).message}`);
    }
  };

  const askAbout = (quote: string) => {
    prefillChat(askAboutText(quote, title, page));
    clearSelection();
    setActive(null);
  };

  const recolor = async (annotation: Annotation, next: HighlightColor) => {
    setColor(next);
    try {
      const saved = await changeAnnotation(annotation.id, { color: next });
      setAnnotations((current) => current.map((item) => (item.id === saved.id ? saved : item)));
    } catch (reason) {
      setNotice(`Couldn't change the colour: ${(reason as Error).message}`);
    }
  };

  const accept = async (annotation: Annotation) => {
    try {
      const saved = await changeAnnotation(annotation.id, { status: "accepted" });
      setAnnotations((current) => current.map((item) => (item.id === saved.id ? saved : item)));
    } catch (reason) {
      setNotice(`Couldn't accept it: ${(reason as Error).message}`);
    }
  };

  const remove = async (annotation: Annotation) => {
    try {
      await deleteAnnotation(annotation.id);
      setAnnotations((current) => current.filter((item) => item.id !== annotation.id));
      setActive(null);
    } catch (reason) {
      setNotice(`Couldn't remove the highlight: ${(reason as Error).message}`);
    }
  };

  const activeAnnotation = onPage.find((annotation) => annotation.id === active) ?? null;
  const toolbarAt = (rect: Rect | number[]) => ({
    left: Math.max(0, rect[0] * scale),
    top: Math.max(0, rect[1] * scale - 40),
  });

  return (
    <section className="relative flex min-h-0 min-w-0 flex-col border-l border-stone-200 bg-stone-200/60" data-testid="viewer">
      <ExplainPanel />
      <header className="flex items-center gap-2 border-b border-stone-200 bg-stone-50 px-3 py-2">
        <p className="min-w-0 flex-1 truncate text-sm font-medium" title={title ?? ""}>{title ?? "Viewer"}</p>
        {pdf && (
          <div className="flex shrink-0 items-center gap-1 text-sm">
            <button type="button" aria-label="Previous page" className="rounded px-2 py-0.5 hover:bg-stone-200" onClick={() => go(page - 1)}>‹</button>
            <PageField page={page} pages={pages} onGo={go} />
            <button type="button" aria-label="Next page" className="rounded px-2 py-0.5 hover:bg-stone-200" onClick={() => go(page + 1)}>›</button>
            <button type="button" aria-label="Zoom out" className="ml-1 rounded px-2 py-0.5 hover:bg-stone-200" onClick={() => setZoom(zoom / ZOOM_STEP)}>−</button>
            <button type="button" aria-label="Zoom in" className="rounded px-2 py-0.5 hover:bg-stone-200" onClick={() => setZoom(zoom * ZOOM_STEP)}>+</button>
            <button type="button" className="ml-2 rounded border border-stone-300 px-2 py-0.5 text-xs hover:bg-stone-100"
              data-testid="open-suggest" aria-pressed={suggesting} onClick={() => setSuggesting(!suggesting)}>Suggest</button>
            <button type="button" className="rounded border border-stone-300 px-2 py-0.5 text-xs hover:bg-stone-100"
              data-testid="make-copy" title="A copy with your accepted highlights and notes as PDF annotations"
              onClick={() => void makeCopy()}>Annotated PDF</button>
          </div>
        )}
        {target && (
          <button type="button" aria-label="Close the viewer" title="Close the viewer" data-testid="close-viewer"
            className="shrink-0 rounded px-2 py-0.5 text-sm hover:bg-stone-200"
            onClick={() => { setSuggesting(false); closeViewer(); }}>×</button>
        )}
      </header>
      {suggesting && docId !== null && pdf && (
        <SuggestPanel key={docId} docId={docId} page={page} pages={pages} annotations={annotations}
          onChange={reload} onGo={go} onClose={() => setSuggesting(false)} />
      )}
      {copy && (
        <p className="flex items-start gap-2 border-b border-teal-200 bg-teal-50 px-3 py-1.5 text-xs text-teal-900" data-testid="copy-made">
          <span className="flex-1">{copy.text}</span>
          <a href={copy.url} download className="font-medium underline" data-testid="copy-download">Download</a>
          <button type="button" aria-label="Dismiss" className="px-1" onClick={() => setCopy(null)}>×</button>
        </p>
      )}
      {reattaching && reattaching.docId === docId && (
        <p className="flex items-start gap-2 border-b border-teal-200 bg-teal-50 px-3 py-1.5 text-xs text-teal-900" data-testid="reattach-banner">
          <span className="flex-1">Select the passage to attach the note to, then choose Attach note here.</span>
          <button type="button" className="underline" onClick={cancelReattach}>Cancel</button>
        </p>
      )}
      {notice && (
        <p className="flex items-start gap-2 border-b border-amber-200 bg-amber-50 px-3 py-1.5 text-xs text-amber-900" data-testid="viewer-notice">
          <span className="flex-1">{notice}</span>
          <button type="button" aria-label="Dismiss" className="px-1" onClick={() => setNotice(null)}>×</button>
        </p>
      )}
      <div ref={area} className="min-h-0 flex-1 overflow-auto p-3">
        {!target && <p className="pt-16 text-center text-sm text-stone-500">Click a citation to open its page.</p>}
        {error && <p className="p-3 text-sm text-red-700" role="alert">{error}</p>}
        <div
          ref={sheet}
          className={`keep-light relative mx-auto w-fit bg-white shadow ${target ? "" : "hidden"}`}
          onMouseDown={(event) => {
            if (event.target instanceof Node && textLayer.current?.contains(event.target)) {
              textLayer.current.classList.add("selecting");
            }
          }}
          onMouseUp={readSelection}
          onKeyUp={() => readSelection()}
        >
          <canvas ref={canvas} data-testid="pdf-canvas" className="block" />
          {onPage.flatMap((annotation) => annotationBoxes(annotation, scale).map((box, index) => (
            <div
              key={`${annotation.id}-${index}`}
              data-testid="saved-highlight"
              data-annotation={annotation.id}
              data-status={annotation.status}
              className={`pointer-events-none absolute z-[1] mix-blend-multiply ${
                annotation.status === "suggested" ? "border border-dashed border-sky-700 bg-sky-200/30"
                  : HIGHLIGHT_STYLE[annotation.color] ?? HIGHLIGHT_STYLE.yellow} ${
                annotation.id === active ? "ring-2 ring-teal-700/70" : ""}`}
              style={box}
            />
          )))}
          {boxes.map((box, index) => (
            <div
              key={index}
              ref={index === 0 ? firstHighlight : undefined}
              data-testid="highlight"
              className="pointer-events-none absolute z-[1] rounded-sm bg-amber-300/35 ring-1 ring-amber-500/60"
              style={box}
            />
          ))}
          {notesOnPage.map((note) => annotationBoxes(note, scale).map((box, index) => (
            <div key={`n${note.id}-${index}`} data-testid="note-underline" data-note={note.id}
              className="pointer-events-none absolute z-[1] border-b-2 border-dashed border-teal-700/70"
              style={box} />
          )))}
          <div ref={textLayer} className="textLayer" data-testid="text-layer" />
          {notesOnPage.map((note) => {
            const [x0, y0] = note.rects[0];
            return (
              <button key={`m${note.id}`} type="button" data-testid="note-marker" data-note={note.id}
                title={`Note: ${note.body_md.slice(0, 120)}`} aria-label="Open the note"
                onMouseUp={(event) => event.stopPropagation()}
                onClick={() => showNote(note.id)}
                className="absolute z-[3] flex h-4 w-4 -translate-x-full -translate-y-1/2 items-center justify-center rounded-full bg-teal-700 text-[10px] font-bold text-white shadow hover:bg-teal-800"
                style={{ left: Math.max(16, x0 * scale - 2), top: y0 * scale }}>
                N
              </button>
            );
          })}
          {selected && (
            <SelectionToolbar
              at={toolbarAt(selected.rects[0])}
              onAsk={() => askAbout(selected.quote)}
              onHighlight={() => void highlight()}
              onNote={() => addNote(selected.rects, selected.quote)}
              onExplain={() => {
                if (docId === null) return;
                void explain({ passage: { doc_id: docId, page, rects: selected.rects, quote: selected.quote } },
                  `p. ${page}: ${selected.quote.slice(0, 50)}${selected.quote.length > 50 ? "…" : ""}`);
                clearSelection();
              }}
              onAttach={reattaching?.docId === docId ? () => void attachHere(selected.rects, selected.quote) : undefined}
            />
          )}
          {activeAnnotation && !selected && (
            <HighlightMenu
              at={toolbarAt(activeAnnotation.rects[0])}
              annotation={activeAnnotation}
              onAsk={() => askAbout(activeAnnotation.quote)}
              onNote={() => addNote(activeAnnotation.rects, activeAnnotation.quote)}
              onColor={(next) => void recolor(activeAnnotation, next)}
              onRemove={() => void remove(activeAnnotation)}
              onAccept={() => void accept(activeAnnotation)}
              onClose={() => setActive(null)}
            />
          )}
        </div>
      </div>
    </section>
  );
}

/** The page number, which can be typed over to go to a page. */
function PageField({ page, pages, onGo }: { page: number; pages: number; onGo: (page: number) => void }) {
  const [typed, setTyped] = useState<string | null>(null);
  const done = () => {
    const number = Number(typed);
    if (typed !== null && Number.isInteger(number) && number > 0) onGo(number);
    setTyped(null);
  };
  return (
    <span data-testid="page-number" className="tabular-nums">
      <input
        aria-label="Page"
        data-testid="page-input"
        inputMode="numeric"
        value={typed ?? String(page)}
        size={Math.max(2, String(pages).length)}
        className="rounded border border-transparent bg-transparent px-1 text-right hover:border-stone-300 focus:border-teal-700 focus:bg-white focus:outline-none"
        onFocus={(event) => event.target.select()}
        onChange={(event) => setTyped(event.target.value.replace(/\D/g, ""))}
        onKeyDown={(event) => {
          if (event.key === "Enter") event.currentTarget.blur();
          if (event.key === "Escape") { setTyped(null); event.currentTarget.blur(); }
        }}
        onBlur={done}
      />
      {" "}/ {pages}
    </span>
  );
}

const TOOL = "rounded px-2 py-1 hover:bg-stone-700 disabled:cursor-not-allowed disabled:text-stone-400 disabled:hover:bg-transparent";

/** Mouse-down on a tool would end the selection before the click; prevented, the selection stays. */
const keepSelection = (event: ReactMouseEvent) => event.preventDefault();

function SelectionToolbar({ at, onAsk, onHighlight, onNote, onExplain, onAttach }: {
  at: { left: number; top: number }; onAsk: () => void; onHighlight: () => void; onNote: () => void;
  onExplain: () => void; onAttach?: () => void;
}) {
  return (
    <div
      role="toolbar"
      aria-label="Selection"
      data-testid="selection-toolbar"
      className="absolute z-10 flex gap-0.5 rounded-md bg-stone-800 p-1 text-xs font-medium text-white shadow-lg"
      style={at}
      onMouseDown={keepSelection}
      onMouseUp={(event) => event.stopPropagation()}
    >
      <button type="button" className={TOOL} data-testid="tool-ask" onClick={onAsk}>Ask about this</button>
      <button type="button" className={TOOL} data-testid="tool-explain" onClick={onExplain}
        title="Line by line, for an equation or a code block">Explain</button>
      <button type="button" className={TOOL} data-testid="tool-highlight" onClick={onHighlight}>Highlight</button>
      <button type="button" className={TOOL} data-testid="tool-note" onClick={onNote}>Add note</button>
      {onAttach && (
        <button type="button" className={`${TOOL} bg-teal-700`} data-testid="tool-attach" onClick={onAttach}>Attach note here</button>
      )}
    </div>
  );
}

function HighlightMenu({ at, annotation, onAsk, onNote, onColor, onRemove, onAccept, onClose }: {
  at: { left: number; top: number }; annotation: Annotation; onAsk: () => void; onNote: () => void;
  onColor: (color: HighlightColor) => void; onRemove: () => void; onAccept: () => void; onClose: () => void;
}) {
  const suggested = annotation.status === "suggested";
  return (
    <div
      role="dialog"
      aria-label="Highlight"
      data-testid="highlight-menu"
      className="absolute z-10 max-w-md rounded-md bg-stone-800 p-1 text-xs font-medium text-white shadow-lg"
      style={at}
      onMouseDown={keepSelection}
      onMouseUp={(event) => event.stopPropagation()}
      onKeyDown={(event) => event.key === "Escape" && onClose()}
    >
      {annotation.body && (
        <p className="px-2 pb-1 pt-0.5 font-normal text-stone-200" data-testid="highlight-body">
          {annotation.origin === "ai" && <span className="text-sky-300">{suggested ? "Suggested" : "AI"}: </span>}
          {annotation.body}
        </p>
      )}
      {suggested ? (
        <div className="flex items-center gap-1">
          <button type="button" className={TOOL} data-testid="highlight-accept" onClick={onAccept}>Accept</button>
          <button type="button" className={TOOL} data-testid="highlight-reject" onClick={onRemove}>Reject</button>
          <button type="button" aria-label="Close" className={TOOL} onClick={onClose}>×</button>
        </div>
      ) : (
      <div className="flex items-center gap-1">
      {HIGHLIGHT_COLORS.map((color) => (
        <button
          key={color}
          type="button"
          aria-label={`Colour: ${color}`}
          aria-pressed={annotation.color === color}
          data-testid={`color-${color}`}
          className={`mx-0.5 h-4 w-4 rounded-full ${SWATCH[color]} ${annotation.color === color ? "ring-2 ring-white ring-offset-1 ring-offset-stone-800" : ""}`}
          onClick={() => onColor(color)}
        />
      ))}
      <button type="button" className={TOOL} data-testid="highlight-ask" onClick={onAsk}>Ask about this</button>
      <button type="button" className={TOOL} data-testid="highlight-note" onClick={onNote}>Add note</button>
      <button type="button" className={TOOL} data-testid="highlight-remove" onClick={onRemove}>Remove</button>
      <button type="button" aria-label="Close" className={TOOL} onClick={onClose}>×</button>
      </div>
      )}
    </div>
  );
}

/** Nothing open: a strip, so the chat has the room. A citation, a document or Explain opens the
 * viewer. A component of its own, so the viewer mounts only when shown: its effects measure and
 * observe elements that a collapsed viewer wouldn't have (item 9). */
export function ViewerStrip() {
  return (
    <aside className="flex min-h-0 flex-col items-center border-l border-stone-200 bg-stone-100 py-3"
      data-testid="viewer-strip" aria-label="Viewer: click a citation or a document to open it here">
      <span className="rotate-180 select-none text-xs text-stone-500 [writing-mode:vertical-rl]">
        Viewer · click a citation to open it
      </span>
    </aside>
  );
}
