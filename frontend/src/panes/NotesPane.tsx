import { useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import { changeNote, createNote, deleteAnnotation, deleteNote, fetchLostAnnotations, notesExportUrl } from "../api";
import type { Annotation, Note, Passage } from "../api";
import { matchesQuery, noteRegions, noteSource, parseTags } from "../notes/notes";
import { Markdown } from "../render/Markdown";
import { useStore } from "../store/store";
import { localTime } from "./FeedbackPane";

const ignoreCitation = () => {};

/** Notes: written for a passage in the viewer or on their own, in Markdown with LaTeX. */
export function NotesPane() {
  const setView = useStore((state) => state.setView);
  const notes = useStore((state) => state.notes);
  const notesError = useStore((state) => state.notesError);
  const loadNotes = useStore((state) => state.loadNotes);
  const draft = useStore((state) => state.noteDraft);
  const cancelDraft = useStore((state) => state.cancelNoteDraft);
  const putNote = useStore((state) => state.putNote);
  const dropNote = useStore((state) => state.dropNote);
  const shown = useStore((state) => state.shownNote);
  const viewerDoc = useStore((state) => state.viewer?.docId ?? null);
  const documents = useStore((state) => state.documents);
  const [onlyThisDocument, setOnlyThisDocument] = useState(false);
  const [query, setQuery] = useState("");
  const [writingFree, setWritingFree] = useState(false);
  const [editing, setEditing] = useState<number | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    if (notes === null) void loadNotes();
  }, [notes, loadNotes]);

  const titles = useMemo(() => new Map(documents.map((document) => [document.id, document.title])), [documents]);
  const filterDoc = onlyThisDocument ? viewerDoc : null;
  const shownNotes = (notes ?? []).filter((note) =>
    (filterDoc === null || note.doc_id === filterDoc)
    && matchesQuery(note, query, note.doc_id === null ? "" : titles.get(note.doc_id)));

  const save = async (body: string, tags: string[], passage: Passage | null) => {
    try {
      const note = await createNote({ body_md: body, tags, passage });
      putNote(note);
      setProblem(note.snapped === false ? "Saved, but the passage didn't match the page's words exactly." : null);
      return true;
    } catch (error) {
      setProblem(`Not saved: ${(error as Error).message}`);
      return false;
    }
  };

  const change = async (id: number, body: string, tags: string[]) => {
    try {
      putNote(await changeNote(id, { body_md: body, tags }));
      setEditing(null);
      setProblem(null);
    } catch (error) {
      setProblem(`Not saved: ${(error as Error).message}`);
    }
  };

  const remove = async (note: Note) => {
    if (!window.confirm("Delete this note? It can't be brought back.")) return;
    try {
      await deleteNote(note.id);
      dropNote(note.id);
    } catch (error) {
      setProblem(`Not deleted: ${(error as Error).message}`);
    }
  };

  return (
    <main className="flex min-h-0 min-w-0 flex-col" data-testid="notes-pane">
      <header className="flex items-center justify-between border-b border-stone-200 px-4 py-3">
        <h2 className="text-sm font-semibold">Notes</h2>
        <button type="button" onClick={() => setView("chat")} data-testid="notes-back"
          className="text-sm text-stone-600 hover:text-stone-900">
          Back to chat
        </button>
      </header>
      <div className="flex flex-wrap items-center gap-1.5 border-b border-stone-200 px-4 py-2 text-xs">
        <button type="button" aria-pressed={!onlyThisDocument} onClick={() => setOnlyThisDocument(false)}
          className={`rounded px-2 py-1 ${!onlyThisDocument ? "bg-teal-800 text-white" : "text-stone-600 hover:bg-stone-100"}`}>
          All
        </button>
        <button type="button" aria-pressed={onlyThisDocument} disabled={viewerDoc === null} data-testid="notes-this-document"
          title={viewerDoc === null ? "Open a document in the viewer first" : undefined}
          onClick={() => setOnlyThisDocument(true)}
          className={`rounded px-2 py-1 disabled:text-stone-400 ${onlyThisDocument ? "bg-teal-800 text-white" : "text-stone-600 hover:bg-stone-100"}`}>
          This document
        </button>
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Filter notes"
          aria-label="Filter notes" data-testid="notes-filter"
          className="min-w-0 flex-1 rounded border border-stone-300 px-2 py-1 outline-none focus:border-teal-700" />
        <button type="button" onClick={() => setWritingFree(true)} data-testid="new-note"
          className="rounded border border-stone-300 px-2 py-1 text-stone-700 hover:bg-stone-100">
          New note
        </button>
        <a href={notesExportUrl(filterDoc)} download data-testid="export-notes"
          title="A zip of Markdown files, one per document, for an Obsidian vault"
          className="rounded border border-stone-300 px-2 py-1 text-stone-700 hover:bg-stone-100">
          Export for Obsidian
        </a>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        {(problem || notesError) && (
          <p className="mb-3 rounded bg-red-50 p-2 text-sm text-red-800" role="alert">{problem ?? notesError}</p>
        )}
        {draft && (
          <section className="mb-4 rounded border border-teal-700/40 bg-white p-3" data-testid="note-draft">
            <p className="text-xs text-stone-500">New note on {draft.docTitle || "this document"}, p. {draft.page}</p>
            <blockquote className="mt-1 border-l-2 border-stone-300 pl-2 text-sm italic text-stone-600">{draft.quote}</blockquote>
            <NoteEditor key={draft.nonce} onCancel={cancelDraft} onSave={async (body, tags) => {
              const { docTitle: _title, nonce: _nonce, ...passage } = draft;
              if (await save(body, tags, passage)) cancelDraft();
            }} />
          </section>
        )}
        {writingFree && (
          <section className="mb-4 rounded border border-teal-700/40 bg-white p-3" data-testid="note-free">
            <p className="text-xs text-stone-500">New note, not tied to a passage</p>
            <NoteEditor onCancel={() => setWritingFree(false)} onSave={async (body, tags) => {
              if (await save(body, tags, null)) setWritingFree(false);
            }} />
          </section>
        )}
        {notes !== null && shownNotes.length === 0 && !draft && !writingFree && (
          <p className="pt-16 text-center text-sm text-stone-500">
            {notes.length === 0
              ? "No notes yet. Select a passage in the viewer and choose Add note, or write one with New note."
              : "No notes match."}
          </p>
        )}
        <LostHighlights />
        <ul className="space-y-3">
          {shownNotes.map((note) => (
            <NoteCard key={note.id} note={note} title={note.doc_id === null ? undefined : titles.get(note.doc_id)}
              editing={editing === note.id} shown={shown?.id === note.id ? shown.nonce : null}
              onEdit={() => setEditing(note.id)} onCancel={() => setEditing(null)}
              onSave={(body, tags) => change(note.id, body, tags)} onDelete={() => void remove(note)} />
          ))}
        </ul>
      </div>
    </main>
  );
}

function NoteCard({ note, title, editing, shown, onEdit, onCancel, onSave, onDelete }: {
  note: Note; title: string | undefined; editing: boolean; shown: number | null;
  onEdit: () => void; onCancel: () => void; onSave: (body: string, tags: string[]) => Promise<void>; onDelete: () => void;
}) {
  const openSource = useStore((state) => state.openSource);
  const startReattach = useStore((state) => state.startReattach);
  const card = useRef<HTMLLIElement>(null);
  useEffect(() => {
    if (shown !== null) card.current?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [shown]);
  const lost = note.anchor_status === "unanchored";
  const source = lost ? `${noteSource(note, title).replace(/, p\. \d+$/, "")}, was p. ${note.page}` : noteSource(note, title);
  const canOpen = note.doc_id !== null && note.page !== null && !lost;
  return (
    <li ref={card} data-testid="note" data-note={note.id}
      className={`rounded border bg-white p-3 ${shown !== null ? "border-teal-700 ring-1 ring-teal-700/30" : "border-stone-200"}`}>
      <div className="flex items-start gap-2 text-xs text-stone-500">
        {canOpen ? (
          <button type="button" className="min-w-0 flex-1 truncate text-left text-teal-800 hover:underline" title={source}
            data-testid="note-source"
            onClick={() => openSource({ doc_id: note.doc_id as number, focus_page: note.page, page_start: note.page,
              regions: noteRegions(note) })}>
            {source}
          </button>
        ) : (
          <span className="min-w-0 flex-1 truncate" title={source}>{source}</span>
        )}
        <span className="shrink-0" title={`Written ${localTime(note.created_at)}`}>{localTime(note.updated_at)}</span>
      </div>
      {lost && (
        <div className="mt-1 rounded bg-amber-50 p-2 text-xs text-amber-900" data-testid="note-lost">
          The book's file changed and this passage wasn't found in it again, so the note isn't shown on the page.{" "}
          {note.doc_id !== null && (
            <button type="button" className="font-medium underline" data-testid="note-reattach" onClick={() => startReattach(note)}>
              Attach to a passage
            </button>
          )}
        </div>
      )}
      {note.quote && (
        <blockquote className="mt-1 line-clamp-4 border-l-2 border-stone-300 pl-2 text-sm italic text-stone-600">{note.quote}</blockquote>
      )}
      {editing ? (
        <NoteEditor initialBody={note.body_md} initialTags={note.tags} onCancel={onCancel} onSave={onSave} />
      ) : (
        <>
          <div className="answer mt-2" data-testid="note-body">
            <Markdown text={note.body_md} streaming={false} onCite={ignoreCitation} />
          </div>
          <div className="mt-2 flex flex-wrap items-center gap-1.5 text-xs">
            {note.tags.map((tag) => (
              <span key={tag} className="rounded bg-stone-100 px-1.5 py-0.5 text-stone-600">#{tag}</span>
            ))}
            <span className="ml-auto flex gap-3">
              <button type="button" onClick={onEdit} data-testid="note-edit" className="text-stone-600 hover:text-stone-900">Edit</button>
              <button type="button" onClick={onDelete} data-testid="note-delete" className="text-stone-600 hover:text-red-800">Delete</button>
            </span>
          </div>
        </>
      )}
    </li>
  );
}

/** Markdown with LaTeX ($…$, $$…$$), previewed as the notes will show; Ctrl+Enter saves. */
function NoteEditor({ initialBody = "", initialTags = [], onSave, onCancel }: {
  initialBody?: string; initialTags?: string[];
  onSave: (body: string, tags: string[]) => Promise<void> | void; onCancel: () => void;
}) {
  const [body, setBody] = useState(initialBody);
  const [tags, setTags] = useState(initialTags.join(" "));
  const [saving, setSaving] = useState(false);
  const box = useRef<HTMLTextAreaElement>(null);
  useEffect(() => box.current?.focus(), []);
  const submit = async () => {
    if (!body.trim() || saving) return;
    setSaving(true);
    try {
      await onSave(body, parseTags(tags));
    } finally {
      setSaving(false);
    }
  };
  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      void submit();
    }
    if (event.key === "Escape") onCancel();
  };
  return (
    <div className="mt-2" data-testid="note-editor">
      <textarea ref={box} value={body} onChange={(event) => setBody(event.target.value)} onKeyDown={onKeyDown}
        rows={5} aria-label="Note" data-testid="note-text"
        placeholder="Markdown, with maths as $...$ or $$...$$. Ctrl+Enter saves."
        className="w-full resize-y rounded border border-stone-300 px-2 py-1.5 font-mono text-[13px] outline-none focus:border-teal-700" />
      <input value={tags} onChange={(event) => setTags(event.target.value)} onKeyDown={onKeyDown}
        aria-label="Tags" data-testid="note-tags" placeholder="Tags, e.g. relativity exam-2"
        className="mt-1 w-full rounded border border-stone-300 px-2 py-1 text-xs outline-none focus:border-teal-700" />
      {body.trim() && (
        <div className="mt-2 rounded bg-stone-50 p-2" data-testid="note-preview">
          <p className="mb-1 text-[11px] uppercase tracking-wide text-stone-400">Preview</p>
          <div className="answer">
            <Markdown text={body} streaming={false} onCite={ignoreCitation} />
          </div>
        </div>
      )}
      <div className="mt-2 flex justify-end gap-2 text-sm">
        <button type="button" onClick={onCancel} className="rounded px-3 py-1 text-stone-600 hover:bg-stone-100">Cancel</button>
        <button type="button" onClick={() => void submit()} disabled={!body.trim() || saving} data-testid="note-save"
          className="rounded bg-teal-800 px-3 py-1 font-medium text-white disabled:opacity-50">
          {saving ? "Saving…" : "Save"}
        </button>
      </div>
    </div>
  );
}

/** Highlights that lost their passage when their book's file changed: shown so they can be removed. */
function LostHighlights() {
  const [lost, setLost] = useState<Annotation[]>([]);
  const [problem, setProblem] = useState<string | null>(null);
  useEffect(() => {
    fetchLostAnnotations().then(setLost, () => setLost([]));
  }, []);
  if (lost.length === 0) return null;
  const remove = async (annotation: Annotation) => {
    if (!window.confirm("Remove this highlight? It can't be brought back.")) return;
    try {
      await deleteAnnotation(annotation.id);
      setLost((current) => current.filter((item) => item.id !== annotation.id));
    } catch (error) {
      setProblem(`Not removed: ${(error as Error).message}`);
    }
  };
  return (
    <details className="mb-4 rounded border border-amber-200 bg-amber-50 p-3 text-sm" data-testid="lost-highlights">
      <summary className="cursor-pointer font-medium text-amber-900">
        {lost.length} highlight{lost.length === 1 ? "" : "s"} lost {lost.length === 1 ? "its" : "their"} passage when a book changed
      </summary>
      {problem && <p className="mt-2 text-red-800" role="alert">{problem}</p>}
      <ul className="mt-2 space-y-2">
        {lost.map((annotation) => (
          <li key={annotation.id} className="flex items-start gap-2" data-testid="lost-highlight">
            <span className="min-w-0 flex-1">
              <span className="text-xs text-stone-500">{annotation.doc_title || "Untitled"}, was p. {annotation.page}</span>
              <span className="block italic text-stone-700">{annotation.quote}</span>
            </span>
            <button type="button" className="text-xs text-stone-600 hover:text-red-800" onClick={() => void remove(annotation)}>
              Remove
            </button>
          </li>
        ))}
      </ul>
    </details>
  );
}
