import type { Note, Region } from "../api";

/** Tags as typed: separated by commas or spaces, with or without #. The server cleans them again. */
export function parseTags(text: string): string[] {
  const seen = new Set<string>();
  return text.split(/[,\s]+/).map((tag) => tag.replace(/^#+/, "")).filter((tag) => {
    const key = tag.toLowerCase();
    if (!tag || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

/** Whether every word of the filter is in the note, its quote, its tags or its book's title. */
export function matchesQuery(note: Note, query: string, title = ""): boolean {
  const haystack = [note.body_md, note.quote, note.tags.join(" "), title || note.doc_title].join("\n").toLowerCase();
  return query.toLowerCase().split(/\s+/).filter(Boolean).every((word) => haystack.includes(word));
}

/** A note's passage as regions the viewer highlights. */
export function noteRegions(note: Pick<Note, "page" | "rects">): Region[] {
  return note.page === null ? [] : note.rects.map((bbox) => ({ page: note.page as number, bbox }));
}

/** Where a note is from, for its card: the book and page, or that it has none. */
export function noteSource(note: Note, title: string | undefined): string {
  if (note.doc_id === null) {
    return note.doc_title ? `${note.doc_title} (removed from the library)${note.page ? `, p. ${note.page}` : ""}` : "No passage";
  }
  return `${title || note.doc_title || "Untitled"}${note.page ? `, p. ${note.page}` : ""}`;
}
