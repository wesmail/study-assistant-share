import { describe, expect, it } from "vitest";
import type { Note } from "../api";
import { matchesQuery, noteRegions, noteSource, parseTags } from "./notes";

export const note = (change: Partial<Note> = {}): Note => ({
  id: 3, doc_id: 7, doc_title: "Spacetime", page: 12, rects: [[10, 20, 100, 30], [10, 32, 60, 42]], bbox: [10, 20, 100, 42],
  quote: "the proper time", prefix: "", suffix: "", body_md: "Invariant: $\\tau$", tags: ["relativity"],
  created_at: "2026-11-06 10:00:00", updated_at: "2026-11-06 10:00:00", ...change,
});

describe("notes helpers", () => {
  it("reads tags typed with commas, spaces or #", () => {
    expect(parseTags("#a, b  c,,#A d")).toEqual(["a", "b", "c", "d"]);
    expect(parseTags("  ")).toEqual([]);
  });

  it("filters on every word, in the note, quote, tags or title", () => {
    expect(matchesQuery(note(), "PROPER invariant")).toBe(true);
    expect(matchesQuery(note(), "relativity spacetime")).toBe(true);
    expect(matchesQuery(note(), "proper missing")).toBe(false);
    expect(matchesQuery(note(), "")).toBe(true);
  });

  it("gives the passage as regions for the viewer", () => {
    expect(noteRegions(note())).toEqual([{ page: 12, bbox: [10, 20, 100, 30] }, { page: 12, bbox: [10, 32, 60, 42] }]);
    expect(noteRegions(note({ page: null, rects: [] }))).toEqual([]);
  });

  it("says where a note is from", () => {
    expect(noteSource(note(), "Spacetime Physics")).toBe("Spacetime Physics, p. 12");
    expect(noteSource(note({ doc_id: null }), undefined)).toBe("Spacetime (removed from the library), p. 12");
    expect(noteSource(note({ doc_id: null, doc_title: "", page: null }), undefined)).toBe("No passage");
  });
});
