import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { note } from "../notes/notes.test";
import { useStore } from "../store/store";
import { NotesPane } from "./NotesPane";

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });
/** The server: the lost highlights list answers [] unless given, everything else `reply`. */
const server = (reply: () => unknown, lost: unknown[] = []) => vi.fn(async (url: string) =>
  json(String(url).endsWith("/api/annotations/lost") ? { annotations: lost } : reply()));
const calls = (fetch: ReturnType<typeof server>) =>
  fetch.mock.calls.filter(([url]) => !String(url).endsWith("/api/annotations/lost")) as unknown as [string, RequestInit][];

beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn();
  useStore.setState({ view: "notes", viewer: null, documents: [], notes: [], noteDraft: null, shownNote: null, notesError: null });
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("the notes panel", () => {
  it("shows each note with its source, quote, rendered body and tags", () => {
    useStore.setState({ notes: [note()] });
    render(<NotesPane />);
    expect(screen.getByTestId("note-source").textContent).toBe("Spacetime, p. 12");
    expect(screen.getByText("the proper time")).toBeTruthy();
    expect(screen.getByTestId("note-body").querySelector(".katex")).toBeTruthy();
    expect(screen.getByText("#relativity")).toBeTruthy();
  });

  it("opens the note's passage in the viewer, highlighted", () => {
    useStore.setState({ notes: [note()] });
    render(<NotesPane />);
    fireEvent.click(screen.getByTestId("note-source"));
    const viewer = useStore.getState().viewer!;
    expect(viewer.docId).toBe(7);
    expect(viewer.page).toBe(12);
    expect(viewer.regions).toHaveLength(2);
  });

  it("filters", () => {
    useStore.setState({ notes: [note(), note({ id: 4, body_md: "Something else", quote: "", tags: [] })] });
    render(<NotesPane />);
    expect(screen.getAllByTestId("note")).toHaveLength(2);
    fireEvent.change(screen.getByTestId("notes-filter"), { target: { value: "else" } });
    expect(screen.getAllByTestId("note")).toHaveLength(1);
  });

  it("saves a note for a passage from the viewer, with a preview while writing", async () => {
    const fetch = server(() => note({ id: 9, body_md: "New $x$", tags: ["a", "b"] }));
    vi.stubGlobal("fetch", fetch);
    useStore.setState({ noteDraft: { doc_id: 7, page: 12, rects: [[1, 2, 3, 4]], quote: "the proper time", docTitle: "Spacetime", nonce: 1 } });
    render(<NotesPane />);
    expect(screen.getByTestId("note-draft").textContent).toContain("New note on Spacetime, p. 12");
    fireEvent.change(screen.getByTestId("note-text"), { target: { value: "New $x$" } });
    expect(screen.getByTestId("note-preview").querySelector(".katex")).toBeTruthy();
    fireEvent.change(screen.getByTestId("note-tags"), { target: { value: "#a, b" } });
    fireEvent.click(screen.getByTestId("note-save"));
    await waitFor(() => expect(useStore.getState().noteDraft).toBeNull());
    const sent = JSON.parse(calls(fetch)[0][1].body as string);
    expect(sent).toEqual({ body_md: "New $x$", tags: ["a", "b"], passage: { doc_id: 7, page: 12, rects: [[1, 2, 3, 4]], quote: "the proper time" } });
    expect(useStore.getState().notes?.[0].id).toBe(9);
  });

  it("asks before deleting", async () => {
    const fetch = server(() => ({ ok: true }));
    vi.stubGlobal("fetch", fetch);
    useStore.setState({ notes: [note()] });
    render(<NotesPane />);
    vi.spyOn(window, "confirm").mockReturnValueOnce(false);
    fireEvent.click(screen.getByTestId("note-delete"));
    expect(calls(fetch)).toHaveLength(0);
    vi.spyOn(window, "confirm").mockReturnValueOnce(true);
    fireEvent.click(screen.getByTestId("note-delete"));
    await waitFor(() => expect(useStore.getState().notes).toEqual([]));
  });

  it("exports all notes, or this document's", () => {
    useStore.setState({ notes: [note()], viewer: { docId: 7, page: 1, regions: [], opened: 1 } });
    render(<NotesPane />);
    expect(screen.getByTestId("export-notes").getAttribute("href")).toBe("/api/notes/export");
    fireEvent.click(screen.getByTestId("notes-this-document"));
    expect(screen.getByTestId("export-notes").getAttribute("href")).toBe("/api/notes/export?doc_id=7");
  });
});

describe("after a book's file changed (step 4)", () => {
  it("marks a lost note, and attaching it opens its book with the toolbar's Attach", () => {
    vi.stubGlobal("fetch", server(() => ({})));
    useStore.setState({ notes: [note({ anchor_status: "unanchored" })], reattaching: null });
    render(<NotesPane />);
    expect(screen.getByTestId("note-lost").textContent).toContain("wasn't found");
    expect(screen.getByText("Spacetime, was p. 12")).toBeTruthy();
    expect(screen.queryByTestId("note-source")).toBeNull();
    fireEvent.click(screen.getByTestId("note-reattach"));
    expect(useStore.getState().reattaching).toEqual({ noteId: 3, docId: 7 });
    expect(useStore.getState().viewer?.docId).toBe(7);
  });

  it("lists lost highlights, and removes one only when asked", async () => {
    const lost = [{ id: 5, doc_id: 7, doc_title: "Spacetime", page: 4, rects: [], bbox: [0, 0, 1, 1], quote: "an old line",
      prefix: "", suffix: "", kind: "highlight", color: "yellow", body: null, origin: "user", status: "accepted",
      created_at: "", anchor_status: "unanchored" }];
    const fetch = server(() => ({ ok: true }), lost);
    vi.stubGlobal("fetch", fetch);
    render(<NotesPane />);
    await waitFor(() => expect(screen.getByTestId("lost-highlights")).toBeTruthy());
    expect(screen.getByTestId("lost-highlight").textContent).toContain("an old line");
    vi.spyOn(window, "confirm").mockReturnValueOnce(true);
    fireEvent.click(screen.getByText("Remove"));
    await waitFor(() => expect(screen.queryByTestId("lost-highlights")).toBeNull());
    expect(calls(fetch)[0][0]).toBe("/api/annotations/5");
  });
});
