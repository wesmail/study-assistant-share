import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useStore } from "../store/store";
import { dollars, LibraryPane, removalLines, shortAuthors, stateLabel } from "./LibraryPane";

const open = { docId: 7, page: 1, regions: [], opened: 1 };
const document = (id: number) => ({
  id, path: `Author - Book ${id}-The Press (2024)-55-386.pdf`, title: `Book ${id}`, author: "Ann One, Bob Two, Cy Three",
  year: 2020, kind: "pdf", pages: 1, status: "indexed", chunks: 1,
});

beforeEach(() => {
  useStore.setState({ documents: [document(7), document(8)], scope: [], scopeMode: "library", viewer: null, hideTest: false });
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("what the library says it's searching", () => {
  it("is everything by default", () => {
    render(<LibraryPane />);
    expect(screen.getByText("Searching everything")).toBeTruthy();
  });

  it("is the open document, without crashing the page", () => {
    useStore.setState({ viewer: open, scopeMode: "document" });
    render(<LibraryPane />);
    expect(screen.getByText("Searching 1 of 2")).toBeTruthy();
  });

  it("is the ticked documents", () => {
    useStore.setState({ scope: [7, 8], scopeMode: "ticked" });
    render(<LibraryPane />);
    expect(screen.getByText("Searching 2 of 2")).toBeTruthy();
  });
});

describe("what the library shows", () => {
  it("shows each document's title, author and year, and never its file name", () => {
    render(<LibraryPane />);
    expect(screen.getByText("Book 7")).toBeTruthy();
    expect(screen.getAllByTestId("byline")[0].textContent).toBe("One et al. · 2020");
    expect(document_text().includes(".pdf")).toBe(false);
    expect(document_text().includes("55-386")).toBe(false);
  });

  it("leaves the byline out when nothing is known", () => {
    useStore.setState({ documents: [{ ...document(7), author: null, year: null }] });
    render(<LibraryPane />);
    expect(screen.queryByTestId("byline")).toBeNull();
  });
});

describe("each document's state", () => {
  it("shows the stage and its progress", () => {
    useStore.setState({ documents: [{ ...document(7), state: "parsing", progress: { done: 3, total: 10 } }] });
    render(<LibraryPane />);
    expect(screen.getByTestId("state").textContent).toBe("parsing 3/10");
  });

  it("falls back to the status from an older server", () => {
    expect(stateLabel({ ...document(7), status: "indexed" })).toBe("indexed");
  });

  it("shows why a stage failed, and that a late failure leaves the book searchable", () => {
    useStore.setState({ documents: [{ ...document(7), state: "failed", error: "RuntimeError: 2 of 9 figures weren't described",
                                      failed_stage: "describe", searchable: true }] });
    render(<LibraryPane />);
    expect(screen.getByText(/describe failed/)).toBeTruthy();
    expect(screen.getByText(/still searchable/)).toBeTruthy();
  });

  it.each([[0.004, "under $0.01"], [0.1349, "about $0.13"], [null, "an unknown amount"]])("prices %s as %s", (usd, text) => {
    expect(dollars(usd)).toBe(text);
  });
});

describe("the OK before figures are described", () => {
  const waiting = { stage: "describe" as const, figures: 12, usd: 0.1349, model: "claude-sonnet-5-5", provider: "anthropic" };

  it("shows how many figures, with what, for how much, and where they go", () => {
    useStore.setState({ documents: [{ ...document(7), state: "done", waiting }] });
    render(<LibraryPane />);
    const card = screen.getByTestId("figures-ok").textContent ?? "";
    expect(card).toContain("Describe 12 figures with claude-sonnet-5-5 for about $0.13");
    expect(card).toContain("go to Anthropic");
  });

  it("sends the estimate it showed, then reloads the library", async () => {
    const fetch = vi.fn(async () => new Response("{}", { status: 200 }));
    vi.stubGlobal("fetch", fetch);
    const loadDocuments = vi.fn(async () => undefined);
    useStore.setState({ documents: [{ ...document(7), waiting }], loadDocuments });
    render(<LibraryPane />);
    fireEvent.click(screen.getByText("Describe"));
    await waitFor(() => expect(loadDocuments).toHaveBeenCalled());
    const [url, init] = fetch.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/documents/7/figures/approve");
    expect(JSON.parse(String(init.body))).toEqual({ usd: 0.1349 });
  });

  it("shows the server's reason when the OK is refused", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response('{"detail": "The estimate is now $0.40"}', { status: 409 })));
    useStore.setState({ documents: [{ ...document(7), waiting }] });
    render(<LibraryPane />);
    fireEvent.click(screen.getByText("Describe"));
    expect(await screen.findByText("The estimate is now $0.40")).toBeTruthy();
  });
});

describe("the details form", () => {
  it("opens with the suggestion and saves what the reader typed", async () => {
    const fetch = vi.fn(async (_url: string, init?: RequestInit) =>
      new Response(init?.method === "POST" ? String(init.body) :
        '{"title": "From the pages", "author": null, "year": null, "publisher": null}', { status: 200 }));
    vi.stubGlobal("fetch", fetch);
    const loadDocuments = vi.fn(async () => undefined);
    useStore.setState({ documents: [{ ...document(7), needs_details: true }], loadDocuments });
    render(<LibraryPane />);
    fireEvent.click(screen.getByText("Add title, authors and year"));
    const title = (await screen.findByLabelText("Title")) as HTMLInputElement;
    expect(title.value).toBe("From the pages");
    fireEvent.change(screen.getByLabelText("Year"), { target: { value: "2010" } });
    fireEvent.click(screen.getByText("Save"));
    await waitFor(() => expect(loadDocuments).toHaveBeenCalled());
    const [url, init] = fetch.mock.calls[1] as unknown as [string, RequestInit];
    expect(url).toBe("/api/documents/7/details");
    expect(JSON.parse(String(init.body))).toMatchObject({ title: "From the pages", year: "2010" });
  });

  it("isn't offered for a document that has its entry", () => {
    render(<LibraryPane />);
    expect(screen.queryByText("Add title, authors and year")).toBeNull();
  });
});

describe("removing a book whose file has gone", () => {
  const preview = {
    title: "Book 7", chunks: { text: 120, figure: 9, summary: 14 }, blocks: 400, figures: 9, jobs: 6,
    annotations: 0, notes_kept: 2, figure_descriptions: 9, figure_descriptions_usd: 0.1012,
    summary_cache: 20, summary_cache_seconds: 610, folders: ["parsed/abc"], folder_bytes: 12_400_000,
  };

  it("lists only what there is to delete, with what it would cost to redo", () => {
    const lines = removalLines({ ...preview, figure_descriptions: 0, notes_kept: 0 });
    expect(lines).toEqual([
      "120 text passages, 9 figure passages and 14 summaries, from search",
      "400 blocks, 9 figures and 6 jobs",
      "12.4 MB of parsed files (parsed/abc)",
      "20 cached summary replies: writing them again would take about 11 min",
    ]);
    expect(removalLines(preview)).toContain("9 cached figure descriptions: describing them again would cost about $0.10");
    expect(removalLines(preview)).toContain("Kept: 2 notes, no longer tied to the document");
  });

  it("asks first, deletes only on the second click, and Keep deletes nothing", async () => {
    const fetch = vi.fn(async (_url: string, init?: RequestInit) =>
      new Response(init?.method === "POST" ? '{"removed": {}}' : JSON.stringify(preview), { status: 200 }));
    vi.stubGlobal("fetch", fetch);
    const loadDocuments = vi.fn(async () => undefined);
    useStore.setState({ documents: [{ ...document(7), state: "removed", status: "missing" }], loadDocuments });
    render(<LibraryPane />);

    fireEvent.click(screen.getByText("Remove from the index…"));
    expect(await screen.findByText("This deletes:")).toBeTruthy();
    expect(fetch.mock.calls.map((call) => (call[1] as RequestInit | undefined)?.method ?? "GET")).toEqual(["GET"]);
    fireEvent.click(screen.getByText("Keep"));
    expect(screen.queryByText("This deletes:")).toBeNull();
    expect(fetch).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByText("Remove from the index…"));
    fireEvent.click(await screen.findByText("Remove"));
    await waitFor(() => expect(loadDocuments).toHaveBeenCalled());
    expect(fetch.mock.calls.at(-1)?.[0]).toBe("/api/documents/7/remove");
  });

  it("isn't offered for a book whose file is there", () => {
    render(<LibraryPane />);
    expect(screen.queryByText("Remove from the index…")).toBeNull();
  });
});

describe("the test collection", () => {
  const books = () => [{ ...document(7), collection: "test" }, { ...document(8), collection: "test" }, document(9)];

  it("is listed, with a switch, until the reader hides it", () => {
    useStore.setState({ documents: books(), hideTest: false });
    render(<LibraryPane />);
    expect(screen.getAllByTestId("document")).toHaveLength(3);
    expect(screen.getByLabelText("Hide the test collection (2 books)")).toBeTruthy();
    expect(screen.getByTestId("searching").textContent).toBe("Searching everything");
  });

  it("is left out of the list and of what the header says is searched when hidden", () => {
    useStore.setState({ documents: books(), hideTest: true });
    render(<LibraryPane />);
    expect(screen.getAllByTestId("document")).toHaveLength(1);
    expect(screen.getByText("Book 9")).toBeTruthy();
    expect(screen.getByTestId("searching").textContent).toBe("Searching everything but the test collection");
  });

  it("saves the switch as a library setting", () => {
    const setHideTest = vi.fn(async () => undefined);
    useStore.setState({ documents: books(), hideTest: false, setHideTest });
    render(<LibraryPane />);
    fireEvent.click(screen.getByLabelText("Hide the test collection (2 books)"));
    expect(setHideTest).toHaveBeenCalledWith(true);
  });

  it("offers no switch when there is no test collection", () => {
    render(<LibraryPane />);
    expect(screen.queryByText(/Hide the test collection/)).toBeNull();
  });
});

describe("short author lists", () => {
  it.each([
    ["Kevin P. Murphy", "Murphy"],
    ["A One, B Two", "One and Two"],
    ["Waleed Esmail, Stuart Russell, Jana Klinge", "Esmail et al."],
    [null, null],
  ])("%s becomes %s", (author, short) => {
    expect(shortAuthors(author)).toBe(short);
  });
});

function document_text(): string {
  // All of the page, attributes included, but the Add button's file picker: its accept=".pdf,.pptx"
  // names file types, not a document.
  const copy = globalThis.document.body.cloneNode(true) as HTMLElement;
  copy.querySelector('[data-testid="add-documents-input"]')?.remove();
  return copy.innerHTML;
}

describe("the theme switch", () => {
  it("switches the page between light, dark and the system's mode, and remembers it", () => {
    window.localStorage.clear();
    render(<LibraryPane />);
    expect(screen.getByTestId("theme-system").getAttribute("aria-pressed")).toBe("true");
    fireEvent.click(screen.getByTestId("theme-dark"));
    expect(window.document.documentElement.dataset.theme).toBe("dark");
    expect(window.localStorage.getItem("study-assistant-theme")).toBe("dark");
    expect(screen.getByTestId("theme-dark").getAttribute("aria-pressed")).toBe("true");
    fireEvent.click(screen.getByTestId("theme-light"));
    expect(window.document.documentElement.dataset.theme).toBe("light");
    fireEvent.click(screen.getByTestId("theme-system"));
    expect(window.document.documentElement.dataset.theme).toBe("light");     // jsdom's "system" is light
    window.localStorage.clear();
  });
});

describe("adding documents", () => {
  const choose = (files: File[]) =>
    fireEvent.change(screen.getByTestId("add-documents-input"), { target: { files } });

  it("sends each chosen file as it is, under its own name, then refreshes the library", async () => {
    const fetch = vi.fn(async (url: string, _init?: RequestInit) =>
      new Response(JSON.stringify({ name: decodeURIComponent(url.split("/").pop()!), size: 3 }), { status: 200 }));
    vi.stubGlobal("fetch", fetch);
    const loadDocuments = vi.fn(async () => undefined);
    useStore.setState({ loadDocuments });
    render(<LibraryPane />);
    const book = new File(["%PDF-1"], "My Book (2nd ed.).pdf", { type: "application/pdf" });
    const talk = new File(["PK"], "talk.pptx");
    choose([book, talk]);
    await waitFor(() => expect(screen.getByTestId("upload-status").textContent).toContain("Added talk.pptx"));
    expect(fetch.mock.calls.map((call) => call[0])).toEqual([
      "/api/library/files/My%20Book%20(2nd%20ed.).pdf", "/api/library/files/talk.pptx"]);
    const init = fetch.mock.calls[0][1] as RequestInit;
    expect(init.method).toBe("PUT");
    expect(init.body).toBe(book);
    expect(screen.getByTestId("upload-status").textContent).toContain("Added My Book (2nd ed.).pdf");
    expect(loadDocuments).toHaveBeenCalled();
  });

  it("says why a file wasn't added", async () => {
    vi.stubGlobal("fetch", vi.fn(async () =>
      new Response('{"detail": "There is already a file called book.pdf in the library"}', { status: 409 })));
    useStore.setState({ loadDocuments: vi.fn(async () => undefined) });
    render(<LibraryPane />);
    choose([new File(["%PDF-1"], "book.pdf")]);
    await waitFor(() => expect(screen.getByTestId("upload-status").textContent).toBe(
      "Not added: There is already a file called book.pdf in the library"));
  });
});
