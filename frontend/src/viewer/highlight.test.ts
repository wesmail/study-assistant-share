import { describe, expect, it } from "vitest";
import type { Source } from "../api";
import { highlightBoxes, pageLabel, pageOf } from "./highlight";

const source = (overrides: Partial<Source>): Source => ({
  id: 1, chunk_id: 1, doc_id: 1, doc_title: "Book", page_start: 3, page_end: 3,
  focus_page: null, heading_path: [], regions: [], ...overrides,
});

describe("highlightBoxes", () => {
  it("scales a box", () => {
    expect(highlightBoxes([{ page: 2, bbox: [10, 20, 30, 60] }], 2, 1.5)).toEqual([
      { left: 15, top: 30, width: 30, height: 60 },
    ]);
  });

  it("keeps only the page on screen", () => {
    const regions = [{ page: 1, bbox: [0, 0, 1, 1] }, { page: 2, bbox: [0, 0, 1, 1] }] as Source["regions"];
    expect(highlightBoxes(regions, 2, 1)).toHaveLength(1);
  });
});

describe("pageOf", () => {
  it("prefers the page of the hit over a neighbour's", () => {
    expect(pageOf(source({ focus_page: 12, regions: [{ page: 11, bbox: [0, 0, 1, 1] }] }))).toBe(12);
  });

  it("uses the first region when no focus is known", () => {
    expect(pageOf(source({ regions: [{ page: 9, bbox: [0, 0, 1, 1] }] }))).toBe(9);
  });

  it("falls back to the chunk's first page, then to page 1", () => {
    expect(pageOf(source({ page_start: 4 }))).toBe(4);
    expect(pageOf(source({ page_start: null }))).toBe(1);
  });
});

describe("pageLabel", () => {
  it("names one page or a range", () => {
    expect(pageLabel(source({}))).toBe("p. 3");
    expect(pageLabel(source({ page_end: 5 }))).toBe("pp. 3–5");
    expect(pageLabel(source({ page_start: null, page_end: null }))).toBe("");
  });
});
