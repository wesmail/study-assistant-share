import { describe, expect, it } from "vitest";
import { annotationAt, annotationBoxes, askAboutText, selectedText, selectionRects } from "./selection";

const rect = (left: number, top: number, right: number, bottom: number) => ({ left, top, right, bottom });

describe("selectionRects", () => {
  it("turns screen boxes into PDF points from the page's top left", () => {
    expect(selectionRects([rect(110, 220, 210, 240)], { left: 100, top: 200 }, 2)).toEqual([[5, 10, 55, 20]]);
  });

  it("drops empty boxes and boxes inside others (a Range repeats them per element)", () => {
    const found = selectionRects([rect(0, 0, 100, 10), rect(10, 1, 50, 9), rect(5, 5, 5, 20), rect(0, 0, 100, 10)],
      { left: 0, top: 0 }, 1);
    expect(found).toEqual([[0, 0, 100, 10]]);
  });

  it("merges touching runs on one line, keeps lines and columns apart", () => {
    const found = selectionRects([
      rect(0, 0, 40, 10), rect(42, 1, 90, 11), // one line, two runs
      rect(300, 0, 340, 10), // the other column, same height
      rect(0, 14, 80, 24), // the next line
    ], { left: 0, top: 0 }, 1);
    expect(found).toEqual([[0, 0, 90, 11], [300, 0, 340, 10], [0, 14, 80, 24]]);
  });
});

describe("the rest", () => {
  it("makes selected text one line", () => {
    expect(selectedText("  proper\n time  is\tinvariant ")).toBe("proper time is invariant");
  });

  it("quotes the passage for the question box", () => {
    expect(askAboutText("proper time", "Relativity", 12)).toBe("> proper time\n\n(Relativity, p. 12)\n\n");
    expect(askAboutText("x", undefined, 3)).toBe("> x\n\n(p. 3)\n\n");
  });

  it("scales a highlight's boxes", () => {
    expect(annotationBoxes({ rects: [[10, 20, 30, 25]] }, 2)).toEqual([{ left: 20, top: 40, width: 40, height: 10 }]);
  });

  it("finds the newest highlight under a point", () => {
    const list = [{ id: 1, rects: [[0, 0, 50, 10]] }, { id: 2, rects: [[20, 0, 80, 10]] }] as
      { id: number; rects: [number, number, number, number][] }[];
    expect(annotationAt(list, 30, 5)?.id).toBe(2);
    expect(annotationAt(list, 10, 5)?.id).toBe(1);
    expect(annotationAt(list, 10, 50)).toBeNull();
  });
});
