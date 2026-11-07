import type { Annotation } from "../api";
import type { Box } from "./highlight";

export type Rect = [number, number, number, number];

interface ClientRect {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

/** A selection's boxes on screen, as PDF points from the page's top left: tiny and repeated
 * boxes dropped (a Range gives one per element as well as per text run), boxes on one line
 * merged when they touch. The server snaps them to the page's words. */
export function selectionRects(rects: ClientRect[], origin: { left: number; top: number }, scale: number): Rect[] {
  const boxes: Rect[] = rects
    .filter((rect) => rect.right - rect.left >= 0.5 && rect.bottom - rect.top >= 0.5)
    .map((rect) => [
      (rect.left - origin.left) / scale,
      (rect.top - origin.top) / scale,
      (rect.right - origin.left) / scale,
      (rect.bottom - origin.top) / scale,
    ]);
  const inside = (a: Rect, b: Rect) => a[0] >= b[0] - 0.5 && a[1] >= b[1] - 0.5 && a[2] <= b[2] + 0.5 && a[3] <= b[3] + 0.5;
  const kept = boxes.filter((box, index) => !boxes.some((other, j) => j !== index && inside(box, other)
    && (!inside(other, box) || j < index)));
  kept.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const merged: Rect[] = [];
  for (const box of kept) {
    const height = (other: Rect) => Math.min(box[3] - box[1], other[3] - other[1]);
    const touching = merged.findIndex((other) =>
      Math.min(box[3], other[3]) - Math.max(box[1], other[1]) > 0.5 * height(other)
      && box[0] - other[2] <= height(other) && other[0] - box[2] <= height(other));
    if (touching >= 0) {
      const other = merged[touching];
      merged[touching] = [Math.min(other[0], box[0]), Math.min(other[1], box[1]),
        Math.max(other[2], box[2]), Math.max(other[3], box[3])];
    } else {
      merged.push(box);
    }
  }
  merged.sort((a, b) => a[1] - b[1] || a[0] - b[0]);
  return merged.map((box) => box.map((value) => Math.round(value * 100) / 100) as Rect);
}

/** The selected text with line breaks and runs of spaces made single spaces. */
export function selectedText(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

/** What "Ask about this" puts in the question box: the passage quoted, where it's from, and
 * room for the question. */
export function askAboutText(quote: string, title: string | undefined, page: number): string {
  return `> ${quote}\n\n(${title ? `${title}, ` : ""}p. ${page})\n\n`;
}

/** A saved highlight's boxes on the page on screen. */
export function annotationBoxes(annotation: Pick<Annotation, "rects">, scale: number): Box[] {
  return annotation.rects.map(([x0, y0, x1, y1]) => ({
    left: x0 * scale, top: y0 * scale, width: (x1 - x0) * scale, height: (y1 - y0) * scale,
  }));
}

/** The highlight under a point on the page (in PDF points), the newest one if several overlap. */
export function annotationAt<T extends Pick<Annotation, "rects" | "id">>(annotations: T[], x: number, y: number): T | null {
  const hits = annotations.filter((annotation) =>
    annotation.rects.some(([x0, y0, x1, y1]) => x >= x0 && x <= x1 && y >= y0 && y <= y1));
  return hits.reduce<T | null>((best, hit) => (best === null || hit.id > best.id ? hit : best), null);
}
