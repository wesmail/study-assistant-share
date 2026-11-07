import type { Region, Source } from "../api";

export interface Box {
  left: number;
  top: number;
  width: number;
  height: number;
}

/** Stored boxes are PDF points from the page's top left, so scaling is all it takes. */
export function highlightBoxes(regions: Region[], page: number, scale: number): Box[] {
  return regions
    .filter((region) => region.page === page)
    .map(({ bbox: [x0, y0, x1, y1] }) => ({
      left: x0 * scale,
      top: y0 * scale,
      width: (x1 - x0) * scale,
      height: (y1 - y0) * scale,
    }));
}

/** The page a citation opens: where the hit itself is, not the neighbours merged around it. */
export function pageOf(source: Pick<Source, "focus_page" | "regions" | "page_start">): number {
  return source.focus_page ?? source.regions[0]?.page ?? source.page_start ?? 1;
}

export function pageLabel(source: Pick<Source, "page_start" | "page_end">): string {
  const { page_start: start, page_end: end } = source;
  if (start === null) return "";
  return end === null || end === start ? `p. ${start}` : `pp. ${start}–${end}`;
}
