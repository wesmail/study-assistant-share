"""Score gate item 1 of roadmap item 4: do the stored highlights cover the selected words?

    .venv/bin/python eval/annotations/score_selections.py RESULT.json [--save NAME]

RESULT.json comes from frontend/e2e/annotations.mjs (PHASE=select). For each selection,
the expected region is the planned words' PyMuPDF boxes, one box per line; the stored region
is the highlight the server saved. IoU = area of their intersection over the area of their
union, with each region the union of its boxes, measured on a 0.1 pt grid. The boxes the
browser sent (pdf.js's, before snapping) are scored the same way, to show what snapping does.
Gate: IoU at least 0.8 for every selection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PLAN = Path(__file__).with_name("selections.json")
RESULTS = ROOT / "eval" / "results"
GRID = 0.1
BAR = 0.8


def lines_of(words: list[dict]) -> list[list[float]]:
    lines: dict[tuple, list[list[float]]] = {}
    for word in words:
        lines.setdefault(tuple(word["line"]), []).append(word["box"])
    return [[min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
            for boxes in lines.values()]


def iou(a: list[list[float]], b: list[list[float]]) -> float:
    if not a or not b:
        return 0.0
    every = a + b
    x0, y0 = min(r[0] for r in every), min(r[1] for r in every)
    x1, y1 = max(r[2] for r in every), max(r[3] for r in every)
    shape = (int((y1 - y0) / GRID) + 2, int((x1 - x0) / GRID) + 2)

    def mask(rects):
        grid = np.zeros(shape, dtype=bool)
        for r in rects:
            grid[int((r[1] - y0) / GRID):int((r[3] - y0) / GRID), int((r[0] - x0) / GRID):int((r[2] - x0) / GRID)] = True
        return grid

    ma, mb = mask(a), mask(b)
    union = (ma | mb).sum()
    return float((ma & mb).sum() / union) if union else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--save", metavar="NAME")
    args = parser.parse_args()
    data = json.loads(args.result.read_text(encoding="utf-8"))
    results = data["results"]
    plan_path = PLAN.with_name(f"{data.get('plan', 'selections')}.json")
    plan = {s["id"]: s for s in json.loads(plan_path.read_text(encoding="utf-8"))["selections"]}
    rows = []
    for result in results:
        selection = plan[result["id"]]
        expected = lines_of(selection["words"])
        expected_text = " ".join(w["text"] for w in selection["words"])
        stored = result.get("stored")
        row = {"id": result["id"], "doc_id": selection["doc_id"], "page": selection["page"], "kind": selection["kind"],
               "expected": expected_text, "selected": result.get("selected", ""),
               "error": result.get("error"), "dragged_twice": bool(result.get("retried"))}
        if stored:
            row.update({"snapped": stored["snapped"], "stored_quote": stored["quote"],
                        "iou": round(iou(expected, stored["rects"]), 3),
                        "iou_before_snap": round(iou(expected, result["sent"]["rects"]), 3),
                        "quote_exact": stored["quote"] == expected_text})
        else:
            row.update({"snapped": False, "iou": 0.0, "iou_before_snap": 0.0, "quote_exact": False})
        rows.append(row)
        print(f"{row['id']} doc {row['doc_id']} p. {row['page']:3d} {row['kind']:8s} IoU {row['iou']:.3f} "
              f"(sent {row['iou_before_snap']:.3f}) snapped={row['snapped']} exact={row['quote_exact']}"
              + (f" ERROR {row['error']}" if row["error"] else ""))
        if not row["quote_exact"]:
            print(f"      expected: {expected_text[:100]}\n      stored:   {row.get('stored_quote', '')[:100]}")
    met = [row["iou"] >= BAR for row in rows]
    summary = {"selections": len(rows), "iou_at_least_0_8": sum(met), "snapped": sum(r["snapped"] for r in rows),
               "quote_exact": sum(r["quote_exact"] for r in rows),
               "dragged_twice": [r["id"] for r in rows if r["dragged_twice"]],
               "min_iou": min(r["iou"] for r in rows), "median_iou": float(np.median([r["iou"] for r in rows])),
               "median_iou_before_snap": float(np.median([r["iou_before_snap"] for r in rows])),
               "gate_met": all(met) and len(rows) == 20}
    print(json.dumps(summary, indent=1))
    if args.save:
        out = RESULTS / f"selections-{args.save}.json"
        out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"saved {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
