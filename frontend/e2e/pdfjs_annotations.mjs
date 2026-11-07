// Open PDFs with pdf.js itself (the viewer's own library) and count their annotations by type,
// per page: gate item 2 of roadmap item 4 ("the copy opens in pdf.js").
//   node e2e/pdfjs_annotations.mjs a.pdf b.pdf ...   prints JSON {file: {pages, annotations: {page: {Highlight: n, ...}}}}
import { readFileSync } from "node:fs";
import { getDocument } from "pdfjs-dist/legacy/build/pdf.mjs";

const out = {};
for (const file of process.argv.slice(2)) {
  try {
    const task = getDocument({ data: new Uint8Array(readFileSync(file)), isEvalSupported: false });
    const pdf = await task.promise;
    const annotations = {};
    for (let number = 1; number <= pdf.numPages; number++) {
      const page = await pdf.getPage(number);
      for (const annotation of await page.getAnnotations()) {
        annotations[number] ??= {};
        annotations[number][annotation.subtype] = (annotations[number][annotation.subtype] ?? 0) + 1;
      }
    }
    out[file] = { pages: pdf.numPages, annotations };
    await task.destroy();
  } catch (error) {
    out[file] = { error: String(error) };
  }
}
console.log(JSON.stringify(out));
