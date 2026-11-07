// Reads a JSON list of LaTeX strings on stdin and reports, for each, whether
// KaTeX compiles it in display mode. It uses the frontend's KaTeX, so a formula
// passes only if the app can render it; `strict: false` matches the frontend too.
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";

const require = createRequire(new URL("../frontend/", import.meta.url));
const katex = require("katex");

const formulas = JSON.parse(readFileSync(0, "utf8"));
const results = formulas.map((latex) => {
  try {
    katex.renderToString(latex, { displayMode: true, throwOnError: true, strict: false });
    return { ok: true };
  } catch (error) {
    return { ok: false, error: String(error.message).slice(0, 200) };
  }
});
process.stdout.write(JSON.stringify(results));
