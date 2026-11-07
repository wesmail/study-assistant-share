/**
 * Prepares a model's text for react-markdown: math delimiters become the $ and $$
 * that remark-math reads, and [S n] citations become links.
 *
 * The text is split into segments first, so nothing inside code or existing math
 * is touched. `\\[2pt]` is a LaTeX line break, not a display opener, hence the
 * lookbehinds.
 */
const SEGMENT = new RegExp(
  [
    "```[\\s\\S]*?(?:```|$)",                       // fenced code
    "`[^`\\n]+`",                                   // inline code
    "\\$\\$[\\s\\S]*?\\$\\$",                       // $$ display $$
    "(?<!\\\\)\\\\\\[[\\s\\S]*?(?<!\\\\)\\\\\\]",   // \[ display \]
    "(?<!\\\\)\\\\\\([\\s\\S]*?(?<!\\\\)\\\\\\)",   // \( inline \)
    "(?<![\\\\$\\w])\\$[^\\s$](?:[^$\\n]*[^\\s$])?\\$(?![\\w$])", // $ inline $
  ].join("|"),
  "g",
);

const LINE_LEAD = /^([ \t>]*)((?:[-*+]|\d+[.)])[ \t]+)?/;
const ONLY_CITATIONS = /^\s*(?:\[S\d+\]\s*)+$/;
const TAG_CITATION = /\\tag\*?\{\s*(S\d+(?:\s*,\s*S\d+)*)\s*(?::\s*([^{}]*?))?\s*\}/g;

function linkCitations(prose: string): string {
  return prose.replace(/\[S(\d+)\]/g, "[S$1](#cite-$1)").replace(/\[Web (\d+)\](?!\()/g, "[Web $1](#web-$1)");
}

/** A citation in a formula's \tag can't be clicked, so it moves out and any detail stays. */
function takeTagCitations(body: string): { math: string; links: string } {
  const found: string[] = [];
  const math = body.replace(TAG_CITATION, (_tag, ids: string, detail?: string) => {
    found.push(...ids.split(",").map((id) => id.trim()));
    return detail ? `\\tag{${detail}}` : "";
  });
  return { math, links: linkCitations(found.map((id) => `[${id}]`).join(" ")) };
}

/** In a table row an unescaped | ends the cell, even inside math. */
function escapePipes(body: string): string {
  return body.replace(/\\\|/g, "\\Vert ").replace(/\|/g, "\\vert ");
}

function lineContaining(text: string, index: number): string {
  const start = text.lastIndexOf("\n", index - 1) + 1;
  const end = text.indexOf("\n", index);
  return text.slice(start, end < 0 ? text.length : end);
}

function inlineMath(body: string, text: string, index: number): string {
  if (ONLY_CITATIONS.test(body)) return linkCitations(body.trim()); // gpt-oss writes \([S1]\)
  const inTable = lineContaining(text, index).trimStart().startsWith("|");
  return `$${(inTable ? escapePipes(body) : body).trim()}$`;
}

/**
 * remark-math needs $$ alone on its line, or it renders the formula inline. Each
 * line keeps the indent of the line it came from, so math in a list stays in the list.
 */
function displayMath(body: string, text: string, index: number, end: number): string {
  const before = text.slice(text.lastIndexOf("\n", index - 1) + 1, index);
  const [lead, quote, marker = ""] = before.match(LINE_LEAD)!;
  const indent = quote + " ".repeat(marker.length);
  const { math: formula, links } = takeTagCitations(body);
  const math = formula.trim().split("\n").map((line) => indent + line.trim()).join("\n");
  const block = `$$\n${math}\n${indent}$$${links ? `\n\n${indent}${links}` : ""}`;
  const opening = before === lead ? block : `\n\n${indent}${block}`;
  const next = text[end];
  return next === undefined || next === "\n" ? opening : `${opening}\n\n${indent}`;
}

export function prepareMarkdown(text: string): string {
  let out = "";
  let last = 0;
  for (const match of text.matchAll(SEGMENT)) {
    const segment = match[0];
    const index = match.index;
    const end = index + segment.length;
    out += linkCitations(text.slice(last, index));
    last = end;
    if (segment.startsWith("\\[")) {
      out += displayMath(segment.slice(2, -2), text, index, end);
    } else if (segment.startsWith("$$")) {
      out += displayMath(segment.slice(2, -2), text, index, end);
    } else if (segment.startsWith("\\(")) {
      out += inlineMath(segment.slice(2, -2), text, index);
    } else if (segment.startsWith("$")) {
      out += inlineMath(segment.slice(1, -1), text, index);
    } else {
      out += segment;
    }
  }
  return out + linkCitations(text.slice(last));
}
