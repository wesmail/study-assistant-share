import { isValidElement, memo } from "react";
import type { ComponentProps, ReactElement, ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { CodeBlock } from "./CodeBlock";
import { prepareMarkdown } from "./prepareMarkdown";

const KATEX_OPTIONS = { throwOnError: false, strict: false, errorColor: "inherit" };
const REMARK = [remarkGfm, remarkMath];
interface HastNode {
  type: string;
  tagName?: string;
  value?: string;
  children?: HastNode[];
}

const textOf = (node: HastNode): string => node.value ?? (node.children ?? []).map(textOf).join("");

/** The TeX KaTeX keeps in its output, in <annotation encoding="application/x-tex">. */
function texOf(node: HastNode | undefined): string | null {
  if (!node) return null;
  if (node.tagName === "annotation") return textOf(node);
  for (const child of node.children ?? []) {
    const found = texOf(child);
    if (found !== null) return found;
  }
  return null;
}

const REHYPE: ComponentProps<typeof ReactMarkdown>["rehypePlugins"] = [[rehypeKatex, KATEX_OPTIONS]];

interface Props {
  text: string;
  streaming: boolean;
  onCite: (id: number) => void;
  /** Where [Web n] goes: the page it cites, in a new tab. */
  webUrl?: (number: number) => string | undefined;
  /** An Explain button on each code block and display equation (roadmap item 4). */
  onExplain?: (text: string, kind: "equation" | "code") => void;
}

/** Only the app's own pictures are shown. An image elsewhere would be fetched by the browser the
 * moment the answer renders, and its address can carry what a page told the model to leak. */
function ownImage(src: string | undefined): boolean {
  if (!src) return false;
  if (src.startsWith("data:image/")) return true;
  try {
    return new URL(src, window.location.href).origin === window.location.origin;
  } catch {
    return false;
  }
}

function hostOf(src: string | undefined): string {
  try {
    return new URL(src ?? "", window.location.href).host || "another site";
  } catch {
    return "another site";
  }
}

function codeOf(children: ReactNode): { code: string; language: string } {
  const element = (Array.isArray(children) ? children[0] : children) as ReactElement<{
    className?: string;
    children?: ReactNode;
  }>;
  if (!isValidElement(element)) return { code: String(children), language: "" };
  const language = /language-([\w+-]+)/.exec(element.props.className ?? "")?.[1] ?? "";
  return { code: String(element.props.children ?? "").replace(/\n$/, ""), language };
}

export const Markdown = memo(function Markdown({ text, streaming, onCite, webUrl, onExplain }: Props) {
  return (
    <div className="answer">
      <ReactMarkdown
        remarkPlugins={REMARK}
        rehypePlugins={REHYPE}
        components={{
          pre: ({ children }) => {
            const { code, language } = codeOf(children);
            return <CodeBlock code={code} language={language} streaming={streaming}
              onExplain={onExplain && !streaming ? () => onExplain(code, "code") : undefined} />;
          },
          ...(onExplain && !streaming ? {
            span: ({ node, children, ...props }: ComponentProps<"span"> & { node?: unknown }) => {
              const tex = String(props.className ?? "").includes("katex-display") ? texOf(node as HastNode) : null;
              if (tex === null) return <span {...props}>{children}</span>;
              return (
                <span className="group relative block">
                  <span {...props}>{children}</span>
                  <button type="button" data-testid="explain-equation" onClick={() => onExplain(tex.trim(), "equation")}
                    className="absolute top-0 right-0 rounded border border-stone-200 bg-white px-1.5 py-0.5 text-xs text-stone-600 opacity-0 group-hover:opacity-100 focus:opacity-100">
                    Explain
                  </button>
                </span>
              );
            },
          } : {}),
          img: ({ src, alt }) =>
            ownImage(typeof src === "string" ? src : undefined) ? (
              <img src={src as string} alt={alt ?? ""} />
            ) : (
              <span className="text-xs text-stone-500" data-testid="blocked-image">
                [image from {hostOf(typeof src === "string" ? src : undefined)} not loaded]
              </span>
            ),
          a: ({ href, children }) => {
            const web = /^#web-(\d+)$/.exec(href ?? "");
            if (web) {
              const url = webUrl?.(Number(web[1]));
              return url ? (
                <a className="citation" href={url} target="_blank" rel="noreferrer noopener" data-testid="web-citation">
                  {children}
                </a>
              ) : (
                <span className="citation">{children}</span>
              );
            }
            const cited = /^#cite-(\d+)$/.exec(href ?? "");
            if (cited) {
              return (
                <a
                  className="citation"
                  href={href}
                  data-testid="citation"
                  onClick={(event) => {
                    event.preventDefault();
                    onCite(Number(cited[1]));
                  }}
                >
                  {children}
                </a>
              );
            }
            return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
          },
        }}
      >
        {prepareMarkdown(text)}
      </ReactMarkdown>
    </div>
  );
});
