import { useEffect, useState } from "react";

interface Props {
  code: string;
  language: string;
  streaming: boolean;
  onExplain?: () => void;
}

/** Shiki loads on first use. A block still streaming stays plain until it's complete. */
export function CodeBlock({ code, language, streaming, onExplain }: Props) {
  const [html, setHtml] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (streaming) return;
    let current = true;
    import("shiki")
      .then(({ codeToHtml }) => codeToHtml(code, { lang: language || "text", theme: "github-light" }))
      .then((highlighted) => {
        if (current) setHtml(highlighted);
      })
      .catch(() => {
        if (current) setHtml(null);      // an unknown language stays plain
      });
    return () => {
      current = false;
    };
  }, [code, language, streaming]);

  const copy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="group relative rounded border border-stone-200 bg-white text-sm" data-testid="code-block">
      <button
        type="button"
        onClick={copy}
        className="absolute top-1.5 right-1.5 rounded border border-stone-200 bg-white px-2 py-0.5 text-xs text-stone-600 opacity-0 group-hover:opacity-100 focus:opacity-100"
      >
        {copied ? "Copied" : "Copy"}
      </button>
      {onExplain && (
        <button type="button" onClick={onExplain} data-testid="explain-code"
          className="absolute top-1.5 right-16 rounded border border-stone-200 bg-white px-2 py-0.5 text-xs text-stone-600 opacity-0 group-hover:opacity-100 focus:opacity-100">
          Explain
        </button>
      )}
      {html ? (
        <div
          className="overflow-x-auto p-3 [&_pre]:!bg-transparent"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      ) : (
        <pre className="overflow-x-auto p-3 font-mono">{code}</pre>
      )}
    </div>
  );
}
