import { useState } from "react";

export function CodeBlock({ code, label = "Command" }) {
  const [copied, setCopied] = useState(false);

  function onCopy() {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    });
  }

  return (
    <div className="code-card">
      <div className="code-head">
        <span>{label}</span>
        <button type="button" className="copy-button" onClick={onCopy}>
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre>
        <code>{code}</code>
      </pre>
    </div>
  );
}

export function StatCard({ title, body }) {
  return (
    <article className="stat-card">
      <h3>{title}</h3>
      <p>{body}</p>
    </article>
  );
}

export function ChapterSection({ eyebrow, title, body, children }) {
  return (
    <section className="content-card reveal">
      {eyebrow ? <p className="eyebrow">{eyebrow}</p> : null}
      {title ? <h2>{title}</h2> : null}
      {body ? <p className="section-body">{body}</p> : null}
      {children}
    </section>
  );
}
