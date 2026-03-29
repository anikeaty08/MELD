import { useEffect, useState, startTransition } from "react";
import {
  HashRouter,
  Link,
  NavLink,
  Navigate,
  Route,
  Routes,
  useLocation,
  useNavigate,
  useSearchParams,
} from "react-router-dom";
import Prism from "prismjs";
import navigation from "./navigation.json";
import { pages, pagesByPath, siteMeta } from "./docsData.js";

function copyText(text) {
  return navigator.clipboard?.writeText(text) ?? Promise.resolve();
}

function escapeHtml(text) {
  return text.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function inlineMarkup(text) {
  return escapeHtml(text)
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>')
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}

function RichText({ as = "p", text, className = "" }) {
  const Tag = as;
  return <Tag className={className} dangerouslySetInnerHTML={{ __html: inlineMarkup(text) }} />;
}

function CodeBlock({ language = "text", code }) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    Prism.highlightAll();
  }, [language, code]);

  return (
    <div className="code-block">
      <div className="code-toolbar">
        <span className="code-language">{language}</span>
        <button
          type="button"
          className="copy-button"
          onClick={() => {
            copyText(code).then(() => {
              setCopied(true);
              window.setTimeout(() => setCopied(false), 1200);
            });
          }}
        >
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className={`language-${language}`}>
        <code className={`language-${language}`}>{code}</code>
      </pre>
    </div>
  );
}

function HeadingLink({ page, section }) {
  const [copied, setCopied] = useState(false);

  return (
    <button
      type="button"
      className="heading-anchor"
      aria-label={`Copy link to ${section.title}`}
      onClick={() => {
        const url = `${window.location.origin}${window.location.pathname}#${page.path}?section=${section.id}`;
        copyText(url).then(() => {
          setCopied(true);
          window.setTimeout(() => setCopied(false), 1200);
        });
      }}
    >
      {copied ? "Copied" : "#"}
    </button>
  );
}

function Heading({ page, section }) {
  const Tag = section.level === 3 ? "h3" : "h2";
  return (
    <Tag id={section.id} className="doc-heading">
      <Link to={{ pathname: page.path, search: `?section=${section.id}` }}>{section.title}</Link>
      <HeadingLink page={page} section={section} />
    </Tag>
  );
}

function TableBlock({ columns, rows }) {
  return (
    <div className="table-wrap">
      <table className="doc-table">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={`row-${rowIndex}`}>
              {row.map((cell, cellIndex) => (
                <td
                  key={`${rowIndex}-${cellIndex}`}
                  dangerouslySetInnerHTML={{ __html: inlineMarkup(String(cell)) }}
                />
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DefinitionList({ items }) {
  return (
    <dl className="definition-list">
      {items.map(([term, description]) => (
        <div key={term} className="definition-row">
          <dt dangerouslySetInnerHTML={{ __html: inlineMarkup(term) }} />
          <dd dangerouslySetInnerHTML={{ __html: inlineMarkup(description) }} />
        </div>
      ))}
    </dl>
  );
}

function FAQBlock({ items }) {
  return (
    <div className="faq-list">
      {items.map(([question, answer]) => (
        <details key={question} className="faq-item">
          <summary>{question}</summary>
          <RichText text={answer} />
        </details>
      ))}
    </div>
  );
}

function Callout({ tone = "info", title, body }) {
  return (
    <div className={`callout callout-${tone}`}>
      <strong>{title}</strong>
      <RichText text={body} />
    </div>
  );
}

function BlockRenderer({ block }) {
  switch (block.type) {
    case "paragraph":
      return <RichText text={block.text} />;
    case "ordered":
      return (
        <ol className="doc-list doc-list-ordered">
          {block.items.map((item) => (
            <li key={item}>
              <RichText as="span" text={item} />
            </li>
          ))}
        </ol>
      );
    case "list":
      return (
        <ul className="doc-list">
          {block.items.map((item) => (
            <li key={item}>
              <RichText as="span" text={item} />
            </li>
          ))}
        </ul>
      );
    case "code":
      return <CodeBlock language={block.language} code={block.code} />;
    case "table":
      return <TableBlock columns={block.columns} rows={block.rows} />;
    case "definition":
      return <DefinitionList items={block.items} />;
    case "faq":
      return <FAQBlock items={block.items} />;
    case "callout":
      return <Callout tone={block.tone} title={block.title} body={block.body} />;
    default:
      return null;
  }
}

function Hero({ page }) {
  const [copied, setCopied] = useState(false);
  const hero = page.hero;
  if (!hero) return null;

  return (
    <section className="hero">
      <div className="hero-copy">
        <span className="eyebrow">delta-framework</span>
        <h1>{hero.headline}</h1>
        <p className="hero-subtitle">{hero.subheadline}</p>
        <div className="hero-actions">
          <button
            type="button"
            className="cta-button"
            onClick={() => {
              copyText(hero.installCommand).then(() => {
                setCopied(true);
                window.setTimeout(() => setCopied(false), 1200);
              });
            }}
          >
            {copied ? "Copied pip install" : hero.installCommand}
          </button>
          <Link className="secondary-button" to="/getting-started">
            Go to docs
          </Link>
        </div>
      </div>
      <div className="hero-visual-wrap">
        <div className="hero-code">
          <div className="hero-code-header">
            <span>Quickstart</span>
            <small>8 lines</small>
          </div>
          <CodeBlock language="python" code={hero.quickstart} />
        </div>
      </div>
    </section>
  );
}

function LandingPage() {
  const [copied, setCopied] = useState(false);
  const hero = pagesByPath["/"]?.hero;
  if (!hero) return null;

  return (
    <>
      <header className="topbar">
        <div className="topbar-left">
          <Link className="brand" to="/">
            <span className="brand-mark">D</span>
            <span className="brand-copy">
              <strong>{siteMeta.title}</strong>
              <small>Home</small>
            </span>
          </Link>
        </div>
        <div className="topbar-actions">
          <a className="topbar-link" href={siteMeta.githubUrl} target="_blank" rel="noreferrer">
            GitHub
          </a>
          <a className="topbar-link" href={siteMeta.pypiUrl} target="_blank" rel="noreferrer">
            PyPI
          </a>
        </div>
      </header>

      <main className="landing-shell">
        <section className="landing-panel">
          <span className="eyebrow">delta-framework</span>
          <h1>{hero.headline}</h1>
          <p className="hero-subtitle">{hero.subheadline}</p>
          <div className="hero-actions">
            <button
              type="button"
              className="cta-button"
              onClick={() => {
                copyText(hero.installCommand).then(() => {
                  setCopied(true);
                  window.setTimeout(() => setCopied(false), 1200);
                });
              }}
            >
              {copied ? "Copied pip install" : hero.installCommand}
            </button>
            <Link className="secondary-button" to="/getting-started">
              Go to docs
            </Link>
          </div>
        </section>
      </main>
    </>
  );
}

function PageContent({ page }) {
  return (
    <article className="doc-page">
      {page.hero ? <Hero page={page} /> : null}
      {!page.hero ? (
        <header className="page-header">
          <span className="eyebrow">Documentation</span>
          <h1>{page.title}</h1>
          <p>{page.description}</p>
        </header>
      ) : null}
      {page.sections.map((section) => (
        <section key={section.id} className="doc-section">
          <Heading page={page} section={section} />
          <div className="section-stack">
            {section.blocks.map((block, index) => (
              <BlockRenderer key={`${section.id}-${index}`} block={block} />
            ))}
          </div>
        </section>
      ))}
    </article>
  );
}

function Sidebar({ activePath, openGroups, onToggleGroup, onNavigate, mobileOpen, onCloseMobile }) {
  return (
    <>
      <div className={`sidebar-scrim ${mobileOpen ? "is-open" : ""}`} onClick={onCloseMobile} aria-hidden="true" />
      <aside className={`sidebar ${mobileOpen ? "is-open" : ""}`}>
        <div className="sidebar-header">
          <span className="eyebrow">delta docs</span>
          <h2>Documentation</h2>
        </div>
        {navigation.map((group) => (
          <section key={group.title} className="sidebar-group">
            <button type="button" className="sidebar-group-toggle" onClick={() => onToggleGroup(group.title)}>
              <span>{group.title}</span>
              <span>{openGroups[group.title] ? "-" : "+"}</span>
            </button>
            {openGroups[group.title] ? (
              <nav className="sidebar-links">
                {group.items.map((item) => (
                  <NavLink
                    key={item.id}
                    to={item.path}
                    className={({ isActive }) =>
                      `sidebar-link ${isActive || activePath === item.path ? "is-active" : ""}`
                    }
                    onClick={() => {
                      onNavigate(item.path);
                      onCloseMobile();
                    }}
                  >
                    <span className="sidebar-link-pointer" />
                    <span>{item.label}</span>
                  </NavLink>
                ))}
              </nav>
            ) : null}
          </section>
        ))}
      </aside>
    </>
  );
}

function DocsFrame() {
  const location = useLocation();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [mobileSidebar, setMobileSidebar] = useState(false);
  const [openGroups, setOpenGroups] = useState(() =>
    Object.fromEntries(navigation.map((group) => [group.title, true])),
  );

  const activePage = pagesByPath[location.pathname] ?? pages[0];
  const sectionId = searchParams.get("section");

  useEffect(() => {
    document.title = activePage.title + " - " + siteMeta.title;
    Prism.highlightAll();
  }, [activePage, sectionId]);

  useEffect(() => {
    const target = sectionId ? document.getElementById(sectionId) : null;
    const frame = window.requestAnimationFrame(() => {
      if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
      else window.scrollTo({ top: 0, behavior: "auto" });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [location.pathname, sectionId]);

  useEffect(() => {
    function onKeyDown(event) {
      if (event.key === "Escape") setMobileSidebar(false);
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  function goTo(path, nextSectionId = null) {
    startTransition(() => {
      navigate({
        pathname: path,
        search: nextSectionId ? `?section=${nextSectionId}` : "",
      });
    });
  }

  return (
    <>
      <header className="topbar">
        <div className="topbar-left">
          <button type="button" className="menu-button" onClick={() => setMobileSidebar(true)}>
            Menu
          </button>
          <Link className="brand" to="/">
            <span className="brand-mark">D</span>
            <span className="brand-copy">
              <strong>{siteMeta.title}</strong>
              <small>Docs</small>
            </span>
          </Link>
        </div>
        <div className="topbar-actions">
          <a className="topbar-link" href={siteMeta.githubUrl} target="_blank" rel="noreferrer">
            GitHub
          </a>
          <a className="topbar-link" href={siteMeta.pypiUrl} target="_blank" rel="noreferrer">
            PyPI
          </a>
        </div>
      </header>

      <div className="docs-shell">
        <Sidebar
          activePath={activePage.path}
          openGroups={openGroups}
          onToggleGroup={(title) => setOpenGroups((current) => ({ ...current, [title]: !current[title] }))}
          onNavigate={goTo}
          mobileOpen={mobileSidebar}
          onCloseMobile={() => setMobileSidebar(false)}
        />
        <main className="content-column">
          <PageContent page={activePage} />
          <footer className="footer">
            <div className="footer-brand">
              <strong>{siteMeta.title}</strong>
            </div>
            <div className="footer-links">
              <a href={siteMeta.githubUrl} target="_blank" rel="noreferrer">
                GitHub
              </a>
              <a href={siteMeta.pypiUrl} target="_blank" rel="noreferrer">
                PyPI
              </a>
              <span>{siteMeta.license}</span>
              <span className="footer-badge">v{siteMeta.version}</span>
            </div>
          </footer>
        </main>
      </div>
    </>
  );
}

export default function App() {
  return (
    <HashRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        {pages
          .filter((page) => page.path !== "/")
          .map((page) => (
            <Route key={page.id} path={page.path} element={<DocsFrame />} />
          ))}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </HashRouter>
  );
}
