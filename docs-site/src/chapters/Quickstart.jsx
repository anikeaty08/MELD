import { examples } from "../siteData.js";
import { ChapterSection, CodeBlock, StatCard } from "../ui.jsx";

export default function QuickstartChapter() {
  return (
    <>
      <ChapterSection
        eyebrow="Chapter 2"
        title="Run the first MELD jobs"
        body="These commands are grounded in the actual CLI and package APIs already present in the repo."
      >
        <div className="info-grid">
          <StatCard title="Compare mode" body="Run delta updates and a full-retrain baseline side by side." />
          <StatCard title="Delta mode" body="Run only the incremental path when you care about deployability and speed." />
          <StatCard title="Text mode" body="Switch the backbone to text_encoder to run AGNews or similar datasets." />
          <StatCard title="Result files" body="MELD writes JSON output and can also persist run history to SQLite." />
        </div>
      </ChapterSection>

      <ChapterSection eyebrow="Smoke test" title="Start with a tiny synthetic run">
        <CodeBlock label="CLI smoke test" code={examples.syntheticCompare} />
      </ChapterSection>

      <ChapterSection eyebrow="Text example" title="Run a HuggingFace-backed text benchmark">
        <CodeBlock label="AGNews example" code={examples.text} />
      </ChapterSection>
    </>
  );
}
