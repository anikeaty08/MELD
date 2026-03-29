import { chapters } from "../content.js";
import { ChapterSection, StatCard } from "../ui.jsx";

export default function OverviewChapter() {
  return (
    <>
      <section className="hero-card reveal">
        <p className="eyebrow">Overview</p>
        <h1>Learn MELD from installation to deployment-safe continual updates.</h1>
        <p>
          This site is organized like a lesson track: install the framework, run the core
          compare workflow, teach MELD your dataset, understand the architecture, and then
          interpret the benchmark outputs with the right research context.
        </p>
        <div className="hero-actions">
          <a className="primary-link" href="#installation">
            Start installation
          </a>
          <a className="secondary-link" href="#manual-data">
            Teach MELD a manual dataset
          </a>
        </div>
      </section>

      <ChapterSection
        eyebrow="How to use this site"
        title="Follow the same order someone would use MELD in real life"
        body="The docs are not arranged like a marketing page. They are chapter-based, with one left navigation rail and a clear path from setup to custom data."
      >
        <div className="chapter-grid">
          {chapters.slice(1).map((chapter) => (
            <a key={chapter.id} className="chapter-card" href={`#${chapter.id}`}>
              <span className="chapter-number">{chapter.order.replace("Chapter ", "")}</span>
              <strong>{chapter.title}</strong>
              <p>{chapter.blurb}</p>
            </a>
          ))}
        </div>
      </ChapterSection>

      <ChapterSection
        eyebrow="What MELD ships"
        title="Core product surfaces"
        body="The docs focus on the real entry points already present in the repo."
      >
        <div className="info-grid">
          <StatCard title="CLI runner" body="Use python -m meld.cli for compare, delta-only, or full-retrain workflows." />
          <StatCard title="Python API" body="Use MELDConfig, TrainConfig, run, register_dataset, and DeltaModel from Python." />
          <StatCard title="Dataset hooks" body="Register custom providers and split your own train/test datasets into incremental tasks." />
          <StatCard title="Benchmark reporting" body="Read equivalence gap, compute savings, forgetting, risk estimate, and drift together." />
        </div>
      </ChapterSection>
    </>
  );
}
