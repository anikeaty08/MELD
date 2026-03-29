import { examples } from "../siteData.js";
import { ChapterSection, CodeBlock, StatCard } from "../ui.jsx";

export default function ExamplesChapter() {
  return (
    <>
      <ChapterSection
        eyebrow="Chapter 5"
        title="Ready-to-run examples"
        body="Use these as starting points for real runs instead of piecing commands together by hand."
      >
        <CodeBlock label="CIFAR-10 compare mode" code={examples.cifar10} />
      </ChapterSection>

      <ChapterSection eyebrow="Bootstrap assets" title="Download benchmark assets before the first real run">
        <CodeBlock
          label="Bootstrap datasets"
          code={"python -m meld.bootstrap --datasets CIFAR-10 CIFAR-100 CIFAR-10-C --data-root ./data"}
        />
      </ChapterSection>

      <ChapterSection eyebrow="When to use which entry point" title="Choose the workflow that matches your job">
        <div className="bullet-grid">
          <StatCard title="meld.cli" body="Best when you want reproducible benchmark runs from the command line." />
          <StatCard title="run(MELDConfig(...))" body="Best when you are scripting experiments in Python." />
          <StatCard title="register_dataset(...)" body="Best when your data is manual and you still want the runner and compare-mode pipeline." />
          <StatCard title="DeltaModel" body="Best when you already manage the DataLoaders yourself and want direct update and predict calls." />
        </div>
      </ChapterSection>
    </>
  );
}
