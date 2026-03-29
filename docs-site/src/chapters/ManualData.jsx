import { examples } from "../siteData.js";
import { ChapterSection, CodeBlock, StatCard } from "../ui.jsx";

export default function ManualDataChapter() {
  return (
    <>
      <ChapterSection
        eyebrow="Chapter 3"
        title="Teach MELD your own dataset"
        body="If your data is not one of the built-ins, the Python API is the right place to teach MELD how to split, label, and feed it."
      >
        <div className="info-grid">
          <StatCard title="Dataset provider path" body="Register a provider that returns a list of (train_dataset, test_dataset) task pairs." />
          <StatCard title="Manual DataLoader path" body="Use DeltaModel directly when you already have task-specific DataLoaders and want fine-grained control." />
          <StatCard title="Global labels" body="Keep class IDs consistent across tasks. MELD expands the classifier based on the observed label IDs." />
          <StatCard title="Best place to extend" body="Use register_dataset plus split_classification_dataset_into_tasks when you want MELD runner support with your own data." />
        </div>
      </ChapterSection>

      <ChapterSection
        eyebrow="Provider contract"
        title="Register a custom dataset provider"
        body="This is the cleanest way to plug manual image datasets into compare-mode benchmarking."
      >
        <CodeBlock label="Custom provider example" code={examples.provider} />
      </ChapterSection>

      <ChapterSection
        eyebrow="Direct training"
        title="Use DeltaModel when you want to drive the tasks yourself"
        body="This is useful when your task schedule is already prepared externally and you want MELD to focus only on the update logic."
      >
        <CodeBlock label="DeltaModel manual update" code={examples.deltaModel} />
      </ChapterSection>

      <ChapterSection
        eyebrow="Manual dataset checklist"
        title="What MELD expects from your custom data"
        body="This is the shortest path to getting a manual dataset working without fighting the framework."
      >
        <div className="checklist">
          <div>
            <strong>1. Dataset objects</strong>
            <p>Your train and eval datasets must implement __len__ and __getitem__ and return (input, target).</p>
          </div>
          <div>
            <strong>2. Stable class IDs</strong>
            <p>Targets should be integer class IDs that stay stable across the whole incremental sequence.</p>
          </div>
          <div>
            <strong>3. Task slicing</strong>
            <p>Return a list of task pairs or use split_classification_dataset_into_tasks to build it automatically.</p>
          </div>
          <div>
            <strong>4. Correct backbone</strong>
            <p>Use a vision backbone for images and text_encoder for NLP style datasets.</p>
          </div>
        </div>
      </ChapterSection>
    </>
  );
}
