import { ChapterSection, StatCard } from "../ui.jsx";

export default function MetricsChapter() {
  return (
    <>
      <ChapterSection
        eyebrow="Chapter 6"
        title="Benchmark reporting"
        body="These are the metrics that matter in MELD, but they should be read together rather than overfitting the story to a single number."
      >
        <div className="info-grid">
          <StatCard title="Equivalence gap" body="A closeness check between the delta and full-retrain confusion matrices. Useful, but not the only signal that matters." />
          <StatCard title="Compute savings" body="The time delta between full retraining and the delta path, reported as a percentage." />
          <StatCard title="Forgetting" body="Old-task accuracy drop after later tasks are learned." />
          <StatCard title="ECE" body="Expected calibration error, which checks confidence quality rather than only raw accuracy." />
          <StatCard title="Risk estimate" body="The pre-training estimate logged before the update starts." />
          <StatCard title="Drift realized" body="The post-update drift signal measured after the update completes." />
        </div>
      </ChapterSection>

      <ChapterSection
        eyebrow="How to read it"
        title="Do not treat equivalence gap as the whole story"
        body="A run can have a decent equivalence gap and still be bad if it forgets old classes, saves little compute, or becomes poorly calibrated. MELD is strongest when the metrics agree."
      >
        <div className="checklist">
          <div>
            <strong>Compare equivalence gap with forgetting</strong>
            <p>If the gap looks fine but forgetting rises, the delta path still is not tracking full retraining well.</p>
          </div>
          <div>
            <strong>Check savings with accuracy</strong>
            <p>Fast updates only matter if the delta path stays close enough to the retrained reference.</p>
          </div>
          <div>
            <strong>Watch calibration</strong>
            <p>ECE helps catch cases where top-1 looks stable but confidence quality has collapsed.</p>
          </div>
          <div>
            <strong>Use risk plus drift</strong>
            <p>Pre-update risk and post-update drift tell you whether the deployment decision makes sense.</p>
          </div>
        </div>
      </ChapterSection>
    </>
  );
}
