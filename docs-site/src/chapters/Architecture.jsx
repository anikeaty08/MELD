import { ChapterSection, StatCard } from "../ui.jsx";

export default function ArchitectureChapter() {
  return (
    <>
      <ChapterSection
        eyebrow="Chapter 4"
        title="How MELD moves through one incremental task"
        body="The framework works as a chain: snapshot the prior state, estimate risk, update on new data, audit drift, then decide whether the delta path is safe enough."
      >
        <div className="flow-board">
          <div className="flow-node">
            <span>01</span>
            <strong>Task arrives</strong>
            <p>New task data is loaded through the runner or your DataLoader.</p>
          </div>
          <div className="flow-node">
            <span>02</span>
            <strong>Snapshot prior state</strong>
            <p>FisherManifoldSnapshot captures class statistics, anchors, and Fisher information before the update.</p>
          </div>
          <div className="flow-node">
            <span>03</span>
            <strong>Risk gate</strong>
            <p>SpectralSafetyOracle reports the pre-training risk estimate and PAC-style metadata used for gating.</p>
          </div>
          <div className="flow-node">
            <span>04</span>
            <strong>Delta update</strong>
            <p>GeometryConstrainedUpdater trains only on new data with geometry, EWC, and importance weighting.</p>
          </div>
          <div className="flow-node">
            <span>05</span>
            <strong>Audit the result</strong>
            <p>MELD re-snapshots the model, measures realized drift, and compares delta against full retraining when compare mode is on.</p>
          </div>
          <div className="flow-node">
            <span>06</span>
            <strong>Deployment decision</strong>
            <p>FourStateDeployPolicy chooses SAFE_DELTA, CAUTIOUS_DELTA, BOUND_VIOLATED, SHIFT_CRITICAL, or BOUND_EXCEEDED.</p>
          </div>
        </div>
      </ChapterSection>

      <ChapterSection eyebrow="Core modules" title="What each main piece is responsible for">
        <div className="info-grid">
          <StatCard title="GeometryConstrainedUpdater" body="Learns from new task data while constraining old knowledge with EWC, geometry, and related penalties." />
          <StatCard title="FisherManifoldSnapshot" body="Stores the compact memory of the current state without keeping the whole historical dataset." />
          <StatCard title="SpectralSafetyOracle" body="Produces pre-update risk estimates and PAC-style reporting terms for deployment decisions." />
          <StatCard title="BenchmarkRunner" body="Orchestrates task loading, compare mode, metrics, result writing, and baseline evaluation." />
        </div>
      </ChapterSection>
    </>
  );
}
