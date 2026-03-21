import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

function toneForStatus(status) {
  const value = String(status || "").toLowerCase();
  if (value.includes("ready") || value === "built-in" || value === "code-ready") {
    return "good";
  }
  if (value.includes("manual") || value.includes("missing") || value.includes("pending")) {
    return "warn";
  }
  if (value.includes("failed") || value.includes("error")) {
    return "bad";
  }
  return "neutral";
}

function describeDetail(detail) {
  if (detail == null) {
    return "";
  }
  if (typeof detail === "string") {
    return detail;
  }
  try {
    return JSON.stringify(detail, null, 2);
  } catch (_error) {
    return String(detail);
  }
}

function buildPolyline(values, width, height) {
  if (!Array.isArray(values) || values.length === 0) {
    return "";
  }
  const points = [];
  const max = Math.max(...values.map((value) => Number(value || 0)), 1e-6);
  values.forEach((value, index) => {
    const x = values.length === 1 ? width / 2 : (index / (values.length - 1)) * width;
    const y = height - (Number(value || 0) / max) * height;
    points.push(`${x},${y}`);
  });
  return points.join(" ");
}

function extractEpochAccuracy(results) {
  const history = results?.epoch_history;
  if (!Array.isArray(history)) {
    return [];
  }
  const values = [];
  history.forEach((entry) => {
    const perEpoch = entry?.delta?.train_accuracy_per_epoch;
    if (Array.isArray(perEpoch)) {
      perEpoch.forEach((value) => values.push(Number(value || 0)));
    }
  });
  return values;
}

function extractRiskValues(results) {
  const timeline = results?.bounds_timeline;
  if (!Array.isArray(timeline)) {
    return [];
  }
  return timeline.map((point) => Number(point?.risk_estimate_pre || 0));
}

function statusLookup(items, id) {
  if (!Array.isArray(items)) {
    return null;
  }
  return items.find((item) => item.id === id) || null;
}

function buildRequestBody(form, options, prepareSettings, prepareEnabled) {
  const allTextModels = Array.isArray(options?.textModels)
    ? options.textModels.map((item) => item.id)
    : [];
  const textModels = prepareSettings.warmAllTextModels
    ? allTextModels
    : prepareSettings.warmSelectedModel
      ? [form.textEncoderModel]
      : [];
  return {
    dataset: form.dataset,
    num_tasks: Number(form.numTasks),
    classes_per_task: Number(form.classesPerTask),
    epochs: Number(form.epochs),
    batch_size: Number(form.batchSize),
    lr: Number(form.lr),
    backbone: form.backbone,
    pretrained_backbone: Boolean(form.pretrainedBackbone),
    text_encoder_model: form.textEncoderModel,
    bound_tolerance: Number(form.boundTolerance),
    pac_gate_tolerance: Number(form.pacGateTolerance),
    mixup_alpha: Number(form.mixupAlpha),
    num_workers: Number(form.numWorkers),
    data_root: form.dataRoot,
    database_path: form.databasePath,
    results_path: form.resultsPath,
    prefer_cuda: Boolean(form.preferCuda),
    prepare: {
      enabled: Boolean(prepareEnabled),
      install_requirements: Boolean(prepareSettings.installRequirements),
      all_datasets: Boolean(prepareSettings.allDatasets),
      datasets: prepareSettings.allDatasets ? [] : [form.dataset],
      text_models: textModels,
    },
  };
}

function StatusBadge({ status }) {
  return <span className={`status-badge tone-${toneForStatus(status)}`}>{status || "unknown"}</span>;
}

function LineChart({ values, color, empty, label }) {
  if (!Array.isArray(values) || values.length === 0) {
    return <div className="chart-empty">{empty}</div>;
  }
  return (
    <div className="chart-frame">
      <div className="chart-label">{label}</div>
      <svg viewBox="0 0 320 140" className="chart-svg" aria-hidden="true">
        <defs>
          <linearGradient id={`fill-${color.replace("#", "")}`} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.28" />
            <stop offset="100%" stopColor={color} stopOpacity="0.02" />
          </linearGradient>
        </defs>
        <path d="M0 139.5 H320" className="chart-axis" />
        <polyline
          points={`${buildPolyline(values, 320, 120)} 320,140 0,140`}
          fill={`url(#fill-${color.replace("#", "")})`}
          stroke="none"
        />
        <polyline points={buildPolyline(values, 320, 120)} fill="none" stroke={color} strokeWidth="4" />
      </svg>
    </div>
  );
}

function App() {
  const [options, setOptions] = useState(null);
  const [form, setForm] = useState(null);
  const [readiness, setReadiness] = useState(null);
  const [runtime, setRuntime] = useState({
    running: false,
    preparing: false,
    error: null,
    prepareReport: null,
    results: null,
    logs: "",
  });
  const [prepareSettings, setPrepareSettings] = useState({
    installRequirements: false,
    allDatasets: false,
    warmSelectedModel: true,
    warmAllTextModels: false,
    prepareBeforeRun: false,
  });
  const [message, setMessage] = useState("");
  const [busyAction, setBusyAction] = useState("");
  const [inspector, setInspector] = useState(null);

  async function fetchJson(path, init) {
    const response = await fetch(path, init);
    const data = await response.json();
    if (!response.ok) {
      const detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data);
      throw new Error(detail);
    }
    return data;
  }

  async function refreshState() {
    const data = await fetchJson("/api/state");
    setRuntime(data);
  }

  async function refreshReadiness(targetRoot) {
    const query = targetRoot ? `?data_root=${encodeURIComponent(targetRoot)}` : "";
    const data = await fetchJson(`/api/readiness${query}`);
    setReadiness(data);
  }

  async function bootstrap() {
    const payload = await fetchJson("/api/options");
    setOptions(payload);
    const datasetChoice = payload.datasets[0];
    const defaults = payload.defaults;
    setForm({
      dataset: defaults.dataset || datasetChoice?.id || "CIFAR-10",
      numTasks: String(defaults.numTasks ?? 2),
      classesPerTask: String(defaults.classesPerTask ?? 5),
      epochs: String(defaults.epochs ?? 5),
      batchSize: String(defaults.batchSize ?? 64),
      lr: String(defaults.lr ?? 0.1),
      backbone: defaults.backbone || "auto",
      pretrainedBackbone: Boolean(defaults.pretrainedBackbone),
      textEncoderModel: defaults.textEncoderModel || payload.textModels[0]?.id || "",
      boundTolerance: String(defaults.boundTolerance ?? 10.0),
      pacGateTolerance: String(defaults.pacGateTolerance ?? 0.1),
      mixupAlpha: String(defaults.mixupAlpha ?? 0.2),
      numWorkers: String(defaults.numWorkers ?? 0),
      dataRoot: defaults.dataRoot || "./data",
      databasePath: defaults.databasePath || "meld_results.db",
      resultsPath: defaults.resultsPath || "results.json",
      preferCuda: Boolean(defaults.preferCuda),
    });
    await Promise.all([refreshReadiness(defaults.dataRoot || "./data"), refreshState()]);
  }

  useEffect(() => {
    bootstrap().catch((error) => setMessage(error.message));
  }, []);

  useEffect(() => {
    const source = new EventSource("/api/state/stream");
    source.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        setRuntime((current) => ({
          ...current,
          ...payload,
        }));
      } catch (_error) {
        // Ignore malformed stream payloads.
      }
    };
    source.onerror = () => {
      source.close();
    };
    return () => source.close();
  }, []);

  useEffect(() => {
    if (window.location.pathname === "/monitor") {
      window.setTimeout(() => {
        document.getElementById("monitor-panel")?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 140);
    }
  }, []);

  if (!options || !form) {
    return (
      <main className="app-shell loading-shell">
        <section className="hero-card">
          <p className="eyebrow">MELD Dashboard</p>
          <h1>Loading workspace controls...</h1>
        </section>
      </main>
    );
  }

  const selectedDataset = options.datasets.find((item) => item.id === form.dataset) || options.datasets[0];
  const selectedBackbone = options.backbones.find((item) => item.id === form.backbone) || options.backbones[0];
  const riskValues = extractRiskValues(runtime.results);
  const accuracyValues = extractEpochAccuracy(runtime.results);

  function updateForm(name, value) {
    setForm((current) => ({ ...current, [name]: value }));
  }

  function handleDatasetSelect(dataset) {
    setForm((current) => ({
      ...current,
      dataset: dataset.id,
      numTasks: String(dataset.defaultTasks),
      classesPerTask: String(dataset.defaultClassesPerTask),
      backbone: dataset.recommendedBackbone,
    }));
    setMessage(`Loaded ${dataset.label} defaults.`);
  }

  async function handlePrepare() {
    setBusyAction("prepare");
    setMessage("");
    try {
      const body = buildRequestBody(form, options, prepareSettings, true);
      const payload = await fetchJson("/api/prepare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      setRuntime((current) => ({ ...current, prepareReport: payload.report }));
      setReadiness(payload.readiness);
      setMessage(payload.report.success ? "Workspace preparation finished." : "Workspace preparation completed with warnings.");
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusyAction("");
    }
  }

  async function handleLaunch(event) {
    event.preventDefault();
    setBusyAction("launch");
    setMessage("");
    try {
      const body = buildRequestBody(form, options, prepareSettings, prepareSettings.prepareBeforeRun);
      const payload = await fetchJson("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      setMessage(`Run launched${payload.pid ? ` (PID ${payload.pid})` : ""}.`);
      await refreshState();
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusyAction("");
    }
  }

  async function handleStop() {
    setBusyAction("stop");
    setMessage("");
    try {
      const payload = await fetchJson("/api/stop", { method: "POST" });
      setMessage(payload.stopped ? "Experiment stopped." : payload.message);
      await refreshState();
    } catch (error) {
      setMessage(error.message);
    } finally {
      setBusyAction("");
    }
  }

  return (
    <main className="app-shell">
      <section className="hero-card">
        <div>
          <p className="eyebrow">React control room</p>
          <h1>MELD benchmark launcher with early asset prep.</h1>
          <p className="hero-copy">
            Pick a dataset, see every backbone and text encoder choice, then warm requirements, datasets, and model
            assets before the experiment starts.
          </p>
        </div>
        <div className="hero-actions">
          <button type="button" className="button button-secondary" onClick={() => refreshReadiness(form.dataRoot)}>
            Refresh readiness
          </button>
          <a className="button button-ghost" href="/results/csv">Export CSV</a>
          <a className="button button-ghost" href="/results">Results JSON</a>
        </div>
      </section>

      {message ? (
        <section className="message-strip">
          <StatusBadge status={runtime.error ? "error" : "info"} />
          <span>{message}</span>
        </section>
      ) : null}

      <section className="dashboard-grid">
        <div className="stack-column">
          <section className="panel">
            <div className="panel-head">
              <div>
                <p className="eyebrow">Dataset library</p>
                <h2>Every supported benchmark</h2>
              </div>
              <p className="panel-note">Click a card to load its recommended task layout and backbone.</p>
            </div>
            <div className="tile-grid dataset-grid">
              {options.datasets.map((dataset) => {
                const status = statusLookup(readiness?.datasets, dataset.id);
                const selected = dataset.id === form.dataset;
                return (
                  <button
                    type="button"
                    key={dataset.id}
                    className={`choice-tile ${selected ? "selected" : ""}`}
                    onClick={() => handleDatasetSelect(dataset)}
                  >
                    <div className="tile-topline">
                      <strong>{dataset.label}</strong>
                      <StatusBadge status={status?.status || (dataset.autoDownload ? "not checked" : "manual")} />
                    </div>
                    <div className="tile-meta">
                      <span>{dataset.domain}</span>
                      <span>{dataset.classCount ? `${dataset.classCount} classes` : "generated"}</span>
                    </div>
                    <p>{dataset.note}</p>
                  </button>
                );
              })}
            </div>
          </section>

          <section className="panel">
            <div className="panel-head">
              <div>
                <p className="eyebrow">Launch setup</p>
                <h2>Run configuration</h2>
              </div>
              <div className="status-row">
                <StatusBadge status={runtime.running ? "running" : runtime.preparing ? "preparing" : "idle"} />
              </div>
            </div>
            <form className="config-form" onSubmit={handleLaunch}>
              <div className="field-grid">
                <label>
                  <span>Dataset</span>
                  <select value={form.dataset} onChange={(event) => handleDatasetSelect(options.datasets.find((item) => item.id === event.target.value))}>
                    {options.datasets.map((dataset) => (
                      <option key={dataset.id} value={dataset.id}>{dataset.label}</option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Data root</span>
                  <input value={form.dataRoot} onChange={(event) => updateForm("dataRoot", event.target.value)} />
                </label>
                <label>
                  <span>Tasks</span>
                  <input type="number" min="1" value={form.numTasks} onChange={(event) => updateForm("numTasks", event.target.value)} />
                </label>
                <label>
                  <span>Classes per task</span>
                  <input type="number" min="1" value={form.classesPerTask} onChange={(event) => updateForm("classesPerTask", event.target.value)} />
                </label>
                <label>
                  <span>Epochs</span>
                  <input type="number" min="1" value={form.epochs} onChange={(event) => updateForm("epochs", event.target.value)} />
                </label>
                <label>
                  <span>Batch size</span>
                  <input type="number" min="1" value={form.batchSize} onChange={(event) => updateForm("batchSize", event.target.value)} />
                </label>
                <label>
                  <span>Learning rate</span>
                  <input type="number" step="0.001" min="0" value={form.lr} onChange={(event) => updateForm("lr", event.target.value)} />
                </label>
                <label>
                  <span>Workers</span>
                  <input type="number" min="0" value={form.numWorkers} onChange={(event) => updateForm("numWorkers", event.target.value)} />
                </label>
                <label>
                  <span>Bound tolerance</span>
                  <input type="number" step="0.1" min="0" value={form.boundTolerance} onChange={(event) => updateForm("boundTolerance", event.target.value)} />
                </label>
                <label>
                  <span>PAC gate tolerance</span>
                  <input type="number" step="0.1" min="0" value={form.pacGateTolerance} onChange={(event) => updateForm("pacGateTolerance", event.target.value)} />
                </label>
                <label>
                  <span>Mixup alpha</span>
                  <input type="number" step="0.1" min="0" value={form.mixupAlpha} onChange={(event) => updateForm("mixupAlpha", event.target.value)} />
                </label>
                <label className="checkbox-field">
                  <input type="checkbox" checked={form.pretrainedBackbone} onChange={(event) => updateForm("pretrainedBackbone", event.target.checked)} />
                  <span>Use pretrained image weights</span>
                </label>
                <label className="checkbox-field">
                  <input type="checkbox" checked={form.preferCuda} onChange={(event) => updateForm("preferCuda", event.target.checked)} />
                  <span>Prefer CUDA</span>
                </label>
              </div>

              <div className="choice-strip">
                <div className="strip-head">
                  <div>
                    <p className="eyebrow">Backbones</p>
                    <h3>Every model choice is visible here</h3>
                  </div>
                  <div className="inline-note">{selectedBackbone?.note}</div>
                </div>
                <div className="tile-grid">
                  {options.backbones.map((backbone) => {
                    const status = statusLookup(readiness?.backbones, backbone.id);
                    return (
                      <button
                        key={backbone.id}
                        type="button"
                        className={`choice-tile compact ${form.backbone === backbone.id ? "selected" : ""}`}
                        onClick={() => updateForm("backbone", backbone.id)}
                      >
                        <div className="tile-topline">
                          <strong>{backbone.label}</strong>
                          <StatusBadge status={status?.status || "ready"} />
                        </div>
                        <div className="tile-meta">
                          <span>{backbone.family}</span>
                          <span>{backbone.supportsPretrained ? "pretrained optional" : "runtime only"}</span>
                        </div>
                        <p>{backbone.note}</p>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="choice-strip">
                <div className="strip-head">
                  <div>
                    <p className="eyebrow">Text encoders</p>
                    <h3>Warm them early if you plan to benchmark NLP</h3>
                  </div>
                  <div className="inline-note">Selected model: {form.textEncoderModel}</div>
                </div>
                <div className="tile-grid">
                  {options.textModels.map((model) => {
                    const status = statusLookup(readiness?.textModels, model.id);
                    return (
                      <button
                        key={model.id}
                        type="button"
                        className={`choice-tile compact ${form.textEncoderModel === model.id ? "selected" : ""}`}
                        onClick={() => updateForm("textEncoderModel", model.id)}
                      >
                        <div className="tile-topline">
                          <strong>{model.label}</strong>
                          <StatusBadge status={status?.status || "not checked"} />
                        </div>
                        <div className="tile-meta">
                          <span>{model.outDim} dims</span>
                          <span>{model.id}</span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div className="cta-row">
                <button className="button" type="submit" disabled={busyAction !== ""}>
                  {busyAction === "launch" ? "Launching..." : "Launch experiment"}
                </button>
                <button className="button button-secondary" type="button" onClick={handleStop} disabled={busyAction !== ""}>
                  {busyAction === "stop" ? "Stopping..." : "Stop run"}
                </button>
              </div>
            </form>
          </section>
        </div>

        <div className="stack-column">
          <section className="panel">
            <div className="panel-head">
              <div>
                <p className="eyebrow">Prepare first</p>
                <h2>Install and cache what the run needs</h2>
              </div>
              <StatusBadge status={runtime.preparing ? "preparing" : "ready"} />
            </div>
            <div className="toggle-stack">
              <label className="checkbox-field">
                <input
                  type="checkbox"
                  checked={prepareSettings.installRequirements}
                  onChange={(event) => setPrepareSettings((current) => ({ ...current, installRequirements: event.target.checked }))}
                />
                <span>Install everything from requirements.txt before prepping assets</span>
              </label>
              <label className="checkbox-field">
                <input
                  type="checkbox"
                  checked={prepareSettings.allDatasets}
                  onChange={(event) => setPrepareSettings((current) => ({ ...current, allDatasets: event.target.checked }))}
                />
                <span>Download every auto-supported dataset, not just the selected one</span>
              </label>
              <label className="checkbox-field">
                <input
                  type="checkbox"
                  checked={prepareSettings.warmSelectedModel}
                  onChange={(event) => setPrepareSettings((current) => ({ ...current, warmSelectedModel: event.target.checked }))}
                />
                <span>Warm the selected model assets before launch</span>
              </label>
              <label className="checkbox-field">
                <input
                  type="checkbox"
                  checked={prepareSettings.warmAllTextModels}
                  onChange={(event) => setPrepareSettings((current) => ({ ...current, warmAllTextModels: event.target.checked }))}
                />
                <span>Warm every text encoder option in the dashboard</span>
              </label>
              <label className="checkbox-field">
                <input
                  type="checkbox"
                  checked={prepareSettings.prepareBeforeRun}
                  onChange={(event) => setPrepareSettings((current) => ({ ...current, prepareBeforeRun: event.target.checked }))}
                />
                <span>Run the preparation step automatically before launching</span>
              </label>
            </div>
            <div className="cta-row">
              <button className="button" type="button" onClick={handlePrepare} disabled={busyAction !== ""}>
                {busyAction === "prepare" ? "Preparing..." : "Prepare workspace now"}
              </button>
            </div>
            {runtime.prepareReport ? (
              <div className="report-card">
                <div className="tile-topline">
                  <strong>Latest prepare report</strong>
                  <StatusBadge status={runtime.prepareReport.success ? "ready" : "warning"} />
                </div>
                <pre>{describeDetail(runtime.prepareReport)}</pre>
              </div>
            ) : (
              <p className="panel-note">No prepare report yet. Use this panel to install requirements and cache datasets early.</p>
            )}
          </section>

          <section className="panel">
            <div className="panel-head">
              <div>
                <p className="eyebrow">Readiness snapshot</p>
                <h2>What is already local?</h2>
              </div>
              <StatusBadge status={readiness?.requirements?.ready ? "ready" : "pending"} />
            </div>
            <div className="stats-row">
              <div className="stat-card">
                <span>Requirements</span>
                <strong>{readiness?.requirements?.items?.filter((item) => item.ready).length || 0}/{readiness?.requirements?.items?.length || 0}</strong>
              </div>
              <div className="stat-card">
                <span>Datasets ready</span>
                <strong>{readiness?.datasets?.filter((item) => item.ready).length || 0}/{readiness?.datasets?.length || 0}</strong>
              </div>
              <div className="stat-card">
                <span>Text models cached</span>
                <strong>{readiness?.textModels?.filter((item) => item.ready).length || 0}/{readiness?.textModels?.length || 0}</strong>
              </div>
            </div>
            <div className="mini-grid">
              {(readiness?.requirements?.items || []).map((item) => (
                <button
                  key={item.name}
                  type="button"
                  className="mini-card"
                  onClick={() => setInspector({ title: item.name, body: describeDetail(item) })}
                >
                  <div className="tile-topline">
                    <strong>{item.name}</strong>
                    <StatusBadge status={item.ready ? "ready" : "missing"} />
                  </div>
                  <p>{item.importName}</p>
                </button>
              ))}
            </div>
          </section>

          <section className="panel" id="monitor-panel">
            <div className="panel-head">
              <div>
                <p className="eyebrow">Live monitor</p>
                <h2>Training signals and logs</h2>
              </div>
              <div className="status-row">
                <StatusBadge status={runtime.running ? "running" : runtime.preparing ? "preparing" : "idle"} />
              </div>
            </div>
            <div className="chart-grid">
              <LineChart values={riskValues} color="#ed6a3a" label="Risk estimate" empty="No bound timeline yet." />
              <LineChart values={accuracyValues} color="#1f9d8b" label="Training accuracy" empty="No epoch history yet." />
            </div>
            <div className="report-card">
              <div className="tile-topline">
                <strong>Console log</strong>
                <StatusBadge status={runtime.error ? "error" : "streaming"} />
              </div>
              <pre>{runtime.logs || "Waiting for logs..."}</pre>
            </div>
            <div className="report-card">
              <div className="tile-topline">
                <strong>Latest results</strong>
                <StatusBadge status={runtime.results ? "ready" : "pending"} />
              </div>
              <pre>{describeDetail(runtime.results)}</pre>
            </div>
          </section>
        </div>
      </section>

      <section className="panel inspector-panel">
        <div className="panel-head">
          <div>
            <p className="eyebrow">Focused detail</p>
            <h2>{inspector?.title || selectedDataset.label}</h2>
          </div>
          <button type="button" className="button button-ghost" onClick={() => setInspector(null)}>Clear</button>
        </div>
        <pre>{inspector ? inspector.body : describeDetail({ dataset: selectedDataset, backbone: selectedBackbone })}</pre>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
