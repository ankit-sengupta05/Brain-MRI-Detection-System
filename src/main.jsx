import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Activity,
  AlertCircle,
  Brain,
  CheckCircle2,
  Download,
  FileImage,
  GitCompareArrows,
  LayoutDashboard,
  LoaderCircle,
  ScanLine,
  ShieldAlert,
  Upload,
} from "lucide-react";
import "./styles.css";

const MODEL_COLORS = {
  "Custom CNN": "#a78bfa",
  ResNet18: "#22d3ee",
  MobileNetV3: "#fb7185",
};
const CLASS_LABELS = {
  glioma: "Glioma",
  meningioma: "Meningioma",
  notumor: "No Tumor",
  pituitary: "Pituitary",
};
const NAV_ITEMS = [
  ["Overview", LayoutDashboard],
  ["Model Comparison", GitCompareArrows],
  ["Training & Validation", Activity],
  ["Predictions", ScanLine],
  ["Confusion Matrix", Brain],
];
const tooltipStyle = {
  contentStyle: {
    background: "#101528",
    border: "1px solid #2b3855",
    borderRadius: 12,
    color: "#eef2ff",
  },
};

function percentage(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function downloadJson(data, fileName) {
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json",
  });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = fileName;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

function Panel({ eyebrow, title, children }) {
  return (
    <section className="panel">
      <div className="ph">
        <div>
          <label>{eyebrow}</label>
          <h3>{title}</h3>
        </div>
      </div>
      {children}
    </section>
  );
}

function Stat({ label, value, detail }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <b>{value}</b>
      {detail && <small>{detail}</small>}
    </div>
  );
}

function ModelPicker({ models, selected, onSelect }) {
  return (
    <div className="pills" role="group" aria-label="Select model">
      {models.map((model) => (
        <button
          className={selected === model.name ? "sel" : ""}
          key={model.name}
          onClick={() => onSelect(model.name)}
          type="button"
        >
          <i style={{ background: MODEL_COLORS[model.name] }} />
          {model.name}
        </button>
      ))}
    </div>
  );
}

function PageHeading({ eyebrow, title, models, selected, onSelect }) {
  return (
    <div className="head">
      <div>
        <label>{eyebrow}</label>
        <h2>{title}</h2>
      </div>
      {models.length > 0 && (
        <ModelPicker models={models} selected={selected} onSelect={onSelect} />
      )}
    </div>
  );
}

function ClassReport({ model }) {
  return (
    <div className="class-report">
      <div className="report-row report-header">
        <span>CLASS</span><span>PRECISION</span><span>RECALL</span>
        <span>F1</span><span>SUPPORT</span>
      </div>
      {Object.entries(model.test_report.classes).map(([name, metrics]) => (
        <div className="report-row" key={name}>
          <b>{CLASS_LABELS[name] || name}</b>
          <span>{percentage(metrics.precision)}</span>
          <span>{percentage(metrics.recall)}</span>
          <span>{percentage(metrics.f1)}</span>
          <span>{metrics.support}</span>
        </div>
      ))}
      <div className="report-summary">
        Macro average F1: <b>{percentage(model.test_report.macro_f1)}</b>
        <span>{model.test_report.support} held-out test images</span>
      </div>
    </div>
  );
}

function Overview({ models, selected, onSelect, report }) {
  const bestModel = [...models].sort(
    (first, second) =>
      second.test_report.accuracy - first.test_report.accuracy,
  )[0];
  return (
    <>
      <PageHeading
        eyebrow="EXECUTIVE SUMMARY"
        title="Performance at a glance"
        models={models}
        selected={selected}
        onSelect={onSelect}
      />
      <div className="stats">
        <Stat
          label="BEST TEST ACCURACY"
          value={percentage(bestModel.test_report.accuracy)}
          detail={`${bestModel.name} · held-out test set`}
        />
        <Stat
          label="BEST MACRO F1"
          value={percentage(
            Math.max(...models.map((model) => model.test_report.macro_f1)),
          )}
          detail="Highest model score in this run"
        />
        <Stat
          label="TRAINING RUN"
          value={`${report.epochs} epochs`}
          detail={`Run ${report.run_id}`}
        />
        <Stat
          label="INFERENCE DEVICE"
          value={report.device.toUpperCase()}
          detail="Loaded trained checkpoints"
        />
      </div>
      <div className="two">
        <Panel eyebrow="HELD-OUT TEST" title="Model leaderboard">
          <div className="leaders">
            {[...models]
              .sort(
                (first, second) =>
                  second.test_report.accuracy - first.test_report.accuracy,
              )
              .map((model, index) => (
                <div className="leader" key={model.name}>
                  <em>0{index + 1}</em>
                  <div>
                    <b>
                      <i style={{ background: MODEL_COLORS[model.name] }} />
                      {model.name}
                    </b>
                    <small>{model.test_report.support} test images</small>
                  </div>
                  <strong>{percentage(model.test_report.accuracy)}</strong>
                  <strong>F1 {percentage(model.test_report.macro_f1)}</strong>
                </div>
              ))}
          </div>
        </Panel>
        <Panel eyebrow="MODEL RELIABILITY" title="Test confidence and uncertainty">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart
              data={models.map((model) => ({
                name: model.name,
                Confidence: model.history.test_mean_confidence.at(-1) * 100,
                Uncertainty: model.history.test_uncertainty.at(-1) * 100,
              }))}
              layout="vertical"
            >
              <CartesianGrid stroke="#26314a" strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 100]} />
              <YAxis type="category" dataKey="name" width={105} />
              <Tooltip {...tooltipStyle} formatter={(value) => `${Number(value).toFixed(1)}%`} />
              <Legend />
              <Bar dataKey="Confidence" fill="#8b5cf6" />
              <Bar dataKey="Uncertainty" fill="#334155" />
            </BarChart>
          </ResponsiveContainer>
        </Panel>
      </div>
      <div className="two">
        <Panel eyebrow="BENCHMARK REPORT" title={`${selected} · class metrics`}>
          <ClassReport model={models.find((model) => model.name === selected)} />
        </Panel>
        <Panel eyebrow="RESEARCH USE" title="How to interpret these results">
          <div className="insight">
            <div className="icon"><ShieldAlert /></div>
            <div>
              <b>Not for clinical decisions</b>
              <p>
                This research demo classifies images into the model&apos;s
                training labels. It is not a diagnosis, and confidence is not a
                measure of clinical certainty.
              </p>
              <div className="chips">
                <span>4 trained classes</span>
                <span>Saved test reports</span>
                <span>{report.run_id}</span>
              </div>
            </div>
          </div>
        </Panel>
      </div>
    </>
  );
}

function Comparison({ models, selected, onSelect }) {
  return (
    <>
      <PageHeading
        eyebrow="HEAD-TO-HEAD"
        title="Three-model comparison"
        models={models}
        selected={selected}
        onSelect={onSelect}
      />
      <Panel eyebrow="SAVED TEST REPORT" title="Classification metrics by model">
        <div className="table">
          <table>
            <thead>
              <tr>
                <th>MODEL</th><th>ACCURACY</th><th>PRECISION</th>
                <th>RECALL</th><th>MACRO F1</th><th>TEST IMAGES</th>
              </tr>
            </thead>
            <tbody>
              {models.map((model) => (
                <tr
                  className={selected === model.name ? "chosen" : ""}
                  key={model.name}
                >
                  <td>
                    <i style={{ background: MODEL_COLORS[model.name] }} />
                    {model.name}
                  </td>
                  <td>{percentage(model.test_report.accuracy)}</td>
                  <td>{percentage(model.test_report.macro_precision)}</td>
                  <td>{percentage(model.test_report.macro_recall)}</td>
                  <td>{percentage(model.test_report.macro_f1)}</td>
                  <td>{model.test_report.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
      <div className="two">
        <Panel eyebrow="HELD-OUT TEST" title="Accuracy comparison">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={models.map((model) => ({
                name: model.name,
                Accuracy: model.test_report.accuracy * 100,
              }))}
            >
              <CartesianGrid stroke="#26314a" strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={[0, 100]} />
              <Tooltip {...tooltipStyle} formatter={(value) => `${Number(value).toFixed(2)}%`} />
              <Bar dataKey="Accuracy" fill="#a78bfa" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Panel>
        <Panel eyebrow="SELECTED MODEL" title={`${selected} · classification report`}>
          <ClassReport model={models.find((model) => model.name === selected)} />
        </Panel>
      </div>
    </>
  );
}

function Training({ models, selected, onSelect }) {
  const model = models.find((item) => item.name === selected);
  const loss = model.history.train_loss.map((train, index) => ({
    epoch: index + 1,
    Train: train,
    Test: model.history.test_loss[index],
  }));
  const accuracy = model.history.train_accuracy.map((train, index) => ({
    epoch: index + 1,
    Train: train * 100,
    Test: model.history.test_accuracy[index] * 100,
  }));
  return (
    <>
      <PageHeading
        eyebrow="LEARNING DYNAMICS"
        title="Training & validation telemetry"
        models={models}
        selected={selected}
        onSelect={onSelect}
      />
      <div className="stats">
        <Stat label="TRAIN LOSS" value={model.history.train_loss.at(-1).toFixed(4)} />
        <Stat label="TEST LOSS" value={model.history.test_loss.at(-1).toFixed(4)} />
        <Stat label="TRAIN ACCURACY" value={percentage(model.history.train_accuracy.at(-1))} />
        <Stat label="TEST ACCURACY" value={percentage(model.history.test_accuracy.at(-1))} />
      </div>
      <div className="two">
        <Panel eyebrow={`${model.epochs} EPOCHS`} title="Training and test loss">
          <ResponsiveContainer width="100%" height={330}>
            <LineChart data={loss}>
              <CartesianGrid stroke="#26314a" strokeDasharray="3 3" />
              <XAxis dataKey="epoch" /><YAxis />
              <Tooltip {...tooltipStyle} /><Legend />
              <Line dataKey="Train" stroke="#a78bfa" dot={false} strokeWidth={2} />
              <Line dataKey="Test" stroke="#fb7185" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </Panel>
        <Panel eyebrow="ACCURACY · %" title="Training and test accuracy">
          <ResponsiveContainer width="100%" height={330}>
            <LineChart data={accuracy}>
              <CartesianGrid stroke="#26314a" strokeDasharray="3 3" />
              <XAxis dataKey="epoch" /><YAxis domain={[0, 100]} />
              <Tooltip {...tooltipStyle} formatter={(value) => `${Number(value).toFixed(2)}%`} />
              <Legend />
              <Line dataKey="Train" stroke="#22d3ee" dot={false} strokeWidth={2} />
              <Line dataKey="Test" stroke="#fb7185" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </Panel>
      </div>
    </>
  );
}

function Confusion({ models, selected, onSelect }) {
  const model = models.find((item) => item.name === selected);
  const matrix = model.confusion_matrix;
  const max = Math.max(...matrix.flat());
  const classes = Object.keys(model.test_report.classes);
  return (
    <>
      <PageHeading
        eyebrow="ERROR ANALYSIS"
        title="Held-out test classification report"
        models={models}
        selected={selected}
        onSelect={onSelect}
      />
      <div className="two">
        <Panel eyebrow="ROWS = TRUE · COLUMNS = PREDICTED" title={`${selected} · confusion matrix`}>
          <div className="matrixWrap">
            <div className="matrixLabels top">
              {classes.map((name) => <span key={name}>{CLASS_LABELS[name] || name}</span>)}
            </div>
            <div className="matrixBody">
              <div className="matrixLabels side">
                {classes.map((name) => <span key={name}>{CLASS_LABELS[name] || name}</span>)}
              </div>
              <div className="matrix">
                {matrix.flatMap((row, rowIndex) =>
                  row.map((value, columnIndex) => (
                    <div
                      className={rowIndex === columnIndex ? "diag" : ""}
                      key={`${rowIndex}-${columnIndex}`}
                      style={{ opacity: 0.2 + 0.75 * value / max }}
                    >
                      <b>{value}</b>
                    </div>
                  )),
                )}
              </div>
            </div>
          </div>
        </Panel>
        <Panel eyebrow="TEST SET" title="Per-class report">
          <ClassReport model={model} />
          <p className="report-note">
            Matrix rows are true dataset labels; columns are model predictions.
          </p>
        </Panel>
      </div>
    </>
  );
}

function Predictions({ models, selected, onSelect, report }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const selectedModel = models.find((model) => model.name === selected);

  useEffect(() => {
    if (!file) {
      setPreview("");
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function chooseFile(nextFile) {
    setError("");
    setPrediction(null);
    if (!nextFile) return;
    if (!["image/jpeg", "image/png", "image/webp"].includes(nextFile.type)) {
      setFile(null);
      setError("Choose a JPEG, PNG, or WebP image.");
      return;
    }
    if (nextFile.size > 10 * 1024 * 1024) {
      setFile(null);
      setError("Image must be 10 MB or smaller.");
      return;
    }
    setFile(nextFile);
  }

  async function runPrediction(event) {
    event.preventDefault();
    if (!file) {
      setError("Choose an image before running inference.");
      return;
    }
    setBusy(true);
    setError("");
    setPrediction(null);
    const formData = new FormData();
    formData.append("model_name", selected);
    formData.append("image", file);
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.detail || "The prediction request failed.");
      }
      setPrediction(result);
    } catch (requestError) {
      setError(requestError.message || "Unable to reach the prediction API.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PageHeading
        eyebrow="INFERENCE CONSOLE"
        title="Upload an image for prediction"
        models={models}
        selected={selected}
        onSelect={(name) => {
          onSelect(name);
          setPrediction(null);
        }}
      />
      <div className="predGrid">
        <form className="upload" onSubmit={runPrediction}>
          <label className="drop" htmlFor="mri-file">
            {preview ? (
              <img className="upload-preview" src={preview} alt="Selected MRI scan preview" />
            ) : (
              <FileImage size={36} />
            )}
            <b>{file ? file.name : "Choose a brain MRI image"}</b>
            <small>JPEG / PNG / WebP · maximum 10 MB</small>
            <span className="choose-button"><Upload size={15} /> Browse files</span>
          </label>
          <input
            accept="image/jpeg,image/png,image/webp"
            className="file-input"
            id="mri-file"
            onChange={(event) => chooseFile(event.target.files?.[0])}
            type="file"
          />
          {error && <div className="error-message"><AlertCircle size={15} />{error}</div>}
          <button className="run" disabled={busy || !file} type="submit">
            {busy ? <LoaderCircle className="spin" size={15} /> : <ScanLine size={15} />}
            {busy ? "Running model…" : `Run ${selected} inference`}
          </button>
          <small className="privacy-note">Image is processed locally and is not saved by the API.</small>
        </form>
        <div className="result">
          <div className="resultHead">
            <div><label>SELECTED CHECKPOINT</label><h3>{selectedModel.name}</h3></div>
            <span className="live">RUN {report.run_id}</span>
          </div>
          {prediction ? (
            <>
              <div className="diagnosis">
                <div className="ring">
                  <b>{Math.round(prediction.confidence * 100)}%</b>
                  <small>confidence</small>
                </div>
                <div>
                  <label>PREDICTED TRAINING LABEL</label>
                  <h2>{CLASS_LABELS[prediction.predicted_class] || prediction.predicted_class}</h2>
                  <p>{prediction.filename}</p>
                </div>
              </div>
              <div className="probability-title">Class probabilities</div>
              {prediction.probabilities.map((item) => (
                <div className="prob" key={item.class_name}>
                  <span>{CLASS_LABELS[item.class_name] || item.class_name}</span>
                  <div><i style={{ width: `${item.probability * 100}%` }} /></div>
                  <b>{percentage(item.probability)}</b>
                </div>
              ))}
              <div className="prediction-disclaimer">
                <ShieldAlert size={16} />
                <span>{prediction.disclaimer}</span>
              </div>
              <button
                className="report-download"
                onClick={() => downloadJson(prediction, "brain-mri-prediction-report.json")}
                type="button"
              >
                <Download size={15} /> Download prediction report
              </button>
            </>
          ) : (
            <div className="empty-result">
              <ScanLine size={30} />
              <b>Prediction report will appear here</b>
              <p>Choose a scan and run one of the three trained models.</p>
              <small>Benchmark test accuracy: {percentage(selectedModel.test_report.accuracy)}</small>
            </div>
          )}
        </div>
      </div>
      {prediction && (
        <div className="two prediction-report">
          <Panel eyebrow="MODEL EVALUATION" title={`${selected} · saved test-set report`}>
            <ClassReport model={selectedModel} />
          </Panel>
          <Panel eyebrow="RUN DETAILS" title="Inference metadata">
            <div className="settings">
              <div><span>Checkpoint run</span><b>{prediction.run_id}</b></div>
              <div><span>Selected model</span><b>{prediction.model}</b></div>
              <div><span>Top-class uncertainty</span><b>{percentage(prediction.uncertainty)}</b></div>
              <div><span>Test-set support</span><b>{prediction.benchmark.support} images</b></div>
            </div>
          </Panel>
        </div>
      )}
    </>
  );
}

function App() {
  const [report, setReport] = useState(null);
  const [page, setPage] = useState("Overview");
  const [selected, setSelected] = useState("Custom CNN");
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    let attempts = 0;
    let timer;
    async function loadReport() {
      try {
        const response = await fetch("/api/models");
        const body = await response.json();
        if (!response.ok) throw new Error(body.detail || "Could not load model reports.");
        if (active) {
          setReport(body);
          setSelected(body.models[0].name);
        }
      } catch (requestError) {
        if (!active) return;
        if (attempts < 30) {
          attempts += 1;
          timer = window.setTimeout(loadReport, 1000);
        } else {
          setError(requestError.message || "Could not connect to the backend.");
        }
      }
    }
    loadReport();
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, []);

  const models = report?.models || [];
  const selectedModel = models.find((model) => model.name === selected);

  if (error) {
    return (
      <div className="startup-error">
        <AlertCircle size={28} />
        <h1>Brain MRI dashboard could not connect</h1>
        <p>{error}</p>
        <p>Start the local backend and refresh this page. The launcher is <code>start_dashboard.bat</code>.</p>
      </div>
    );
  }
  if (!report || !selectedModel) {
    return <div className="loading"><LoaderCircle className="spin" />Loading trained models and reports…</div>;
  }

  return (
    <div className="app">
      <aside>
        <div className="brand"><div><Brain size={22} /></div><b>NEURO<span>VISION</span><small>RESEARCH / MRI</small></b></div>
        <label className="sl">WORKSPACE</label>
        {NAV_ITEMS.map(([name, Icon]) => (
          <button
            className={`nav ${page === name ? "active" : ""}`}
            key={name}
            onClick={() => setPage(name)}
            type="button"
          >
            <Icon size={17} /><span>{name}</span>
          </button>
        ))}
        <button
          className="nav export-nav"
          onClick={() => downloadJson(report, `brain-mri-model-reports-${report.run_id}.json`)}
          type="button"
        >
          <Download size={17} /><span>Download all reports</span>
        </button>
        <div className="bottom"><span><CheckCircle2 size={13} /> {models.length} models ready</span><small>Run {report.run_id}</small></div>
      </aside>
      <main>
        <header>
          <div><label>RESEARCH / BRAIN MRI / BENCHMARK</label><h1>{page}</h1></div>
          <div className="header-status"><span className="status-dot" /> Backend connected</div>
        </header>
        <div className="hero">
          <div>
            <div className="eyebrow"><Activity size={13} /> TRAINED MODEL BENCHMARK</div>
            <h2>Brain MRI Tumor<br /><em>Classification</em></h2>
            <p>Explore the three trained models, their saved test-set reports, and local image inference.</p>
            <div className="tags">
              <span>✓ {report.epochs} epochs complete</span>
              <span>◉ {report.class_names.length} classes</span>
              <span>◇ Run {report.run_id}</span>
            </div>
          </div>
          <div className="orb"><div className="r r1" /><div className="r r2" /><div className="core"><Brain size={54} /><small>MODEL<br />INFERENCE</small></div></div>
        </div>
        {page === "Overview" && <Overview models={models} selected={selected} onSelect={setSelected} report={report} />}
        {page === "Model Comparison" && <Comparison models={models} selected={selected} onSelect={setSelected} />}
        {page === "Training & Validation" && <Training models={models} selected={selected} onSelect={setSelected} />}
        {page === "Predictions" && <Predictions models={models} selected={selected} onSelect={setSelected} report={report} />}
        {page === "Confusion Matrix" && <Confusion models={models} selected={selected} onSelect={setSelected} />}
        <p className="medical-disclaimer">
          <ShieldAlert size={14} /> Research demonstration only — not a medical device or a substitute for professional medical evaluation.
        </p>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
