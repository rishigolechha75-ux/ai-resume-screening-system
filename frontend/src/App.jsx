import { useEffect, useMemo, useRef, useState } from "react";
import { getConfig, getSamples, screenFiles, screenText } from "./api.js";

const WEIGHT_LABELS = {
  similarity: "Content similarity",
  skills: "Skill coverage",
  category: "Category fit",
  experience: "Experience fit",
};

const API_DOCS = import.meta.env.DEV ? "http://localhost:8000/docs" : "/docs";

const percent = (value) => `${Math.round(value * 100)}%`;

export default function App() {
  const [config, setConfig] = useState(null);
  const [jobDescription, setJobDescription] = useState("");
  const [files, setFiles] = useState([]);
  const [sampleResumes, setSampleResumes] = useState(null);
  const [weights, setWeights] = useState(null);
  const [showWeights, setShowWeights] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef(null);

  useEffect(() => {
    getConfig()
      .then((cfg) => {
        setConfig(cfg);
        setWeights(cfg.default_weights);
      })
      .catch(() => setError("Cannot reach the API. Start the backend on http://localhost:8000."));
  }, []);

  const resumeCount = sampleResumes ? sampleResumes.length : files.length;
  const canSubmit = jobDescription.trim() && resumeCount > 0 && !loading;

  function addFiles(list) {
    const accepted = config?.supported_file_types || [".pdf", ".docx", ".txt"];
    const incoming = Array.from(list).filter((f) =>
      accepted.some((ext) => f.name.toLowerCase().endsWith(ext))
    );
    setSampleResumes(null);
    setFiles((current) => {
      const names = new Set(current.map((f) => f.name));
      return [...current, ...incoming.filter((f) => !names.has(f.name))];
    });
  }

  async function loadSample() {
    setError("");
    try {
      const sample = await getSamples();
      setJobDescription(sample.job_description);
      setSampleResumes(sample.resumes);
      setFiles([]);
      setResult(null);
    } catch (e) {
      setError(e.message);
    }
  }

  function clearAll() {
    setJobDescription("");
    setFiles([]);
    setSampleResumes(null);
    setResult(null);
    setError("");
  }

  async function runScreening() {
    setLoading(true);
    setError("");
    try {
      const response = sampleResumes
        ? await screenText(jobDescription, sampleResumes, weights)
        : await screenFiles(jobDescription, files, weights);
      setResult(response);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="header">
        <div>
          <h1>AI Resume Screening System</h1>
          <p className="subtitle">
            Rank resumes against a job description using TF-IDF similarity, skill matching and an
            ML category classifier.
          </p>
        </div>
        <div className="header-actions">
          <button className="secondary" onClick={loadSample}>Try sample data</button>
          <button className="ghost" onClick={clearAll}>Clear</button>
        </div>
      </header>

      <section className="inputs">
        <div className="card">
          <label className="card-title" htmlFor="jd">1. Job description</label>
          <textarea
            id="jd"
            placeholder="Paste the job description here. Put optional skills after a 'Nice to have:' line."
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
          />
        </div>

        <div className="card">
          <span className="card-title">2. Resumes</span>
          <div
            className={`dropzone ${dragging ? "dragging" : ""}`}
            onClick={() => fileInput.current.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              addFiles(e.dataTransfer.files);
            }}
          >
            <strong>Drop resumes here</strong> or click to browse
            <span className="hint">PDF, DOCX or TXT · up to {config?.max_files ?? 50} files</span>
            <input
              ref={fileInput}
              type="file"
              multiple
              accept=".pdf,.docx,.txt"
              hidden
              onChange={(e) => {
                addFiles(e.target.files);
                e.target.value = "";
              }}
            />
          </div>

          <ul className="file-list">
            {sampleResumes &&
              sampleResumes.map((r) => (
                <li key={r.name}>
                  <span>{r.name}</span>
                  <span className="tag">sample</span>
                </li>
              ))}
            {files.map((f) => (
              <li key={f.name}>
                <span>{f.name}</span>
                <button
                  className="link"
                  onClick={() => setFiles((current) => current.filter((x) => x !== f))}
                  aria-label={`Remove ${f.name}`}
                >
                  remove
                </button>
              </li>
            ))}
          </ul>

          <button className="link" onClick={() => setShowWeights((s) => !s)}>
            {showWeights ? "Hide" : "Adjust"} scoring weights
          </button>
          {showWeights && weights && (
            <div className="weights">
              {Object.keys(WEIGHT_LABELS).map((key) => (
                <label key={key}>
                  <span>{WEIGHT_LABELS[key]}</span>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={weights[key]}
                    onChange={(e) => setWeights({ ...weights, [key]: Number(e.target.value) })}
                  />
                  <span className="num">{weights[key].toFixed(2)}</span>
                </label>
              ))}
              <button className="link" onClick={() => setWeights(config.default_weights)}>
                Reset to defaults
              </button>
            </div>
          )}
        </div>
      </section>

      <div className="run-row">
        <button className="primary" disabled={!canSubmit} onClick={runScreening}>
          {loading ? "Screening..." : `Screen ${resumeCount || ""} resume${resumeCount === 1 ? "" : "s"}`}
        </button>
        {error && <p className="error">{error}</p>}
      </div>

      {result && <Results result={result} />}

      <footer className="footer">
        <a href={API_DOCS}>REST API documentation</a> ·{" "}
        <a href="https://github.com/rishigolechha75-ux/ai-resume-screening-system">Source on GitHub</a>
      </footer>
    </div>
  );
}

function Results({ result }) {
  const { job, results, errors } = result;
  const [expanded, setExpanded] = useState(results[0]?.name ?? null);

  const counts = useMemo(
    () =>
      results.reduce((acc, r) => ({ ...acc, [r.recommendation]: (acc[r.recommendation] || 0) + 1 }), {}),
    [results]
  );

  function exportCsv() {
    const header = ["Rank", "Name", "Score", "Recommendation", "Predicted category", "Matched skills", "Missing skills"];
    const rows = results.map((r) => [
      r.rank, r.name, r.score, r.recommendation, r.predicted_category ?? "",
      r.matched_skills.join("; "), r.missing_skills.join("; "),
    ]);
    const csv = [header, ...rows].map((row) => row.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: "screening_results.csv" });
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <section className="results">
      <div className="card job-card">
        <span className="card-title">What the system read from the job description</span>
        <div className="job-grid">
          <div>
            <span className="label">Predicted role category</span>
            <strong>{job.category ?? "–"}</strong>
            {job.category_confidence != null && (
              <span className="muted"> ({percent(job.category_confidence)} confidence)</span>
            )}
          </div>
          <div>
            <span className="label">Experience required</span>
            <strong>{job.required_years != null ? `${job.required_years}+ years` : "Not stated"}</strong>
          </div>
          <div className="wide">
            <span className="label">Required skills</span>
            <Chips items={job.skills} kind="neutral" empty="None detected" />
          </div>
          {job.nice_to_have_skills.length > 0 && (
            <div className="wide">
              <span className="label">Nice to have (half weight)</span>
              <Chips items={job.nice_to_have_skills} kind="neutral" />
            </div>
          )}
        </div>
      </div>

      <div className="results-header">
        <h2>Ranked candidates</h2>
        <div className="summary">
          {["Shortlist", "Review", "Reject"].map((label) => (
            <span key={label} className={`badge ${label.toLowerCase()}`}>
              {counts[label] || 0} {label}
            </span>
          ))}
          <button className="secondary small" onClick={exportCsv}>Export CSV</button>
        </div>
      </div>

      {errors?.length > 0 && (
        <div className="warning">
          {errors.length} file{errors.length > 1 ? "s" : ""} skipped:{" "}
          {errors.map((e) => `${e.name} (${e.error})`).join(", ")}
        </div>
      )}

      <div className="candidate-list">
        {results.map((r) => (
          <article key={r.name} className="candidate">
            <button
              className="candidate-row"
              onClick={() => setExpanded(expanded === r.name ? null : r.name)}
              aria-expanded={expanded === r.name}
            >
              <span className="rank">#{r.rank}</span>
              <span className="name">{r.name}</span>
              <span className="score-bar" aria-hidden="true">
                <span style={{ width: `${r.score}%` }} className={r.recommendation.toLowerCase()} />
              </span>
              <span className="score">{r.score.toFixed(1)}</span>
              <span className={`badge ${r.recommendation.toLowerCase()}`}>{r.recommendation}</span>
            </button>

            {expanded === r.name && (
              <div className="details">
                <div className="breakdown">
                  <Metric label="Content similarity" value={Math.min(1, r.similarity / 0.5)} note={`cosine ${r.similarity.toFixed(3)}`} />
                  <Metric label="Skill coverage" value={r.skill_coverage} />
                  <Metric label="Category fit" value={r.category_fit} note={r.predicted_category ? `predicted: ${r.predicted_category}` : ""} />
                  <Metric label="Experience fit" value={r.experience_fit} note={r.years_of_experience != null ? `${r.years_of_experience} years found` : "no years found"} />
                </div>
                <div className="skills-grid">
                  <div>
                    <span className="label">Matched skills</span>
                    <Chips items={r.matched_skills} kind="good" empty="None" />
                  </div>
                  <div>
                    <span className="label">Missing skills</span>
                    <Chips items={r.missing_skills} kind="bad" empty="None" />
                  </div>
                  <div>
                    <span className="label">Other skills on resume</span>
                    <Chips items={r.extra_skills} kind="neutral" empty="None" />
                  </div>
                </div>
              </div>
            )}
          </article>
        ))}
      </div>
    </section>
  );
}

function Metric({ label, value, note }) {
  return (
    <div className="metric">
      <div className="metric-top">
        <span>{label}</span>
        <strong>{percent(value)}</strong>
      </div>
      <div className="meter"><span style={{ width: percent(value) }} /></div>
      {note && <span className="muted small-text">{note}</span>}
    </div>
  );
}

function Chips({ items, kind, empty }) {
  if (!items?.length) return <span className="muted">{empty}</span>;
  return (
    <div className="chips">
      {items.map((item) => (
        <span key={item} className={`chip ${kind}`}>{item}</span>
      ))}
    </div>
  );
}
