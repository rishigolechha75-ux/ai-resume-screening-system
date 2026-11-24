// Small wrapper around the backend REST API.

async function request(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = body.detail;
    const message =
      typeof detail === "string"
        ? detail
        : detail?.message || (Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : null);
    throw new Error(message || `Request failed (${response.status})`);
  }
  return body;
}

export function getConfig() {
  return request("/api/config");
}

export function getSamples() {
  return request("/api/samples");
}

export function screenFiles(jobDescription, files, weights) {
  const form = new FormData();
  form.append("job_description", jobDescription);
  files.forEach((file) => form.append("files", file));
  if (weights) form.append("weights", JSON.stringify(weights));
  return request("/api/screen", { method: "POST", body: form });
}

export function screenText(jobDescription, resumes, weights) {
  return request("/api/screen/text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_description: jobDescription, resumes, weights }),
  });
}
