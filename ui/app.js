const state = {
  busy: false,
  lastRetrieved: [],
};

const $ = (id) => document.getElementById(id);
const chat = $("chat");
const messages = $("messages");
const welcome = $("welcome");
const sourcesBody = $("sourcesBody");
const question = $("question");
const askBtn = $("askBtn");
const toastEl = $("toast");

/* ------------------------------------------------------------------ utils */
function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function toast(message, isError = false) {
  toastEl.textContent = message;
  toastEl.classList.toggle("err", isError);
  toastEl.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => toastEl.classList.remove("show"), 3200);
}

function fmtMs(ms) {
  return ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : Math.round(ms) + " ms";
}

/* ------------------------------------------------------- citation markup */
function renderAnswer(text) {
  const html = escapeHtml(text).replace(
    /\[(\d{1,2})\]/g,
    (_, n) => `<button class="cite" data-idx="${n}">${n}</button>`
  );

  const out = [];
  let inList = false;
  for (const raw of html.split("\n")) {
    const line = raw.trim();
    if (line.startsWith("- ")) {
      if (!inList) {
        out.push("<ul>");
        inList = true;
      }
      out.push(`<li>${line.slice(2)}</li>`);
      continue;
    }
    if (inList) {
      out.push("</ul>");
      inList = false;
    }
    if (line) out.push(`<p>${line}</p>`);
  }
  if (inList) out.push("</ul>");
  return out.join("");
}

/* ---------------------------------------------------------------- sources */
function renderSources(retrieved, citations) {
  const cited = new Set((citations || []).map((c) => c.index));
  state.lastRetrieved = retrieved || [];

  if (!retrieved || retrieved.length === 0) {
    sourcesBody.innerHTML = '<p class="empty">Nothing retrieved for that question.</p>';
    return;
  }

  sourcesBody.innerHTML = retrieved
    .map((r, i) => {
      const chunk = r.chunk;
      const rank = i + 1;
      const isCited = cited.has(i + 1) || cited.has(rank);
      const score = r.rerank_score != null ? r.rerank_score : r.score;
      return `
        <div class="source ${isCited ? "" : ""}" data-idx="${i + 1}">
          <div class="source-top">
            <span class="source-idx">${i + 1}</span>
            <span class="source-title" title="${escapeHtml(chunk.doc_title)}">${escapeHtml(chunk.doc_title)}</span>
            <span class="source-page">p.${chunk.page_number}</span>
          </div>
          <div class="source-snippet">${escapeHtml(chunk.text.slice(0, 260))}${chunk.text.length > 260 ? "…" : ""}</div>
          <div class="source-ranks">
            <span title="cross-encoder score">score ${Number(score).toFixed(3)}</span>
            <span class="${r.dense_rank ? "hit" : ""}" title="rank from dense retrieval">dense #${r.dense_rank ?? "–"}</span>
            <span class="${r.bm25_rank ? "hit" : ""}" title="rank from BM25">bm25 #${r.bm25_rank ?? "–"}</span>
          </div>
        </div>`;
    })
    .join("");
}

document.addEventListener("click", (event) => {
  const cite = event.target.closest(".cite");
  if (cite) {
    const idx = cite.dataset.idx;
    document.querySelectorAll(".cite").forEach((c) => c.classList.remove("active"));
    cite.classList.add("active");
    const card = sourcesBody.querySelector(`.source[data-idx="${idx}"]`);
    if (card) {
      document.querySelectorAll(".source").forEach((s) => s.classList.remove("active"));
      card.classList.add("active");
      card.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }
});

/* ---------------------------------------------------------------- trace */
function renderTrace(trace, latency) {
  if (!trace || trace.length === 0) return "";
  const max = Math.max(...trace.map((t) => t.ms), 1);
  const rows = trace
    .map(
      (t) => `
      <div class="trace-row">
        <span>${escapeHtml(t.name)}</span>
        <span class="bar"><i style="width:${Math.max(2, (t.ms / max) * 100)}%"></i></span>
        <span class="ms">${t.ms.toFixed(1)}</span>
      </div>`
    )
    .join("");
  return `
    <details class="trace">
      <summary><span>pipeline trace</span><span>${fmtMs(latency)} total</span></summary>
      <div class="trace-grid">${rows}</div>
    </details>`;
}

/* ------------------------------------------------------------- messages */
function addUserMessage(text) {
  welcome.style.display = "none";
  const el = document.createElement("div");
  el.className = "msg user";
  el.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  messages.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
}

function addThinking() {
  const el = document.createElement("div");
  el.className = "msg assistant";
  el.id = "thinking";
  el.innerHTML = '<div class="thinking"><span></span><span></span><span></span></div>';
  messages.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
  return el;
}

function addAnswer(payload) {
  const el = document.createElement("div");
  el.className = "msg assistant";

  const groundedClass = payload.grounded ? "ok" : "no";
  const groundedLabel = payload.grounded ? "grounded" : "ungrounded";
  const provider = payload.provider || "unknown";
  const model = payload.model && payload.model !== "none" ? ` · ${payload.model}` : "";

  el.innerHTML = `
    <div class="bubble">
      <div class="answer">${renderAnswer(payload.answer)}</div>
      ${renderTrace(payload.trace, payload.latency_ms)}
    </div>
    <div class="meta">
      <span class="badge ${groundedClass}">${groundedLabel} ${(payload.grounding_score * 100).toFixed(0)}%</span>
      <span>${escapeHtml(provider)}${escapeHtml(model)}</span>
      <span>${payload.citations.length} citation${payload.citations.length === 1 ? "" : "s"}</span>
      <span>${fmtMs(payload.latency_ms)}</span>
    </div>`;

  messages.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
  renderSources(payload.retrieved, payload.citations);
}

/* ------------------------------------------------------------------ ask */
async function ask(text) {
  if (state.busy || !text.trim()) return;
  state.busy = true;
  askBtn.disabled = true;

  addUserMessage(text);
  question.value = "";
  question.style.height = "auto";
  const thinking = addThinking();

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: text, include_trace: true }),
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${response.status}`);
    }

    const payload = await response.json();
    thinking.remove();
    addAnswer(payload);
  } catch (error) {
    thinking.remove();
    addAnswer({
      answer: `Request failed: ${error.message}`,
      citations: [],
      grounded: false,
      grounding_score: 0,
      provider: "error",
      model: "",
      latency_ms: 0,
      trace: [],
      retrieved: [],
    });
  } finally {
    state.busy = false;
    askBtn.disabled = false;
    question.focus();
  }
}

/* ---------------------------------------------------------------- status */
async function refreshStatus() {
  try {
    const stats = await (await fetch("/api/stats")).json();
    const pills = [
      `<span class="pill"><span class="dot"></span><b>${stats.documents}</b> docs</span>`,
      `<span class="pill"><b>${stats.chunks}</b> chunks</span>`,
      `<span class="pill"><b>${escapeHtml(stats.embedder.split("/").pop())}</b></span>`,
      `<span class="pill"><b>${escapeHtml(stats.llm_provider)}</b></span>`,
    ];
    if (!stats.indexed) {
      pills.push('<span class="pill warn"><span class="dot"></span>no index</span>');
    }
    $("pills").innerHTML = pills.join("");
  } catch {
    $("pills").innerHTML = '<span class="pill warn"><span class="dot"></span>offline</span>';
  }
}

/* ------------------------------------------------------------ suggestion */
const SAMPLE_QUESTIONS = [
  "How many days of paid annual leave do employees receive?",
  "What is the minimum password length requirement?",
  "How long do Atlas API access tokens remain valid?",
  "Within how many hours must a data breach be reported?",
];

function buildSuggestions() {
  $("suggestions").innerHTML = SAMPLE_QUESTIONS.map(
    (q) => `<button class="suggestion">${escapeHtml(q)}</button>`
  ).join("");
  $("suggestions").addEventListener("click", (event) => {
    const button = event.target.closest(".suggestion");
    if (button) ask(button.textContent);
  });
}

/* ---------------------------------------------------------------- events */
$("askForm").addEventListener("submit", (event) => {
  event.preventDefault();
  ask(question.value);
});

question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    ask(question.value);
  }
});

question.addEventListener("input", () => {
  question.style.height = "auto";
  question.style.height = Math.min(question.scrollHeight, 150) + "px";
});

$("reindexBtn").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  button.disabled = true;
  button.textContent = "Indexing…";
  try {
    const response = await fetch("/api/ingest", { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    toast(`Indexed ${payload.chunks} chunks from ${payload.documents} documents in ${fmtMs(payload.elapsed_ms)}`);
    await refreshStatus();
  } catch (error) {
    toast(`Reindex failed: ${error.message}`, true);
  } finally {
    button.disabled = false;
    button.textContent = "Reindex";
  }
});

$("fileInput").addEventListener("change", async (event) => {
  const files = event.target.files;
  if (!files || files.length === 0) return;
  const form = new FormData();
  for (const file of files) form.append("files", file);
  toast(`Uploading ${files.length} file(s)…`);
  try {
    const response = await fetch("/api/upload", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    toast(`Indexed ${payload.chunks} chunks from ${payload.documents} documents`);
    await refreshStatus();
  } catch (error) {
    toast(`Upload failed: ${error.message}`, true);
  } finally {
    event.target.value = "";
  }
});

$("clearSources").addEventListener("click", () => {
  sourcesBody.innerHTML = '<p class="empty">Retrieved passages and their ranking appear here.</p>';
});

/* ----------------------------------------------------------------- boot */
buildSuggestions();
refreshStatus();
question.focus();
