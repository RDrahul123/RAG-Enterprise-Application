const $ = (id) => document.getElementById(id);
const chat = $("chat");
const messages = $("messages");
const welcome = $("welcome");
const sourcesBody = $("sourcesBody");
const question = $("question");
const askBtn = $("askBtn");
const toastEl = $("toast");

let busy = false;

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
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => toastEl.classList.remove("show"), 3000);
}

/* Turn "[1]" into a clickable chip, and "- " lines into list items. */
function renderAnswer(text) {
  const html = escapeHtml(text).replace(
    /\[(\d{1,2})\]/g,
    (_, n) => `<button class="cite" data-idx="${n}">${n}</button>`
  );

  const parts = [];
  let inList = false;
  for (const raw of html.split("\n")) {
    const line = raw.trim();
    if (line.startsWith("- ")) {
      if (!inList) {
        parts.push("<ul>");
        inList = true;
      }
      parts.push(`<li>${line.slice(2)}</li>`);
      continue;
    }
    if (inList) {
      parts.push("</ul>");
      inList = false;
    }
    if (line) parts.push(`<p>${line}</p>`);
  }
  if (inList) parts.push("</ul>");
  return parts.join("");
}

function renderSources(sources, citations) {
  if (!sources || sources.length === 0) {
    sourcesBody.innerHTML = '<p class="empty">Nothing retrieved for that question.</p>';
    return;
  }
  const cited = new Set((citations || []).map((c) => c.index));

  sourcesBody.innerHTML = sources
    .map((source, i) => {
      const marks = [];
      if (source.dense_rank) marks.push(`dense #${source.dense_rank}`);
      if (source.bm25_rank) marks.push(`bm25 #${source.bm25_rank}`);
      if (cited.has(i + 1)) marks.push("cited");
      return `
        <div class="source" data-idx="${i + 1}">
          <div class="source-top">
            <span class="source-idx">${i + 1}</span>
            <span class="source-title">${escapeHtml(source.doc)}</span>
            <span class="source-page">p.${source.page}</span>
          </div>
          <div class="source-snippet">${escapeHtml(source.text)}</div>
          <div class="source-ranks">
            <span>score ${source.score}</span>
            ${marks.map((mark) => `<span class="hit">${mark}</span>`).join("")}
          </div>
        </div>`;
    })
    .join("");
}

document.addEventListener("click", (event) => {
  const chip = event.target.closest(".cite");
  if (!chip) return;
  document.querySelectorAll(".cite").forEach((c) => c.classList.remove("active"));
  chip.classList.add("active");
  const card = sourcesBody.querySelector(`.source[data-idx="${chip.dataset.idx}"]`);
  if (card) {
    document.querySelectorAll(".source").forEach((s) => s.classList.remove("active"));
    card.classList.add("active");
    card.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
});

function addUserMessage(text) {
  welcome.style.display = "none";
  const element = document.createElement("div");
  element.className = "msg user";
  element.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  messages.appendChild(element);
  chat.scrollTop = chat.scrollHeight;
}

function addAnswer(result) {
  const element = document.createElement("div");
  element.className = "msg assistant";
  const badge = result.grounded ? "ok" : "no";
  const label = result.grounded ? "grounded" : "ungrounded";

  element.innerHTML = `
    <div class="bubble"><div class="answer">${renderAnswer(result.answer)}</div></div>
    <div class="meta">
      <span class="badge ${badge}">${label} ${Math.round(result.grounding_score * 100)}%</span>
      <span>${escapeHtml(result.provider)}</span>
      <span>${result.citations.length} citation${result.citations.length === 1 ? "" : "s"}</span>
      <span>retrieval ${result.retrieval_ms} ms · generation ${result.generation_ms} ms</span>
    </div>`;

  messages.appendChild(element);
  chat.scrollTop = chat.scrollHeight;
  renderSources(result.sources, result.citations);
}

async function ask(text) {
  if (busy || !text.trim()) return;
  busy = true;
  askBtn.disabled = true;

  addUserMessage(text);
  question.value = "";
  question.style.height = "auto";

  const thinking = document.createElement("div");
  thinking.className = "msg assistant";
  thinking.innerHTML = '<div class="thinking"><span></span><span></span><span></span></div>';
  messages.appendChild(thinking);
  chat.scrollTop = chat.scrollHeight;

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: text }),
    });
    if (!response.ok) throw new Error((await response.json()).detail || response.status);
    thinking.remove();
    addAnswer(await response.json());
  } catch (error) {
    thinking.remove();
    addAnswer({
      answer: `Request failed: ${error.message}`,
      citations: [],
      grounded: false,
      grounding_score: 0,
      provider: "error",
      sources: [],
      retrieval_ms: 0,
      generation_ms: 0,
    });
  } finally {
    busy = false;
    askBtn.disabled = false;
    question.focus();
  }
}

async function refreshStatus() {
  try {
    const stats = await (await fetch("/api/stats")).json();
    $("pills").innerHTML = [
      `<span class="pill"><span class="dot"></span><b>${stats.documents}</b> docs</span>`,
      `<span class="pill"><b>${stats.chunks}</b> chunks</span>`,
      `<span class="pill"><b>${escapeHtml(stats.embedder.split("/").pop())}</b></span>`,
      `<span class="pill"><b>${escapeHtml(stats.llm)}</b></span>`,
    ].join("");
  } catch {
    $("pills").innerHTML = '<span class="pill warn"><span class="dot"></span>offline</span>';
  }
}

const SUGGESTIONS = [
  "How many days of paid annual leave do employees receive?",
  "What is the minimum password length requirement?",
  "How long do Atlas API access tokens remain valid?",
  "Within how many hours must a data breach be reported?",
];

$("suggestions").innerHTML = SUGGESTIONS.map(
  (text) => `<button class="suggestion">${escapeHtml(text)}</button>`
).join("");

$("suggestions").addEventListener("click", (event) => {
  const button = event.target.closest(".suggestion");
  if (button) ask(button.textContent);
});

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
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail);
    toast(`Indexed ${result.chunks} chunks from ${result.documents} documents`);
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
  try {
    const response = await fetch("/api/upload", { method: "POST", body: form });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail);
    toast(`Indexed ${result.chunks} chunks from ${result.documents} documents`);
    await refreshStatus();
  } catch (error) {
    toast(`Upload failed: ${error.message}`, true);
  } finally {
    event.target.value = "";
  }
});

$("clearSources").addEventListener("click", () => {
  sourcesBody.innerHTML = '<p class="empty">Retrieved passages appear here.</p>';
});

refreshStatus();
question.focus();
