// DNA Risk Dashboard - frontend (no build step)
// Every node is built with the DOM API below, so no text from the server, the model or the user
// is ever parsed as HTML: strings always become text nodes.

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;
const fmtInt = (n) => Math.round(n).toLocaleString("en-GB");
const fmtDate = (iso, opts = { day: "numeric", month: "short" }) =>
  new Date(iso + "T00:00:00").toLocaleDateString("en-GB", opts);
const reduceMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

// ---------------------------------------------------------------------------
// DOM builders
// ---------------------------------------------------------------------------
function setProps(el, props) {
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.setAttribute("class", v);
    else if (k === "text") el.textContent = v;
    else if (k === "style") for (const [p, val] of Object.entries(v)) el.style.setProperty(p, val);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : String(v));
  }
}
function add(el, kids) {
  for (const k of kids.flat(Infinity)) {
    if (k == null || k === false) continue;
    el.append(k instanceof Node ? k : String(k));
  }
  return el;
}
function h(tag, props, ...kids) {
  const el = document.createElement(tag);
  setProps(el, props);
  return add(el, kids);
}
function s(tag, props, ...kids) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  setProps(el, props);
  return add(el, kids);
}
const fill = (el, ...kids) => { el.replaceChildren(); return add(el, kids); };

// Small drawn icons (no icon font, no glyph stand-ins)
const ICON = {
  close: () => s("svg", { viewBox: "0 0 24 24", "aria-hidden": "true" },
    s("path", { d: "M6 6l12 12M18 6L6 18", stroke: "currentColor", "stroke-width": "2", "stroke-linecap": "round" })),
  chevronDown: () => s("svg", { viewBox: "0 0 16 16", "aria-hidden": "true" },
    s("path", { d: "M4 6l4 4 4-4", fill: "none", stroke: "currentColor", "stroke-width": "2", "stroke-linecap": "round", "stroke-linejoin": "round" })),
  phone: () => s("svg", { viewBox: "0 0 20 20", "aria-hidden": "true" },
    s("path", { d: "M6.2 3.2l2 3.1-1.4 1.5a10 10 0 0 0 5.4 5.4l1.5-1.4 3.1 2-.8 2.4c-.2.6-.8 1-1.4.9C8.8 16.5 3.5 11.2 2.9 5.4c-.1-.6.3-1.2.9-1.4z", fill: "none", stroke: "currentColor", "stroke-width": "1.6", "stroke-linejoin": "round" })),
  spark: () => s("svg", { viewBox: "0 0 20 20", "aria-hidden": "true" },
    s("path", { d: "M10 2.5l1.6 4.6 4.6 1.6-4.6 1.6L10 15l-1.6-4.7L3.8 8.7l4.6-1.6z", fill: "none", stroke: "currentColor", "stroke-width": "1.5", "stroke-linejoin": "round" })),
};
const bandMark = (band) => h("i", { class: `mark ${String(band).toLowerCase()}`, "aria-hidden": "true" });
const pill = (band, text = `${band} risk`) => h("span", { class: `pill ${band}` }, bandMark(band), text);
const bandFill = (b) => `var(--${String(b).toLowerCase()}-fill)`;

// Count-up for headline numbers. The visible digits are aria-hidden while they move;
// screen readers get the final value straight away.
function countUp(to, fmt, { cls, animate = true, spoken = fmt } = {}) {
  const shown = h("span", { class: cls, "aria-hidden": "true" }, fmt(to));
  const wrap = h("span", null, shown, h("span", { class: "sr-only" }, spoken(to)));
  if (animate && !reduceMotion()) {
    const t0 = performance.now(), dur = 900;
    shown.textContent = fmt(0);
    const step = (t) => {
      const p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 3);
      shown.textContent = fmt(p < 1 ? to * e : to);
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
  return wrap;
}

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const d = (await res.json()).detail;
      detail = typeof d === "string" ? d : Array.isArray(d) ? "Some inputs are out of range" : detail;
    } catch {}
    throw new Error(detail);
  }
  return res.json();
}

const state = {
  meta: null,
  overview: null,
  wl: { date: null, hood: "", band: "", offset: 0, rows: [], total: 0, selected: null, showOutcome: false, group: true },
  checkSeq: 0,
};
const PAGE = 50;

// ---------------------------------------------------------------------------
// Theme: light / dark / auto (auto follows the system)
// ---------------------------------------------------------------------------
function storage(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch { return null; }
}
function initTheme() {
  const buttons = $$("#theme-switch button");
  const apply = (choice) => {
    if (choice === "light" || choice === "dark") document.documentElement.dataset.theme = choice;
    else delete document.documentElement.dataset.theme;
    buttons.forEach((b) => {
      const on = b.dataset.themeChoice === choice;
      b.setAttribute("aria-checked", String(on));
      b.tabIndex = on ? 0 : -1;
    });
  };
  const saved = storage("dna-theme");
  apply(saved === "light" || saved === "dark" ? saved : "auto");
  const pick = (b) => { apply(b.dataset.themeChoice); storage("dna-theme", b.dataset.themeChoice); };
  buttons.forEach((b, i) => {
    b.addEventListener("click", () => pick(b));
    b.addEventListener("keydown", (e) => {
      const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
      if (!step) return;
      e.preventDefault();
      const next = buttons[(i + step + buttons.length) % buttons.length];
      next.focus();
      pick(next);
    });
  });
}

// ---------------------------------------------------------------------------
// Tooltip. Content is a list of plain-text lines; the first line is the heading.
// ---------------------------------------------------------------------------
const tip = $("#tooltip");
const tipLines = (el, lines) => { el.dataset.tip = lines.join("\n"); return el; };
function showTip(lines, x, y) {
  if (typeof lines === "string") lines = lines.split("\n");
  fill(tip, h("b", null, lines[0]), lines.slice(1).map((l) => h("div", null, l)));
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  let left = x + 14, top = y + 14;
  if (left + r.width > innerWidth - 8) left = x - r.width - 14;
  if (top + r.height > innerHeight - 8) top = y - r.height - 14;
  tip.style.left = `${Math.max(8, left)}px`;
  tip.style.top = `${Math.max(8, top)}px`;
}
const hideTip = () => { tip.hidden = true; };
document.addEventListener("pointermove", (e) => {
  const t = e.target.closest?.("[data-tip]");
  if (t) showTip(t.dataset.tip, e.clientX, e.clientY);
  else if (!e.target.closest?.("[data-manual-tip]")) hideTip();
});
// Scrolling moves the anchor, so drop the tip, unless keyboard focus owns it (focus itself can scroll)
document.addEventListener("scroll", () => { if (!document.activeElement?.matches?.("[data-tip]")) hideTip(); }, true);
document.addEventListener("focusin", (e) => {
  const t = e.target.closest?.("[data-tip]");
  if (!t) return;
  const r = t.getBoundingClientRect();
  showTip(t.dataset.tip, r.right - 24, r.top + r.height / 2);
});
document.addEventListener("focusout", (e) => { if (e.target.closest?.("[data-tip]")) hideTip(); });
// Touch screens have no hover, so a tap shows the same detail
document.addEventListener("click", (e) => {
  const t = e.target.closest?.("[data-tip]");
  if (t) showTip(t.dataset.tip, e.clientX, e.clientY);
  else if (!e.target.closest?.("[data-manual-tip]")) hideTip();
});

// ---------------------------------------------------------------------------
// Router + sliding tab indicator
// ---------------------------------------------------------------------------
const views = { worklist: initWorklist, check: initCheck, model: initModel, monitoring: initMonitoring };
const started = new Set();
let currentView = null;
function moveTabInk() {
  const a = $(".tabs a[aria-current='page']"), ink = $(".tab-ink");
  if (!a || !ink) return;
  ink.style.width = `${a.offsetWidth}px`;  // width snaps; only the position animates
  ink.style.transform = `translateX(${a.offsetLeft}px)`;
  requestAnimationFrame(() => ink.classList.add("ready"));
}
function route() {
  const name = (location.hash.replace("#/", "") || "worklist").split("?")[0];
  const view = views[name] ? name : "worklist";
  $$(".view").forEach((v) => {
    const on = v.id === `view-${view}`;
    v.hidden = !on;
    if (on && currentView && currentView !== view) {
      v.classList.remove("enter"); void v.offsetWidth; v.classList.add("enter");
      scrollTo({ top: 0 });
    }
  });
  $$(".tabs a").forEach((a) => { if (a.dataset.view === view) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); });
  moveTabInk();
  currentView = view;
  if (!started.has(view)) { started.add(view); views[view](); }
  else if (view === "model" && state.overview) renderModelCharts(false);
  hideTip();
}

// ---------------------------------------------------------------------------
// Shared: explanation + assistant
// ---------------------------------------------------------------------------
function explainNodes(res, { animate = true } = {}) {
  const { risk, band, action, ratio_to_average: ratio } = res;
  const avg = state.meta?.average_rate ?? 0.2;
  const note = ratio >= 1.05 ? `${ratio.toFixed(1)}× the average appointment (${pct(avg)})`
    : ratio <= 0.95 ? `Below the average of ${pct(avg)}` : "About average";
  const top = res.factors.slice(0, 8);
  const maxAbs = Math.max(...top.map((f) => Math.abs(f.shap)), 0.01);
  const factors = top.map((f) => {
    const w = (Math.abs(f.shap) / maxAbs) * 50;
    const dir = f.shap > 0 ? "up" : "down";
    const word = f.shap > 0 ? "raises" : "lowers";
    const shap = `${f.shap > 0 ? "+" : ""}${f.shap.toFixed(2)}`;
    return tipLines(h("div", { class: "factor", tabindex: "0" },
      h("div", { class: "factor-label" }, h("b", null, f.label), " ", h("span", { class: "v" }, f.value)),
      h("div", { class: "factor-bar", "aria-hidden": "true" }, h("i", { class: dir, style: { width: `${w.toFixed(1)}%` } })),
      h("span", { class: "sr-only" }, `${word} risk, SHAP ${shap}`),
    ), [`${f.label}: ${f.value}`, `${word} risk (SHAP ${shap} log-odds)`]);
  });
  const marker = Math.min(Math.max(risk, 0.005), 0.995) * 100;
  return [
    h("div", { class: "score" },
      h("div", null,
        h("div", { class: "score-big" }, countUp(risk, (v) => String(Math.round(v * 100)), { animate, spoken: (v) => `${Math.round(v * 100)}%` }), h("small", { "aria-hidden": "true" }, "%")),
        h("p", { class: "score-note" }, `Predicted chance of not attending. ${note}.`)),
      pill(band)),
    h("div", { class: "scale", role: "img", "aria-label": `Risk ${pct(risk)} on a scale where low is under 20% and high is 35% or more` },
      h("div", { class: "scale-bar" }, h("i"), h("i"), h("i"), h("span", { class: "needle", style: { left: `${marker}%` } })),
      h("div", { class: "scale-ticks" },
        h("span", { style: { left: "0" } }, "0%"), h("span", { style: { left: "20%" } }, "20%"),
        h("span", { style: { left: "35%" } }, "35%"), h("span", { style: { left: "100%" } }, "100%"))),
    h("div", { class: `action ${band}` }, h("h3", null, ICON.phone(), "Suggested contact"), h("p", null, action)),
    h("h3", { class: "section-h" }, "What's driving this score ",
      h("span", { class: "muted" }, `${matchMedia("(hover: none)").matches ? "Tap" : "Hover"} a bar for detail`)),
    h("div", { class: `factors${animate && !reduceMotion() ? " grow" : ""}` }, factors),
    h("div", { class: "factor-legend", "aria-hidden": "true" },
      h("span", { class: "d" }, h("i"), "lowers risk"), h("span", { class: "u" }, "raises risk", h("i"))),
  ];
}

const SUGGESTIONS = ["Why is the risk at this level?", "What would lower the risk?", "How reliable is this model?"];

// `ref` says which prediction the assistant is about: { appointment_id } for a worklist row, or
// { appointment } with the Risk check inputs. The server rebuilds the score and factors itself.
function mountAI(el, ref) {
  const ctx = ref;
  const llm = state.meta?.llm_enabled;
  const history = [];
  const kinds = [["summary", "Write staff summary"], ["sms", "Draft SMS"], ["call", "Draft call script"]];
  const buttons = kinds.map(([k, label]) => h("button", { class: "btn quiet", type: "button", dataset: { ai: k } }, label));
  const out = h("div", { class: "ai-result", "aria-live": "polite" });
  fill(el,
    h("div", { class: "ai-head" },
      h("h3", null, ICON.spark(), "Assistant"),
      h("p", null, llm ? "Explains the prediction in plain English. It never changes the score." : "Template text. Set HF_TOKEN on the server to enable the AI model.")),
    h("div", { class: "ai-actions" }, buttons),
    out);

  buttons.forEach((btn) => btn.addEventListener("click", async () => {
    const kind = btn.dataset.ai;
    buttons.forEach((b) => { b.disabled = true; });
    fill(out, h("div", { class: "ai-out muted" }, "Writing…"));
    try {
      const r = kind === "summary"
        ? await api("/api/summary", ctx)
        : await api("/api/reminder", { context: ctx, channel: kind });
      const label = kind === "summary" ? "Staff summary" : kind === "sms" ? "SMS draft" : "Call script";
      const src = r.source === "llm" ? `Written by ${state.meta.llm_model}. Check before use.`
        : r.error === "rate_limited" ? "AI usage limit reached, showing template text."
        : r.error ? "AI unavailable, showing template text." : "Template text.";
      const copy = kind === "summary" ? null : h("button", {
        class: "btn quiet copy", type: "button",
        onclick: async (e) => {
          try { await navigator.clipboard.writeText(r.text); e.currentTarget.textContent = "Copied"; }
          catch { e.currentTarget.textContent = "Copy failed"; }
        },
      }, "Copy");
      fill(out, h("div", { class: "ai-out" },
        h("div", { class: "ai-out-head" }, h("strong", null, label), copy),
        h("div", { class: "ai-out-text" }, r.text),
        h("span", { class: "src" }, src)));
    } catch (err) {
      fill(out, h("div", { class: "ai-out error" }, `Couldn't reach the server: ${err.message}`));
    } finally {
      buttons.forEach((b) => { b.disabled = false; });
    }
  }));

  if (!llm) return;
  const log = h("div", { class: "chat-log", "aria-live": "polite" });
  const input = h("input", { type: "text", name: "q", placeholder: "Ask about this prediction", "aria-label": "Ask about this prediction", maxlength: "400" });
  const send = h("button", { class: "btn", type: "submit" }, "Ask");
  const form = h("form", { class: "chat-form" }, input, send);
  const push = (role, text) => { log.append(h("div", { class: `msg ${role}` }, text)); log.scrollTop = log.scrollHeight; };
  const ask = async (q) => {
    q = q.trim();
    if (!q) return;
    history.push({ role: "user", content: q });
    push("user", q);
    input.value = "";
    send.disabled = true;
    try {
      const r = await api("/api/chat", { context: ctx, messages: history.slice(-8) });
      history.push({ role: "assistant", content: r.text });
      push("assistant", r.text);
    } catch (err) {
      push("assistant", `Couldn't reach the server: ${err.message}`);
    } finally {
      send.disabled = false;
    }
  };
  form.addEventListener("submit", (e) => { e.preventDefault(); ask(input.value); });
  el.append(h("div", { class: "chat" },
    log,
    h("div", { class: "chat-suggest" }, SUGGESTIONS.map((q) => h("button", { type: "button", onclick: () => ask(q) }, q))),
    form));
}

// ---------------------------------------------------------------------------
// Worklist
// ---------------------------------------------------------------------------
async function initWorklist() {
  const wl = state.wl;
  if (!state.meta.has_worklist) {
    fill($("#view-worklist"), h("div", { class: "card chart-card" },
      h("h2", null, "Worklist unavailable"),
      h("p", { class: "muted note" }, "The server couldn't find ", h("code", null, "noshow_features.parquet"), ". Run ",
        h("code", null, "python prep_noshow.py"), " in the project folder, then restart the server. The Risk check still works.")));
    return;
  }
  const hoodSel = $("#f-hood");
  add(hoodSel, state.meta.neighbourhoods.map((n) => h("option", null, n)));
  hoodSel.addEventListener("change", () => { wl.hood = hoodSel.value; loadList(true); });
  const bands = $$("#f-band button");
  const pickBand = (b) => {
    bands.forEach((x) => { x.setAttribute("aria-checked", String(x === b)); x.tabIndex = x === b ? 0 : -1; });
    wl.band = b.dataset.band; loadList(true);
  };
  bands.forEach((b, i) => {
    b.tabIndex = i === 0 ? 0 : -1;
    b.addEventListener("click", () => pickBand(b));
    b.addEventListener("keydown", (e) => {
      const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
      if (!step) return;
      e.preventDefault();
      const next = bands[(i + step + bands.length) % bands.length];
      next.focus();
      pickBand(next);
    });
  });
  $("#f-outcome").addEventListener("change", (e) => {
    wl.showOutcome = e.target.checked;
    $("#appts").classList.toggle("show-outcome", wl.showOutcome);
    renderWeek(false);
    if (wl.selected) loadDetail(wl.selected, false);
  });
  $("#f-group").addEventListener("change", (e) => { wl.group = e.target.checked; loadList(true); });
  $("#load-more").addEventListener("click", () => loadList(false));
  $("#scrim").addEventListener("click", closeDetail);
  fill($("#week"), h("div", { class: "loading", style: { "grid-column": "1 / -1" } }, "Scoring the held-out week…"));

  try {
    state.overview ??= await api("/api/overview");
  } catch (err) {
    fill($("#week"), h("div", { class: "error-box", style: { "grid-column": "1 / -1" } }, `Couldn't load the week: ${err.message}`));
    return;
  }
  wl.date = state.overview.per_day[0].date;
  renderWeek(true);
  loadList(true);
}

function renderWeek(animate) {
  const wl = state.wl;
  const days = state.overview.per_day.map((d) => {
    const total = d.appointments;
    const seg = (n, b) => h("i", { style: { width: `${(n / total) * 100}%`, background: bandFill(b) } });
    const btn = h("button", {
      type: "button", class: "day", dataset: { date: d.date }, "aria-pressed": String(d.date === wl.date),
      "aria-label": `${d.weekday} ${fmtDate(d.date, { day: "numeric", month: "long" })}: ${fmtInt(d.appointments)} appointments, about ${fmtInt(d.expected_dna)} expected no-shows`,
    },
      h("div", { class: "day-top" }, h("span", { class: "day-name" }, d.weekday.slice(0, 3)), h("span", { class: "day-date" }, fmtDate(d.date))),
      h("div", { class: "day-figure" }, h("b", null, countUp(d.expected_dna, fmtInt, { animate })), h("span", null, "expected", h("br"), "no-shows")),
      tipLines(h("div", { class: "day-bar" }, seg(d.high, "High"), seg(d.medium, "Medium"), seg(d.low, "Low")),
        [d.weekday, `High ${fmtInt(d.high)}`, `Medium ${fmtInt(d.medium)}`, `Low ${fmtInt(d.low)}`]),
      h("div", { class: "day-meta" }, h("span", null, `${fmtInt(d.appointments)} booked`), h("span", null, `${fmtInt(d.high)} high risk`),
        wl.showOutcome ? h("span", null, `${fmtInt(d.actual_dna)} actually missed`) : null));
    btn.addEventListener("click", () => {
      wl.date = d.date;
      $$("#week .day").forEach((x) => x.setAttribute("aria-pressed", String(x === btn)));
      loadList(true);
    });
    return btn;
  });
  fill($("#week"), days);
}

const stateRow = (text, cls = "") => h("tr", { class: `state-row ${cls}` }, h("td", { colspan: "7" }, text));

let listSeq = 0;
let curveDrawn = false;
async function loadList(reset) {
  const wl = state.wl;
  const seq = ++listSeq;
  if (reset) { wl.offset = 0; wl.rows = []; }
  const q = new URLSearchParams({ date: wl.date, limit: PAGE, offset: wl.offset, group_patients: wl.group });
  if (wl.hood) q.set("neighbourhood", wl.hood);
  if (wl.band) q.set("band_filter", wl.band);
  const tbody = $("#appts tbody");
  if (reset) fill(tbody, stateRow("Loading appointments…"));
  let data;
  try {
    data = await api(`/api/appointments?${q}`);
  } catch (err) {
    fill(tbody, stateRow(`Couldn't load appointments: ${err.message}`, "error"));
    return;
  }
  if (seq !== listSeq) return;
  wl.total = data.total;
  wl.rows = wl.rows.concat(data.rows);
  wl.offset += data.rows.length;

  const day = state.overview.per_day.find((d) => d.date === wl.date);
  $("#list-title").textContent = `${day.weekday} ${fmtDate(wl.date, { day: "numeric", month: "long" })}${wl.hood ? `, ${wl.hood}` : ""}`;
  const who = data.grouped && data.total !== data.appointments
    ? `${fmtInt(data.total)} patients (${fmtInt(data.appointments)} appointments)`
    : `${fmtInt(data.appointments)} appointments`;
  $("#list-sub").textContent = data.total
    ? `${who}${wl.band ? ` in the ${wl.band.toLowerCase()} band` : ""}, about ${fmtInt(data.expected_dna)} expected no-shows. Highest risk first.`
    : "";
  renderRepeatNote(data, day);
  if (reset) { renderCurve(data.risk_strip, !curveDrawn); curveDrawn = true; }

  if (!wl.rows.length) {
    fill(tbody, stateRow("No appointments match these filters. Try another neighbourhood or risk band."));
  } else {
    const rows = data.rows.map(rowNode);
    if (reset) fill(tbody, rows); else add(tbody, rows);
  }
  $("#page-info").textContent = wl.total ? `Showing ${fmtInt(wl.rows.length)} of ${fmtInt(wl.total)}` : "";
  $("#load-more").hidden = wl.rows.length >= wl.total;
  // A filter change can drop the open appointment from the list; don't leave its details on screen
  if (reset && wl.selected && !wl.rows.some((r) => r.id === wl.selected)) clearDetail();
  // Wide screens have room for the detail panel, so open the riskiest appointment straight away
  if (reset && !wl.selected && wl.rows[0] && matchMedia("(min-width: 1181px)").matches) selectRow(wl.rows[0].id);
}

function renderRepeatNote(data, day) {
  const el = $("#repeat-note");
  const top = state.overview.repeats?.top20_by_day.find((d) => d.date === day.date);
  const dominated = top && top.max_rows_one_patient >= 3;
  if (!data.grouped) {
    el.hidden = !dominated;
    if (dominated) {
      fill(el, h("strong", null, "Repeat bookings fill the top of this list."),
        ` The top 20 rows are only ${top.top20_patients} patients, and one patient has ${top.max_rows_one_patient} of them. Turn on grouping to see one row per patient.`);
    } else fill(el);
    return;
  }
  el.hidden = data.total === data.appointments;
  fill(el, "Patients with several bookings that day are shown once, at their highest-risk booking. One contact covers all of them.",
    dominated ? ` Without grouping, the top 20 rows would be only ${top.top20_patients} patients.` : "");
}

function rowNode(r) {
  const band = r.band;
  return h("tr", { tabindex: "0", dataset: { id: r.id }, "aria-current": String(r.id === state.wl.selected) },
    h("td", null, h("div", { class: "risk-cell" },
      h("b", null, `${Math.round(r.risk * 100)}%`),
      h("span", { class: "risk-track", "aria-hidden": "true" },
        h("i", { style: { width: `${Math.min(r.risk, 1) * 100}%`, background: bandFill(band) } })),
      pill(band, band))),
    h("td", null,
      h("div", { class: "appt-id" }, `#${r.id}`,
        r.appts_that_day > 1 ? tipLines(h("span", { class: "repeat-badge" }, `${r.appts_that_day} bookings`),
          [`This patient has ${r.appts_that_day} bookings on this day`]) : null),
      h("div", { class: "appt-hood" }, r.neighbourhood)),
    h("td", { class: "num hide-sm" }, String(r.age)),
    h("td", { class: "num", "data-label": "Booked" }, r.lead_time_days === 0 ? "Same day" : `${r.lead_time_days} d`),
    h("td", { class: "num", "data-label": r.prior_appts ? "Missed" : null },
      r.prior_appts ? `${r.prior_no_shows} of ${r.prior_appts}` : h("span", { class: "muted" }, "No earlier visit")),
    h("td", { class: "hide-sm hide-md" }, r.sms_received ? "Sent" : h("span", { class: "muted" }, "No")),
    h("td", { class: "outcome-col" }, r.attended ? h("span", { class: "outcome ok" }, "Attended") : h("span", { class: "outcome dna" }, "Did not attend")));
}

// Risk curve: every appointment in the list, riskiest first, as one continuous area.
// The fill is split into bands at the 35% and 20% cut-offs.
function renderCurve(all, animate) {
  const el = $("#curve");
  el.dataset.values = JSON.stringify(all);
  if (!all.length) { fill(el); el.hidden = true; return; }
  el.hidden = false;
  const W = Math.max(200, el.clientWidth), H = Math.max(80, el.clientHeight);
  const n = Math.min(all.length, Math.max(2, Math.floor(W / 2)));
  const values = Array.from({ length: n }, (_, i) => all[Math.floor((i * all.length) / n)]);
  const yMax = Math.min(1, Math.max(0.45, Math.ceil(values[0] * 10) / 10));
  const x = (i) => (n === 1 ? 0 : (i / (n - 1)) * W);
  const y = (v) => H - (Math.min(v, yMax) / yMax) * H;
  const pts = values.map((v, i) => [x(i), y(v)]);
  const edge = pts.map(([px, py], i) => `${i ? "L" : "M"}${px.toFixed(1)},${py.toFixed(1)}`).join("");
  const area = `${edge}L${W},${H}L0,${H}Z`;
  const gid = "curve-bands";
  const stop = (off, col) => s("stop", { offset: off, style: { "stop-color": col } });
  const yHigh = y(0.35) / H, yMed = y(0.2) / H;
  const label = (v, text) => {
    const ty = y(v);
    return [s("line", { class: "band-line", x1: 0, x2: W, y1: ty, y2: ty }),
      s("text", { class: "band-label", x: W - 4, y: ty - 5, "text-anchor": "end" }, text)];
  };
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none", class: animate && !reduceMotion() ? "draw" : null },
    s("defs", null, s("linearGradient", { id: gid, x1: 0, x2: 0, y1: 0, y2: H, gradientUnits: "userSpaceOnUse" },
      stop(0, "var(--high-fill)"), stop(yHigh, "var(--high-fill)"),
      stop(yHigh, "var(--medium-fill)"), stop(yMed, "var(--medium-fill)"),
      stop(yMed, "var(--low-fill)"), stop(1, "var(--low-fill)"))),
    s("path", { d: area, fill: `url(#${gid})`, opacity: ".85" }),
    s("path", { class: "edge", d: edge }),
    label(0.35, "High from 35%"), label(0.2, "Medium from 20%"),
    s("line", { class: "cursor", y1: 0, y2: H, visibility: "hidden" }),
    s("circle", { class: "cursor-dot", r: 4, visibility: "hidden" }));
  fill(el, svg);
  el.dataset.manualTip = "";
  const cursor = $(".cursor", svg), dot = $(".cursor-dot", svg);
  const at = (e) => {
    const r = svg.getBoundingClientRect();
    const i = Math.round(Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)) * (n - 1));
    const cx = (x(i) / W) * r.width, cy = (y(values[i]) / H) * r.height;
    cursor.setAttribute("x1", x(i)); cursor.setAttribute("x2", x(i)); cursor.setAttribute("visibility", "visible");
    dot.setAttribute("cx", x(i)); dot.setAttribute("cy", y(values[i])); dot.setAttribute("visibility", "visible");
    showTip([`${pct(values[i])} risk`, `${pct(i / values.length)} of this list scores higher`], r.left + cx, r.top + cy);
  };
  el.onpointermove = at;
  el.onclick = at;
  el.onpointerleave = () => { cursor.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); hideTip(); };
}

$("#appts tbody").addEventListener("click", (e) => {
  const tr = e.target.closest("tr[data-id]");
  if (tr) selectRow(+tr.dataset.id);
});
$("#appts tbody").addEventListener("keydown", (e) => {
  const tr = e.target.closest("tr[data-id]");
  if (!tr) return;
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectRow(+tr.dataset.id); }
  if (e.key === "ArrowDown") { e.preventDefault(); tr.nextElementSibling?.focus(); }
  if (e.key === "ArrowUp") { e.preventDefault(); tr.previousElementSibling?.focus(); }
});

function selectRow(id) {
  state.wl.selected = id;
  $$("#appts tbody tr").forEach((tr) => tr.setAttribute("aria-current", String(+tr.dataset.id === id)));
  loadDetail(id);
}

const EMPTY_DETAIL = [...$("#detail").childNodes].map((n) => n.cloneNode(true));
const narrow = () => !matchMedia("(min-width: 1181px)").matches;
function setOverlay(open) {
  $("#detail").classList.toggle("open", open);
  $("#scrim").hidden = false;
  $("#scrim").classList.toggle("on", open && narrow());
}
function clearDetail() {
  state.wl.selected = null;
  detailSeq++;
  setOverlay(false);
  fill($("#detail"), EMPTY_DETAIL.map((n) => n.cloneNode(true)));
}
function closeDetail() {
  if (!$("#detail").classList.contains("open") || !narrow()) return;
  setOverlay(false);
  $(`#appts tbody tr[data-id="${state.wl.selected}"]`)?.focus();
}
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDetail(); });

let detailSeq = 0;
async function loadDetail(id, animate = true) {
  const el = $("#detail");
  const seq = ++detailSeq;
  setOverlay(true);
  fill(el, h("div", { class: "loading" }, `Explaining appointment #${id}…`));
  let res;
  try {
    res = await api(`/api/appointments/${id}`);
  } catch (err) {
    fill(el, h("div", { class: "error-box" }, `Couldn't explain this appointment: ${err.message}`));
    return;
  }
  if (seq !== detailSeq) return;
  const outcome = state.wl.showOutcome
    ? h("p", { class: "small", style: { "margin-top": "4px" } }, "Outcome: ",
      h("strong", { class: `outcome ${res.attended ? "ok" : "dna"}` }, res.attended ? "Attended" : "Did not attend")) : null;
  const repeats = res.appts_that_day > 1
    ? h("div", { class: "repeat-callout" }, h("strong", null, `${res.appts_that_day} bookings for this patient on this day.`),
      " One contact covers all of them. Several same-day bookings plus a long run of missed visits can also mean duplicate records, so check the booking system before calling.",
      state.wl.showOutcome ? ` ${res.missed_that_day} of ${res.appts_that_day} were missed.` : "") : null;
  const close = h("button", { class: "icon-btn detail-close", type: "button", "aria-label": "Close details", onclick: closeDetail }, ICON.close());
  const ai = h("div", { class: "ai" });
  fill(el,
    h("div", { class: "sheet-grip", "aria-hidden": "true" }),
    h("div", { class: "detail-head" },
      h("div", null, h("h2", null, `Appointment #${res.id}`),
        h("p", { class: "muted small" }, `${res.neighbourhood}, ${fmtDate(res.date, { weekday: "long", day: "numeric", month: "long" })}`), outcome),
      close),
    repeats,
    h("div", { class: "explain" }, explainNodes(res, { animate })),
    ai);
  el.scrollTop = 0;
  mountAI(ai, { appointment_id: id });
}

// ---------------------------------------------------------------------------
// Risk check
// ---------------------------------------------------------------------------
const EXAMPLES = [
  { name: "Same-day booking", note: "Reliable attender", band: "Low",
    v: { lead_time_days: 0, age: 45, weekday: "Monday", booked_hour: 9, sms_received: false, prior_appts: 2, prior_no_shows: 0, days_since_last_appt: 30 } },
  { name: "Booked 6 weeks ahead", note: "Missed 3 of last 4", band: "High",
    v: { lead_time_days: 45, age: 19, weekday: "Friday", booked_hour: 16, sms_received: true, prior_appts: 4, prior_no_shows: 3, days_since_last_appt: 10 } },
  { name: "Older regular attender", note: "Two weeks ahead", band: "Medium",
    v: { lead_time_days: 14, age: 68, weekday: "Wednesday", booked_hour: 11, sms_received: true, conditions: ["Hypertension", "Diabetes"], prior_appts: 6, prior_no_shows: 0, days_since_last_appt: 21 } },
];

function initCheck() {
  const form = $("#check-form");
  add(form.weekday, state.meta.weekdays.map((d) => h("option", { selected: d === "Tuesday" }, d)));
  add(form.neighbourhood, state.meta.neighbourhoods.map((n) => h("option", { selected: n === state.meta.default_neighbourhood }, n)));

  const presets = EXAMPLES.map((ex) => {
    const b = h("button", { type: "button", class: "example", "aria-pressed": "false" },
      bandMark(ex.band), h("span", null, h("b", null, ex.name), h("small", null, ex.note)));
    b.addEventListener("click", () => {
      fillForm(ex.v);
      presets.forEach((p) => p.setAttribute("aria-pressed", String(p === b)));
    });
    return b;
  });
  fill($("#examples"), presets);

  let t;
  const onChange = (e) => {
    if (e.isTrusted) presets.forEach((p) => p.setAttribute("aria-pressed", "false"));
    syncOutputs(); clearTimeout(t); t = setTimeout(runCheck, 180);
  };
  form.addEventListener("input", onChange);
  form.addEventListener("change", onChange);
  form.addEventListener("submit", (e) => e.preventDefault());
  syncOutputs();
  runCheck();

  // On narrow screens the result sits below the form, so keep the score in reach
  const peek = $("#check-peek");
  peek.addEventListener("click", () => $("#check-result").scrollIntoView({ behavior: reduceMotion() ? "auto" : "smooth", block: "start" }));
  new IntersectionObserver(([entry]) => peek.classList.toggle("away", entry.isIntersecting), { threshold: 0.15 })
    .observe($("#check-result"));
}

function fillForm(v) {
  const form = $("#check-form");
  const d = { lead_time_days: 14, age: 35, weekday: "Tuesday", booked_hour: 10, sms_received: true,
    conditions: [], prior_appts: 0, prior_no_shows: 0, days_since_last_appt: 0,
    multiple_appts_same_day: false, ...v };
  for (const [k, val] of Object.entries(d)) {
    if (k === "conditions") $$('[name="conditions"]', form).forEach((c) => { c.checked = val.includes(c.value); });
    else if (typeof val === "boolean") form[k].checked = val;
    else form[k].value = val;
  }
  syncOutputs();
  runCheck();
}

function syncOutputs() {
  const f = $("#check-form");
  $("#o-lead").textContent = f.lead_time_days.value === "0" ? "Same day" : `${f.lead_time_days.value} days`;
  $("#o-hour").textContent = `${String(f.booked_hour.value).padStart(2, "0")}:00`;
  $("#o-age").textContent = f.age.value;
  // Filled slider track
  $$('input[type="range"]', f).forEach((r) => {
    r.style.setProperty("--fill", `${((r.value - r.min) / (r.max - r.min)) * 100}%`);
  });
}

function readForm() {
  const f = $("#check-form");
  const n = (x, max) => Math.min(max, Math.max(0, parseInt(x, 10) || 0));
  return {
    lead_time_days: n(f.lead_time_days.value, 365), age: n(f.age.value, 110),
    weekday: f.weekday.value,
    booked_hour: n(f.booked_hour.value, 23), sms_received: f.sms_received.checked,
    conditions: $$('[name="conditions"]:checked', f).map((c) => c.value),
    prior_appts: n(f.prior_appts.value, 500), prior_no_shows: n(f.prior_no_shows.value, 500),
    days_since_last_appt: n(f.days_since_last_appt.value, 3650),
    multiple_appts_same_day: f.multiple_appts_same_day.checked, neighbourhood: f.neighbourhood.value,
  };
}

let lastCheckKey = "";
let checkShown = false;
async function runCheck() {
  const body = readForm();
  const key = JSON.stringify(body);
  if (key === lastCheckKey) return;
  lastCheckKey = key;
  const seq = ++state.checkSeq;
  const el = $("#check-result");
  let res;
  try {
    res = await api("/api/predict", body);
  } catch (err) {
    fill(el, h("div", { class: "error-box" }, `Couldn't score this appointment: ${err.message}`));
    lastCheckKey = "";
    return;
  }
  if (seq !== state.checkSeq) return;
  const warn = body.prior_no_shows > body.prior_appts
    ? h("p", { class: "small muted warn-note" }, `Previous no-shows can't exceed previous appointments, so ${body.prior_appts} was used.`) : null;
  const peek = $("#check-peek");
  peek.hidden = false;
  peek.setAttribute("aria-label", `Score ${pct(res.risk)}, ${res.band} risk. Jump to the explanation`);
  fill(peek,
    h("span", null, h("b", null, `${Math.round(res.risk * 100)}%`), " chance of not attending"),
    h("span", { class: `pill ${res.band}` }, bandMark(res.band), `${res.band} risk`, ICON.chevronDown()));
  // Count up only on the first score; later edits update in place so the number doesn't flicker
  const ai = h("div", { class: "ai" });
  fill(el, h("h2", { class: "sr-only" }, "Score for this appointment"),
    h("div", { class: "explain" }, explainNodes(res, { animate: !checkShown }), warn), ai);
  checkShown = true;
  mountAI(ai, { appointment: body });
}

// ---------------------------------------------------------------------------
// Model view: charts (hand-built SVG)
// ---------------------------------------------------------------------------
let chartsDrawn = false;
async function initModel() {
  fill($("#kpis"), h("div", { class: "loading", style: { "grid-column": "1 / -1" } }, "Loading results…"));
  let o;
  try {
    o = state.overview ??= await api("/api/overview");
  } catch (err) {
    fill($("#kpis"), h("div", { class: "error-box", style: { "grid-column": "1 / -1" } }, `Couldn't load results: ${err.message}`));
    return;
  }
  const xgbRow = o.models.find((m) => m.model === "XGBoost") || {};
  const at = (f) => o.capture.find((c) => Math.abs(c.contacted - f) < 1e-6)?.reached ?? 0;
  const bottom30 = 1 - at(0.7);
  $("#model-lede").textContent = `Tested on ${fmtInt(o.totals.appointments)} appointments from ${fmtDate(o.window.start, { day: "numeric", month: "long" })} to ${fmtDate(o.window.end, { day: "numeric", month: "long", year: "numeric" })}, which the model never saw during training.${state.meta.calibrated ? " Scores are calibrated so they read as probabilities." : ""}`;
  const kpi = (value, text) => h("div", { class: "kpi" }, h("b", null, value), h("span", null, text));
  const p0 = (v) => pct(v);
  fill($("#kpis"),
    kpi(countUp(at(0.2), p0), "of no-shows reached by contacting the riskiest 20%, nearly twice random (1.9×)"),
    kpi(countUp(bottom30, p0), "of no-shows fall in the lowest-risk 30%, where reminders could be lighter"),
    kpi([countUp(o.totals.high_band_rate, p0), " vs ", countUp(o.totals.low_band_rate, p0)], "actual no-show rate, high band vs low band"),
    kpi(xgbRow.roc_auc != null ? countUp(xgbRow.roc_auc, (v) => v.toFixed(2)) : "–", "ROC AUC on unseen dates (1.0 is perfect, 0.5 is chance)"));

  renderModelCharts(true);
  fill($("#models-table"),
    h("table", { class: "data-table" },
      h("thead", null, h("tr", null, h("th", null, "Model"), h("th", { class: "num" }, "ROC AUC"), h("th", { class: "num" }, "PR-AUC"), h("th", { class: "num" }, "Top 20% reach"))),
      h("tbody", null, o.models.map((m) => h("tr", null,
        h("td", null, m.model), h("td", { class: "num" }, m.roc_auc.toFixed(3)),
        h("td", { class: "num" }, m.pr_auc.toFixed(3)), h("td", { class: "num" }, pct(m.noshows_captured_top20pct, 1)))))),
    h("p", { class: "small muted note" }, `Logistic regression is almost as good, so a simpler, more transparent model could be preferred in practice. PR-AUC of random guessing is ${o.totals.actual_rate.toFixed(3)}.`));
}

function renderModelCharts(first) {
  const o = state.overview;
  const animate = first && !chartsDrawn && !reduceMotion();
  captureChart($("#c-capture"), o.capture, animate);
  leadChart($("#c-lead"), o.lead_time, animate);
  calibChart($("#c-calib"), o.calibration, animate);
  importanceChart($("#c-importance"), o.importance, animate);
  chartsDrawn = true;
}

function dataTable(head, rows) {
  return h("details", null, h("summary", null, "Show data table"),
    h("table", { class: "data-table" },
      h("thead", null, h("tr", null, head.map((c, i) => h("th", { class: i ? "num" : null }, c)))),
      h("tbody", null, rows.map((r) => h("tr", null, r.map((c, i) => h("td", { class: i ? "num" : null }, String(c))))))));
}

function chartW(el) {
  const cs = getComputedStyle(el);
  return Math.max(240, Math.round(el.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight)));
}
function frame(W, H, m) {
  return { W, H, m, iw: W - m.l - m.r, ih: H - m.t - m.b };
}
function yGrid(f, ticks, y, fmt) {
  return ticks.map((t) => [
    s("g", { class: "grid" }, s("line", { x1: f.m.l, x2: f.W - f.m.r, y1: y(t), y2: y(t) })),
    s("text", { class: "tick-label", x: f.m.l - 8, y: y(t) + 4, "text-anchor": "end" }, fmt(t)),
  ]);
}
const chartHead = (title, sub) => [h("h2", null, title), h("p", { class: "sub" }, sub)];

function captureChart(el, data, animate) {
  const f = frame(chartW(el), 300, { t: 16, r: 24, b: 44, l: 48 });
  const x = (v) => f.m.l + v * f.iw;
  const y = (v) => f.m.t + (1 - v) * f.ih;
  const line = data.map((d, i) => `${i ? "L" : "M"}${x(d.contacted).toFixed(1)},${y(d.reached).toFixed(1)}`).join("");
  const area = `${line}L${x(1)},${y(0)}L${x(0)},${y(0)}Z`;
  const p20 = data.find((d) => Math.abs(d.contacted - 0.2) < 1e-6);
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const angle = Math.atan(f.ih / f.iw) * 180 / Math.PI;
  const cx = s("line", { class: "cx", y1: f.m.t, y2: y(0), stroke: "var(--ink-2)", "stroke-width": "1" });
  const cd = s("circle", { class: "dot", r: 5 });
  const cross = s("g", { visibility: "hidden" }, cx, cd);
  const hit = s("rect", { class: "hit", x: f.m.l, y: f.m.t, width: f.iw, height: f.ih });
  const svg = s("svg", {
    viewBox: `0 0 ${f.W} ${f.H}`, role: "img", "data-manual-tip": "", class: animate ? "drawn" : null,
    "aria-label": `Capture curve. Contacting the top 20% reaches ${pct(p20.reached)} of no-shows versus 20% at random.`,
  },
    yGrid(f, ticks, y, (t) => pct(t)),
    ticks.map((t) => s("text", { class: "tick-label", x: x(t), y: f.H - f.m.b + 20, "text-anchor": "middle" }, pct(t))),
    s("text", { class: "tick-label", x: f.m.l + f.iw / 2, y: f.H - 4, "text-anchor": "middle" }, "Appointments contacted, highest risk first"),
    s("line", { class: "axis-line", x1: f.m.l, x2: f.W - f.m.r, y1: y(0), y2: y(0) }),
    s("path", { class: "series-area", d: area }),
    s("line", { class: "ref-line", x1: x(0), y1: y(0), x2: x(1), y2: y(1) }),
    s("text", { class: "direct-label muted", x: x(0.62), y: y(0.62) + 22, transform: `rotate(-${angle} ${x(0.62)} ${y(0.62) + 22})` }, "Random order"),
    s("path", { class: "series-line", d: line, pathLength: "1" }),
    s("g", { class: "pop", style: { "animation-delay": ".9s" } },
      s("text", { class: "direct-label", x: x(0.45) - 6, y: y(data[9].reached) - 12, "text-anchor": "end" }, "Model ranking"),
      s("line", { x1: x(0.2), x2: x(0.2), y1: y(0), y2: y(p20.reached), stroke: "var(--ink-3)", "stroke-dasharray": "2 3" }),
      s("circle", { class: "dot", cx: x(0.2), cy: y(p20.reached), r: 5 }),
      s("text", { class: "direct-label", x: x(0.2) + 10, y: y(p20.reached) + 18 }, `Top 20% reaches ${pct(p20.reached)}`)),
    cross, hit);
  fill(el, chartHead("Who to contact first", "Share of all no-shows reached when contacting appointments in order of predicted risk."), svg,
    dataTable(["Contacted", "No-shows reached", "Random"], data.filter((_, i) => i % 2 === 0).map((d) => [pct(d.contacted), pct(d.reached, 1), pct(d.contacted)])));
  hit.addEventListener("pointermove", (e) => {
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM().inverse());
    const v = Math.min(1, Math.max(0, (p.x - f.m.l) / f.iw));
    const d = data.reduce((a, b) => (Math.abs(b.contacted - v) < Math.abs(a.contacted - v) ? b : a));
    cross.setAttribute("visibility", "visible");
    cx.setAttribute("x1", x(d.contacted)); cx.setAttribute("x2", x(d.contacted));
    cd.setAttribute("cx", x(d.contacted)); cd.setAttribute("cy", y(d.reached));
    showTip([`Contact the riskiest ${pct(d.contacted)}`, `Reach ${pct(d.reached)} of no-shows`, `Random order: ${pct(d.contacted)}`], e.clientX, e.clientY);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
}

function leadChart(el, data, animate) {
  const f = frame(chartW(el), 260, { t: 24, r: 8, b: 40, l: 40 });
  const max = 0.4;
  const y = (v) => f.m.t + (1 - v / max) * f.ih;
  const bw = f.iw / data.length;
  const peak = data.reduce((a, b) => (b.rate > a.rate ? b : a));
  const svg = s("svg", {
    viewBox: `0 0 ${f.W} ${f.H}`, role: "img", class: animate ? "drawn" : null,
    "aria-label": `No-show rate rises from ${pct(data[0].rate, 1)} for same-day bookings to ${pct(peak.rate)} for ${peak.band}.`,
  },
    yGrid(f, [0, 0.1, 0.2, 0.3, 0.4], y, (t) => pct(t)),
    data.map((d, i) => {
      const x0 = f.m.l + i * bw + bw * 0.18, w = bw * 0.64, top = y(d.rate);
      const label = d === data[0] || d === peak;
      return [
        tipLines(s("path", {
          class: "series bar-v", style: { "animation-delay": `${i * 45}ms` },
          d: `M${x0},${y(0)}V${top + 5}Q${x0},${top} ${x0 + 5},${top}H${x0 + w - 5}Q${x0 + w},${top} ${x0 + w},${top + 5}V${y(0)}Z`,
        }), [d.band, `${pct(d.rate, 1)} no-show rate`, `${fmtInt(d.n)} appointments`]),
        label ? s("text", { class: "direct-label pop", style: { "animation-delay": ".6s" }, x: x0 + w / 2, y: top - 6, "text-anchor": "middle" }, pct(d.rate, 1)) : null,
        s("text", { class: "tick-label", x: x0 + w / 2, y: f.H - f.m.b + 18, "text-anchor": "middle" }, d.band.replace(" days", "d")),
      ];
    }),
    s("line", { class: "axis-line", x1: f.m.l, x2: f.W - f.m.r, y1: y(0), y2: y(0) }));
  fill(el, chartHead("Booking lead time matters most", "No-show rate by days between booking and appointment, all 110k appointments."), svg,
    dataTable(["Booked ahead", "No-show rate", "Appointments"], data.map((d) => [d.band, pct(d.rate, 1), fmtInt(d.n)])));
}

function calibChart(el, data, animate) {
  // Narrow cards put the two legend entries on separate lines
  const cw = chartW(el), stackLegend = cw < 400, legendX = stackLegend ? 4 : 44;
  const f = frame(cw, stackLegend ? 300 : 280, { t: stackLegend ? 54 : 34, r: 16, b: 44, l: 44 });
  const max = 0.5;
  const x = (v) => f.m.l + (Math.min(v, max) / max) * f.iw;
  const y = (v) => f.m.t + (1 - v / max) * f.ih;
  const ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5];
  const cal = state.overview.calibration_summary;
  const hasRaw = cal && data.every((d) => d.raw !== undefined);
  const last = data[data.length - 1];
  const tipFor = (d, i) => [`Risk tenth ${i + 1}`,
    ...(hasRaw ? [`Before calibration ${pct(d.raw, 1)}`, `After ${pct(d.predicted, 1)}`] : [`Predicted ${pct(d.predicted, 1)}`]),
    `Actual ${pct(d.observed, 1)}`, `${fmtInt(d.n)} appointments`];
  const t = cal?.test;
  const svg = s("svg", {
    viewBox: `0 0 ${f.W} ${f.H}`, role: "img", class: animate ? "drawn" : null,
    "aria-label": `Calibration before and after. In the riskiest tenth, predicted ${hasRaw ? `${pct(last.raw)} before and ` : ""}${pct(last.predicted)} after, against ${pct(last.observed)} actual.`,
  },
    hasRaw ? s("g", { class: "legend", transform: `translate(${legendX},12)` },
      s("circle", { class: "dot-hollow", cx: 6, cy: 0, r: 5 }), s("text", { class: "tick-label", x: 16, y: 4 }, "Before calibration"),
      s("circle", { class: "dot", cx: stackLegend ? 6 : 146, cy: stackLegend ? 20 : 0, r: 5 }),
      s("text", { class: "tick-label", x: stackLegend ? 16 : 156, y: stackLegend ? 24 : 4 }, "After (used in the dashboard)")) : null,
    yGrid(f, ticks, y, (v) => pct(v)),
    ticks.map((v) => s("text", { class: "tick-label", x: x(v), y: f.H - f.m.b + 18, "text-anchor": "middle" }, pct(v))),
    s("text", { class: "tick-label", x: f.m.l + f.iw / 2, y: f.H - 4, "text-anchor": "middle" }, "Predicted rate"),
    s("line", { class: "ref-line", x1: x(0), y1: y(0), x2: x(max), y2: y(max) }),
    s("text", { class: "direct-label muted", x: x(0.14), y: y(0.26), "text-anchor": "middle" }, "Perfect match"),
    data.map((d, i) => s("g", { class: "pop", style: { "animation-delay": `${150 + i * 60}ms` } },
      hasRaw ? [s("line", { class: "link-line", x1: x(d.raw), x2: x(d.predicted), y1: y(d.observed), y2: y(d.observed) }),
        s("circle", { class: "dot-hollow", cx: x(d.raw), cy: y(d.observed), r: 5 })] : null,
      s("circle", { class: "dot", cx: x(d.predicted), cy: y(d.observed), r: 6 }),
      tipLines(s("rect", {
        class: "hit", x: Math.min(x(d.predicted), hasRaw ? x(d.raw) : x(d.predicted)) - 10, y: y(d.observed) - 10,
        width: Math.abs((hasRaw ? x(d.raw) : x(d.predicted)) - x(d.predicted)) + 20, height: 20,
      }), tipFor(d, i)))),
    hasRaw ? [s("text", { class: "direct-label muted", x: x(last.raw), y: y(last.observed) + 22, "text-anchor": "end" }, pct(last.raw)),
      s("text", { class: "direct-label", x: x(last.predicted), y: y(last.observed) - 12, "text-anchor": "middle" }, pct(last.predicted))] : null,
    s("line", { class: "axis-line", x1: f.m.l, x2: f.W - f.m.r, y1: y(0), y2: y(0) }));
  const row = (label, a, b) => h("tr", null, h("td", null, label), h("td", { class: "num" }, a), h("td", { class: "num" }, b));
  fill(el,
    chartHead("Do the percentages hold up?", `Predicted vs actual no-show rate in the held-out week, one row per tenth of appointments by risk.${hasRaw ? " Calibration moves each hollow dot to the filled one." : ""}`),
    svg,
    t ? [h("table", { class: "data-table calib-summary" },
      h("thead", null, h("tr", null, h("th", null, "Held-out week"), h("th", { class: "num" }, "Before"), h("th", { class: "num" }, "After"))),
      h("tbody", null,
        row(`Average predicted (actual ${pct(cal.test.observed_rate, 1)})`, pct(t.before.mean_predicted, 1), pct(t.after.mean_predicted, 1)),
        row(`Riskiest tenth predicted (actual ${pct(t.before.top_decile_observed, 1)})`, pct(t.before.top_decile_predicted, 1), pct(t.after.top_decile_predicted, 1)),
        row("Calibration error (lower is better)", t.before.ece.toFixed(3), t.after.ece.toFixed(3)),
        row("Brier score (lower is better)", t.before.brier.toFixed(4), t.after.brier.toFixed(4)),
        row("ROC AUC (ranking, unchanged)", t.before.roc_auc.toFixed(3), t.after.roc_auc.toFixed(3)))),
    h("p", { class: "small muted note" }, `Platt scaling fitted on ${fmtDate(cal.fitted_on.start, { day: "numeric", month: "short" })}–${fmtDate(cal.fitted_on.end, { day: "numeric", month: "short" })} (${fmtInt(cal.fitted_on.rows)} appointments), the week before the held-out week. The top end still runs a few points high.`)] : null,
    dataTable(hasRaw ? ["Risk tenth", "Before", "After", "Actual"] : ["Risk tenth", "Predicted", "Actual"],
      data.map((d, i) => hasRaw ? [i + 1, pct(d.raw, 1), pct(d.predicted, 1), pct(d.observed, 1)] : [i + 1, pct(d.predicted, 1), pct(d.observed, 1)])));
}

function importanceChart(el, data, animate) {
  const W = chartW(el);
  // Narrow cards stack each label above its bar so long feature names stay readable
  const stacked = W < 440;
  const rowH = stacked ? 46 : 28, barY = stacked ? 16 : 0;
  const f = frame(W, data.length * rowH + (stacked ? 8 : 28), { t: 4, r: 44, b: stacked ? 4 : 24, l: stacked ? 0 : Math.min(250, Math.round(W * 0.46)) });
  const max = Math.max(...data.map((d) => d.mean_abs_shap));
  const x = (v) => f.m.l + (v / max) * f.iw;
  const svg = s("svg", { viewBox: `0 0 ${f.W} ${f.H}`, role: "img", class: animate ? "drawn" : null, "aria-label": `Most important feature: ${data[0].label}.` },
    data.map((d, i) => {
      const w = Math.max(2, x(d.mean_abs_shap) - f.m.l), yy = f.m.t + i * rowH + barY;
      const label = stacked
        ? s("text", { class: "tick-label", x: 0, y: yy + 1 }, d.label)
        : s("text", { class: "tick-label", x: f.m.l - 10, y: yy + 16, "text-anchor": "end" }, d.label.length > 40 ? d.label.slice(0, 39) + "…" : d.label);
      return [label,
        s("path", {
          class: "series bar-h", style: { "animation-delay": `${i * 45}ms` },
          d: `M${f.m.l},${yy + 5}H${f.m.l + w - 5}Q${f.m.l + w},${yy + 5} ${f.m.l + w},${yy + 10}V${yy + 16}Q${f.m.l + w},${yy + 21} ${f.m.l + w - 5},${yy + 21}H${f.m.l}Z`,
        }),
        i < 3 ? s("text", { class: "direct-label pop", style: { "animation-delay": ".6s" }, x: f.m.l + w + 6, y: yy + 17 }, d.mean_abs_shap.toFixed(2)) : null,
        tipLines(s("rect", { class: "hit", x: 0, y: yy - barY, width: f.W, height: rowH }), [d.label, `Mean |SHAP| ${d.mean_abs_shap.toFixed(3)}`])];
    }),
    stacked ? null : s("line", { class: "axis-line", x1: f.m.l, x2: f.m.l, y1: f.m.t, y2: f.m.t + data.length * rowH }));
  fill(el, chartHead("What the model relies on", "Average impact on predictions (mean |SHAP|), top ten features."), svg,
    dataTable(["Feature", "Mean |SHAP|"], data.map((d) => [d.label, d.mean_abs_shap.toFixed(3)])));
}

// ---------------------------------------------------------------------------
// Monitoring view (a replay of the held-out days, labelled as a simulation)
// ---------------------------------------------------------------------------
const STATUS = { ok: ["Low", "Stable"], warning: ["Medium", "Watch"], alert: ["High", "Alert"] };
const statusPill = (st) => pill(STATUS[st][0], STATUS[st][1]);
const sortedMetric = (v, d = 3) => (v == null ? "n/a" : v.toFixed(d));

function rateChart(el, br, daily) {
  const bars = [["Training", br.train], ["Validation", br.valid], ["Test (held out)", br.test]];
  const f = frame(chartW(el), 240, { t: 24, r: 8, b: 40, l: 40 });
  const max = 0.25, y = (v) => f.m.t + (1 - v / max) * f.ih, bw = f.iw / bars.length;
  const svg = s("svg", { viewBox: `0 0 ${f.W} ${f.H}`, role: "img",
    "aria-label": `No-show rate falls from ${pct(br.train, 1)} in training to ${pct(br.valid, 1)} in validation and ${pct(br.test, 1)} in the test period.` },
    yGrid(f, [0, 0.05, 0.1, 0.15, 0.2, 0.25], y, (t) => pct(t)),
    bars.map(([name, v], i) => {
      const x0 = f.m.l + i * bw + bw * 0.22, w = bw * 0.56;
      return [
        tipLines(s("rect", { class: "series", x: x0, y: y(v), width: w, height: y(0) - y(v), rx: 5 }), [name, `${pct(v, 1)} no-show rate`]),
        s("text", { class: "direct-label", x: x0 + w / 2, y: y(v) - 6, "text-anchor": "middle" }, pct(v, 1)),
        s("text", { class: "tick-label", x: x0 + w / 2, y: f.H - f.m.b + 18, "text-anchor": "middle" }, name),
      ];
    }),
    s("line", { class: "axis-line", x1: f.m.l, x2: f.W - f.m.r, y1: y(0), y2: y(0) }));
  fill(el, chartHead("The no-show rate is drifting down", "Share of appointments missed, by period. A falling base rate is why the calibration may need refreshing over time."), svg,
    dataTable(["Test day", "Observed", "Predicted"], daily.map((d) => [fmtDate(d.date), pct(d.observed_rate, 1), pct(d.mean_predicted, 1)])));
}

async function initMonitoring() {
  fill($("#mon-kpis"), h("div", { class: "loading", style: { "grid-column": "1 / -1" } }, "Loading monitoring…"));
  let m;
  try {
    m = await api("/api/monitoring");
  } catch (err) {
    fill($("#mon-kpis"), h("div", { class: "error-box", style: { "grid-column": "1 / -1" } }, `Couldn't load monitoring: ${err.message}`));
    return;
  }
  const t = m.thresholds, weekly = m.performance.weekly, last = weekly[weekly.length - 1];
  $("#sim-note").textContent = m.simulation_note;
  const nAlerts = m.alerts.filter((a) => a.level === "alert").length;
  const kpi = (value, text) => h("div", { class: "kpi" }, h("b", null, value), h("span", null, text));
  fill($("#mon-kpis"),
    kpi(String(nAlerts), "alerts raised while replaying the held-out days"),
    kpi(sortedMetric(last.roc_auc), `ROC AUC, latest window (${fmtDate(last.start)} to ${fmtDate(last.end)}). Alert below ${t.auc_min.toFixed(2)}`),
    kpi(sortedMetric(last.ece), `Calibration error (ECE), latest window. Alert above ${t.ece_max.toFixed(2)}`),
    kpi(fmtInt(m.live.predictions), `predictions logged since the server started (model ${m.model_version})`));

  // Alerts
  fill($("#mon-alerts"), h("h2", null, "Alerts"),
    m.alerts.length
      ? h("ul", { class: "alert-list" }, m.alerts.map((a) => h("li", null, statusPill(a.level),
        h("span", null, a.message), h("span", { class: "when" }, fmtDate(a.when)))))
      : h("p", { class: "all-clear" }, "No alerts. Every check stayed inside its threshold across the replayed days."),
    h("p", { class: "rules" }, `Rules: feature or score PSI above ${t.psi_alert} (watch above ${t.psi_warn}, judged once ${t.min_days_for_drift} or more appointment days have accumulated); ROC AUC below ${t.auc_min.toFixed(2)}; calibration error above ${t.ece_max.toFixed(2)}; no-show rate more than ${(t.base_rate_shift * 100).toFixed(0)} points from training.`));

  // Drift table
  const days = m.features[0].cumulative.map((d) => d.date);
  // Colour only the windows long enough to judge; the first days are shown greyed as context
  const psiCell = (c) => {
    const v = c.psi, judged = c.days >= t.min_days_for_drift;
    return h("td", { class: `num psi-${judged && v > t.psi_alert ? "alert" : judged && v > t.psi_warn ? "warning" : "ok"}` }, v.toFixed(3));
  };
  const row = (label, status, cum) => h("tr", null, h("td", null, label), h("td", null, statusPill(status)), cum.map(psiCell));
  fill($("#mon-drift"),
    h("h2", null, "Drift against the training data"),
    h("p", { class: "sub" }, "Population Stability Index (PSI) of each feature and of the model's scores versus the training-time snapshot. Below 0.1 is stable, 0.1 to 0.2 is a moderate shift, above 0.2 is significant. Each column is cumulative: everything replayed up to that day."),
    h("div", { class: "table-wrap" }, h("table", { class: "data-table" },
      h("thead", null, h("tr", null, h("th", null, "Feature"), h("th", null, "Status"), days.map((d) => h("th", { class: "num" }, `to ${fmtDate(d)}`)))),
      h("tbody", null, m.features.map((f) => row(f.label, f.status, f.cumulative)), row("Score distribution", m.score.status, m.score.cumulative)))),
    h("p", { class: "small muted note" }, `The first days are noisy: lead time looks drifted after one or two days only because appointments on a given weekday are booked with different lead times (a single Tuesday scores 0.6 or more even inside the training data). That is why alerts are judged only once ${t.min_days_for_drift} or more appointment days are in. Score reference: ${m.score.basis}.`));

  rateChart($("#mon-rate"), m.base_rate, m.performance.daily);

  // Weekly performance
  const wk = (w) => h("tr", null, h("td", null, `${fmtDate(w.start)} to ${fmtDate(w.end)}`), h("td", { class: "num" }, String(w.days)),
    h("td", { class: "num" }, fmtInt(w.rows)),
    h("td", { class: `num${w.roc_auc != null && w.roc_auc < t.auc_min ? " psi-alert" : ""}` }, sortedMetric(w.roc_auc)),
    h("td", { class: `num${w.ece != null && w.ece > t.ece_max ? " psi-alert" : ""}` }, sortedMetric(w.ece)),
    h("td", { class: "num" }, `${pct(w.observed_rate, 1)} / ${pct(w.mean_predicted, 1)}`));
  fill($("#mon-perf"), h("h2", null, "Accuracy and calibration by week"),
    h("p", { class: "sub" }, "Calendar weeks of the held-out period. Each has only three appointment days, so treat small movements as noise."),
    h("table", { class: "data-table" },
      h("thead", null, h("tr", null, h("th", null, "Window"), h("th", { class: "num" }, "Days"), h("th", { class: "num" }, "Rows"),
        h("th", { class: "num" }, "ROC AUC"), h("th", { class: "num" }, "ECE"), h("th", { class: "num" }, "Observed / predicted"))),
      h("tbody", null, weekly.map(wk))),
    dataTable(["Day", "ROC AUC", "ECE", "Observed / predicted"], m.performance.daily.map((d) =>
      [fmtDate(d.date), sortedMetric(d.roc_auc), sortedMetric(d.ece), `${pct(d.observed_rate, 1)} / ${pct(d.mean_predicted, 1)}`])));
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
async function boot() {
  initTheme();
  try {
    state.meta = await api("/api/meta");
  } catch (err) {
    fill($("#main"), h("div", { class: "card chart-card error-box" }, `Couldn't reach the prediction server: ${err.message}. Start it with `,
      h("code", null, "uvicorn dashboard.server:app"), "."));
    return;
  }
  const st = $("#llm-status");
  st.hidden = false;
  st.classList.toggle("on", state.meta.llm_enabled);
  st.textContent = state.meta.llm_enabled ? "AI assistant on" : "AI assistant off";
  st.title = state.meta.llm_enabled ? `Model: ${state.meta.llm_model}` : "Set HF_TOKEN on the server to enable";
  addEventListener("hashchange", route);
  let rt, lastW = innerWidth;
  addEventListener("resize", () => {
    moveTabInk();
    clearTimeout(rt);
    rt = setTimeout(() => {
      if (innerWidth === lastW) return;
      lastW = innerWidth;
      if (started.has("model") && state.overview) renderModelCharts(false);
      const curve = $("#curve");
      if (curve.dataset.values) renderCurve(JSON.parse(curve.dataset.values), false);
      $("#scrim").classList.toggle("on", $("#detail").classList.contains("open") && narrow());
    }, 150);
  });
  route();
}
boot();
