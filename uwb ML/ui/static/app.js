const canvas = document.getElementById("radar");
const ctx = canvas.getContext("2d");
const $ = (id) => document.getElementById(id);
const rad = (d) => (d * Math.PI) / 180;

let cfg = null;
let last = null;
let lastRx = 0;
let trail = [];
const history = [];

function geom() {
  const W = canvas.width, H = canvas.height, R = cfg.max_range_m;
  const amax = Math.max(Math.abs(cfg.angle_min_deg), Math.abs(cfg.angle_max_deg));
  const s = Math.min((W - 80) / (2 * R * Math.sin(rad(amax))), (H - 70) / R);
  return { ox: W / 2, oy: H - 30, s };
}

function pt(g, d_m, ang) {
  return [g.ox + d_m * g.s * Math.sin(rad(ang)), g.oy - d_m * g.s * Math.cos(rad(ang))];
}

function sector(g, r0, r1, fill, stroke) {
  const a0 = -Math.PI / 2 + rad(cfg.angle_min_deg), a1 = -Math.PI / 2 + rad(cfg.angle_max_deg);
  ctx.beginPath();
  ctx.arc(g.ox, g.oy, r1 * g.s, a0, a1);
  ctx.arc(g.ox, g.oy, r0 * g.s, a1, a0, true);
  ctx.closePath();
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.setLineDash([6, 4]); ctx.lineWidth = 1.5; ctx.stroke(); ctx.setLineDash([]); }
}

function drawGrid(g) {
  sector(g, 0, cfg.max_range_m, "#10151c", null);
  sector(g, cfg.blind_zone_m, cfg.useful_max_m, "rgba(60,160,90,0.14)", null);
  sector(g, 0, cfg.blind_zone_m, "rgba(120,120,120,0.30)", null);
  sector(g, 0, cfg.unlock_radius_m, null, "#ffb84d");

  ctx.strokeStyle = "#27303b"; ctx.lineWidth = 1; ctx.fillStyle = "#6c7787"; ctx.font = "11px Segoe UI";
  const a0 = -Math.PI / 2 + rad(cfg.angle_min_deg), a1 = -Math.PI / 2 + rad(cfg.angle_max_deg);
  for (let r = cfg.range_step_m; r <= cfg.max_range_m + 1e-9; r += cfg.range_step_m) {
    ctx.beginPath(); ctx.arc(g.ox, g.oy, r * g.s, a0, a1); ctx.stroke();
    const [x, y] = pt(g, r, cfg.angle_max_deg);
    ctx.fillText(r + " m", x + 4, y + 4);
  }
  const start = Math.ceil(cfg.angle_min_deg / cfg.angle_step_deg) * cfg.angle_step_deg;
  for (let a = start; a <= cfg.angle_max_deg + 1e-9; a += cfg.angle_step_deg) {
    const [x, y] = pt(g, cfg.max_range_m, a);
    ctx.beginPath(); ctx.moveTo(g.ox, g.oy); ctx.lineTo(x, y); ctx.stroke();
    const [lx, ly] = pt(g, cfg.max_range_m + 0.35, a);
    ctx.textAlign = "center"; ctx.fillText(a + "°", lx, ly); ctx.textAlign = "start";
  }
  ctx.fillStyle = "#ffb84d";
  ctx.fillText("desbloqueo " + cfg.unlock_radius_m + " m", g.ox + 8, g.oy - cfg.unlock_radius_m * g.s - 4);
  ctx.fillStyle = "#4fd18b";
  ctx.beginPath(); ctx.arc(g.ox, g.oy, 5, 0, 2 * Math.PI); ctx.fill();   // antena (origen)
}

function drawTarget(g) {
  const fresh = last && (Date.now() - lastRx) / 1000 < cfg.stale_after_s;
  if (!fresh || last.dist_cm == null || last.aoa_deg == null) return;
  const d = last.dist_cm / 100;
  if (d > cfg.max_range_m) return;
  const out = last.aoa_deg < cfg.angle_min_deg || last.aoa_deg > cfg.angle_max_deg;
  const ang = Math.max(cfg.angle_min_deg, Math.min(cfg.angle_max_deg, last.aoa_deg));
  const [x, y] = pt(g, d, ang);

  trail.push([x, y]);
  if (trail.length > 40) trail.shift();
  trail.forEach(([tx, ty], i) => {
    ctx.fillStyle = "rgba(120,180,255," + (i / trail.length) * 0.35 + ")";
    ctx.beginPath(); ctx.arc(tx, ty, 3, 0, 2 * Math.PI); ctx.fill();
  });

  const color = last.unlock ? "#4fd18b" : (last.label === cfg.positive_label ? "#ffb84d" : "#78b4ff");
  ctx.fillStyle = color; ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(x, y, 9, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  if (out) {
    ctx.strokeStyle = "#ff6b6b"; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.arc(x, y, 15, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
  }
}

function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!cfg) return;
  const g = geom();
  drawGrid(g);
  drawTarget(g);
  requestAnimationFrame(render);
}

function fmt(v, unit, digits = 0) {
  return v == null ? "-" : v.toFixed(digits) + unit;
}

function update(ev) {
  last = ev; lastRx = Date.now();
  const open = !!ev.unlock;
  $("door").textContent = open ? "PUERTA ABIERTA" : "PUERTA CERRADA";
  $("door").className = open ? "open" : "closed";
  $("label").textContent = ev.label;
  $("prob").textContent = ev.p_positive == null ? "-" : (ev.p_positive * 100).toFixed(0) + " %";
  $("dist").textContent = fmt(ev.dist_cm == null ? null : ev.dist_cm / 100, " m", 2);
  $("aoa").textContent = fmt(ev.aoa_deg, "°");
  $("board").textContent = ev.board_present == null ? "-" : (ev.board_present ? "presente" : "no");
  $("source").textContent = ev.source;

  let warn = "";
  if (ev.dist_cm == null) warn = "Sin medida";
  else if (ev.dist_cm / 100 > cfg.max_range_m) warn = "Fuera de alcance (> " + cfg.max_range_m + " m)";
  else if (ev.aoa_deg != null && (ev.aoa_deg < cfg.angle_min_deg || ev.aoa_deg > cfg.angle_max_deg)) warn = "Fuera de campo (" + ev.aoa_deg + "°)";
  $("warn").textContent = warn;

  const key = ev.label + "|" + open;
  if (!history.length || history[0].key !== key) {
    history.unshift({ key, text: new Date(ev.t * 1000).toLocaleTimeString() + "  " + ev.label + (open ? "  (abre)" : "") });
    if (history.length > 12) history.pop();
    $("history").innerHTML = history.map((h) => "<li>" + h.text + "</li>").join("");
  }
}

function connect() {
  const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  ws.onopen = () => { $("conn").textContent = "conectado"; };
  ws.onmessage = (m) => update(JSON.parse(m.data));
  ws.onclose = () => { $("conn").textContent = "reconectando..."; setTimeout(connect, 1000); };
}

setInterval(() => {
  if (last && (Date.now() - lastRx) / 1000 >= cfg.stale_after_s) {
    $("warn").textContent = "Sin datos";
    $("door").textContent = "PUERTA CERRADA"; $("door").className = "closed";
  }
}, 500);

fetch("/config").then((r) => r.json()).then((c) => { cfg = c; render(); connect(); });
