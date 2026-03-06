async function safeFetchJSON(path) {
  try {
    const res = await fetch(path, { cache: "no-store" });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

async function safeFetchText(path) {
  try {
    const res = await fetch(path, { cache: "no-store" });
    if (!res.ok) return null;
    return await res.text();
  } catch {
    return null;
  }
}

function parseCSV(text) {
  const rows = text.trim().split("\n");
  if (rows.length <= 1) return [];
  const headers = rows[0].split(",").map((h) => h.trim());
  return rows.slice(1).map((line) => {
    const cols = line.split(",");
    const out = {};
    headers.forEach((h, idx) => {
      out[h] = (cols[idx] || "").trim();
    });
    return out;
  });
}

function pct(x) {
  const num = Number(x);
  if (Number.isNaN(num)) return "--";
  return `${(num * 100).toFixed(1)}%`;
}

function renderOdds(rows) {
  const mount = document.getElementById("title-odds-list");
  if (!mount) return;

  if (!rows || rows.length === 0) {
    mount.innerHTML = '<li><span class="team">No model output found</span><span class="prob">--</span></li>';
    return;
  }

  rows.sort((a, b) => Number(b.title_prob || 0) - Number(a.title_prob || 0));
  const top5 = rows.slice(0, 5);
  mount.innerHTML = top5
    .map(
      (row) =>
        `<li><span class="team">${row.TEAM_ABBR || "--"}</span><span class="prob">${pct(
          row.title_prob
        )}</span></li>`
    )
    .join("");
}

function renderMetadata(meta, oddsRows) {
  const seasonEl = document.getElementById("meta-season");
  const simsEl = document.getElementById("meta-sims");
  const updatedEl = document.getElementById("meta-updated");

  if (seasonEl) seasonEl.textContent = meta?.season || oddsRows?.[0]?.SEASON || "Current";
  if (simsEl) simsEl.textContent = meta?.n_simulations ? String(meta.n_simulations) : "--";

  const now = new Date();
  const stamp = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(
    now.getDate()
  ).padStart(2, "0")} ${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}`;
  if (updatedEl) updatedEl.textContent = stamp;
}

const RAW_BASE = "https://raw.githubusercontent.com/NHX87/nba-playoff-predictor/main";

async function hydratePortfolio() {
  const [meta, oddsCsv] = await Promise.all([
    safeFetchJSON(`${RAW_BASE}/models/trained/simulation_metadata.json`),
    safeFetchText(`${RAW_BASE}/models/trained/simulation_team_odds_current.csv`),
  ]);

  const oddsRows = oddsCsv ? parseCSV(oddsCsv) : [];
  renderOdds(oddsRows);
  renderMetadata(meta, oddsRows);
}

hydratePortfolio();
