// data-binder.js (updated)
// Loads CSV/JSON from ./Outputs and binds into the dashboard.
// Works when served from a local web server (e.g., `python3 -m http.server 8000`).

(function () {
  const OUT_DIR = 'Outputs';

  // --- helpers -------------------------------------------------------------
  const $ = (sel) => document.querySelector(sel);

  function showFileWarningIfNeeded() {
    try {
      if (location.protocol === 'file:') {
        const banner = document.getElementById('file-warning');
        if (banner) banner.classList.remove('hidden');
      }
    } catch (_) {}
  }

  async function fetchText(path) {
    const res = await fetch(path, { cache: 'no-store' });
    if (!res.ok) throw new Error(`${path} -> ${res.status}`);
    return res.text();
  }

  async function fetchJSON(path) {
    const res = await fetch(path, { cache: 'no-store' });
    if (!res.ok) throw new Error(`${path} -> ${res.status}`);
    return res.json();
  }

  // Very small CSV parser (handles quotes and commas inside quotes reasonably)
  function csvToRows(text) {
    const rows = [];
    let row = [];
    let cur = '';
    let inQuotes = false;

    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      const n = text[i + 1];
      if (c === '"' && inQuotes && n === '"') {
        cur += '"'; // escaped quote
        i++;
      } else if (c === '"') {
        inQuotes = !inQuotes;
      } else if (c === ',' && !inQuotes) {
        row.push(cur); cur = '';
      } else if ((c === '\n' || (c === '\r' && n !== '\n')) && !inQuotes) {
        row.push(cur); rows.push(row); row = []; cur = '';
      } else if (c === '\r' && n === '\n' && !inQuotes) {
        row.push(cur); rows.push(row); row = []; cur = ''; i++; // CRLF
      } else {
        cur += c;
      }
    }
    if (cur.length || row.length) { row.push(cur); rows.push(row); }
    return rows.map(cols => cols.map(c => c.replace(/^\"|\"$/g, '')));
  }

  function clearEl(el) {
    while (el && el.firstChild) el.removeChild(el.firstChild);
  }

  // Build a table given headers + row arrays
  function buildTable(theadEl, tbodyEl, headers, rows) {
    clearEl(theadEl); clearEl(tbodyEl);

    const tr = document.createElement('tr');
    headers.forEach((h) => {
      const th = document.createElement('th');
      th.className = 'px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider';
      th.textContent = h;
      tr.appendChild(th);
    });
    theadEl.appendChild(tr);

    rows.forEach((r) => {
      const trb = document.createElement('tr');
      trb.className = 'hover:bg-gray-50';
      r.forEach((cell, idx) => {
        const td = document.createElement('td');
        td.className = ['px-6 py-4 text-sm text-gray-500', idx === 0 ? 'whitespace-nowrap' : ''].join(' ');
        td.textContent = cell;
        trb.appendChild(td);
      });
      tbodyEl.appendChild(trb);
    });
  }

  // --- Recent Feedback (top 5) --------------------------------------------
  async function bindRecentFeedback() {
    const tbody = document.getElementById('recent-feedback-tbody');
    const empty = document.getElementById('recent-feedback-empty');
    if (!tbody) return;
    try {
      // Prefer themes_tagged_feedback.csv if present; fallback to themes_consolidated_feedback.csv
      let csvText;
      try {
        csvText = await fetchText(`${OUT_DIR}/themes_tagged_feedback.csv`);
      } catch (_) {
        csvText = await fetchText(`${OUT_DIR}/themes_consolidated_feedback.csv`);
      }
      const rows = csvToRows(csvText);
      const header = rows[0];
      const idx = {
        customer_id: header.findIndex((h) => /customer\s*id/i.test(h)),
        created_at: header.findIndex((h) => /created_at/i.test(h)),
        message: header.findIndex((h) => /message/i.test(h)),
        concise: header.findIndex((h) => /concise\s*theme/i.test(h)),
        sentiment: header.findIndex((h) => /sentiment/i.test(h)),
        category: header.findIndex((h) => /theme\s*category|canonical/i.test(h))
      };
      const bodyRows = rows.slice(1).slice(0, 5).map((r) => [
        r[idx.customer_id] || r[0],
        r[idx.created_at] || '',
        r[idx.message] || '',
        r[idx.concise] || '',
        r[idx.sentiment] || '',
        r[idx.category] || ''
      ]);
      clearEl(tbody);
      bodyRows.forEach((r, i) => {
        const tr = document.createElement('tr');
        tr.className = i % 2 ? 'bg-gray-50 hover:bg-gray-100' : 'hover:bg-gray-50';
        const classes = [
          'px-6 py-4 whitespace-nowrap text-sm text-gray-500', // id
          'px-6 py-4 whitespace-nowrap text-sm text-gray-500', // created
          'px-6 py-4 text-sm text-gray-500 break-words', // msg
          'px-6 py-4 text-sm text-gray-500 break-words', // concise
          'px-6 py-4 whitespace-nowrap text-sm font-bold', // sentiment
          'px-6 py-4 whitespace-normal text-sm text-gray-500 break-words' // category
        ];
        r.forEach((val, idx) => {
          const td = document.createElement('td');
          td.className = classes[idx];
          if (idx === 4) {
            td.classList.add(/positive/i.test(val) ? 'text-green-500' : 'text-red-500');
          }
          td.textContent = val || '';
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
      if (empty) empty.classList.add('hidden');
    } catch (err) {
      if (empty) empty.classList.remove('hidden');
      console.warn('Recent feedback:', err.message);
    }
  }

  // --- Trend summary -------------------------------------------------------
  async function bindTrendSummary() {
    const el = document.getElementById('trend-summary');
    if (!el) return;
    try {
      let txt;
      try {
        const j = await fetchJSON(`${OUT_DIR}/trend_summary.json`);
        txt = j.summary || j.text || JSON.stringify(j);
      } catch (_) {
        txt = await fetchText(`${OUT_DIR}/trend_summary.txt`);
      }
      el.textContent = txt;
    } catch (err) {
      el.textContent = 'Unable to load trend summary.';
    }
  }

  // --- Statistical summary -------------------------------------------------
  async function bindStatisticalSummary() {
    const thead = document.getElementById('statistical-summary-thead');
    const tbody = document.getElementById('statistical-summary-tbody');
    const empty = document.getElementById('stats-empty');
    if (!thead || !tbody) return;
    try {
      const csv = await fetchText(`${OUT_DIR}/statistical_summary.csv`);
      const rows = csvToRows(csv);
      const headers = rows[0];
      const body = rows.slice(1);

      buildTable(thead, tbody, headers, body);
      if (empty) empty.classList.add('hidden');
    } catch (err) {
      if (empty) empty.classList.remove('hidden');
      console.warn('Statistical summary:', err.message);
    }
  }

  // --- Generic segmentation table loader ----------------------------------
  async function bindSegmentation(file, theadId, tbodyId, emptyId) {
    const thead = document.getElementById(theadId);
    const tbody = document.getElementById(tbodyId);
    const empty = document.getElementById(emptyId);
    if (!thead || !tbody) return;
    try {
      const csv = await fetchText(`${OUT_DIR}/${file}`);
      const rows = csvToRows(csv);
      buildTable(thead, tbody, rows[0], rows.slice(1));
      if (empty) empty.classList.add('hidden');
    } catch (err) {
      if (empty) empty.classList.remove('hidden');
      console.warn(`${file}:`, err.message);
    }
  }

  // --- Recommendations (keeps green Business Outcome text) -----------------
  async function bindRecommendations() {
    const wrap = document.getElementById('recs');
    if (!wrap) return;
    try {
      const data = await fetchJSON(`${OUT_DIR}/strategic_recommendations.json`);
      const list = Array.isArray(data?.recommendations) ? data.recommendations : (Array.isArray(data) ? data : []);
      const items = list.slice().sort((a, b) => (a.priority ?? 999) - (b.priority ?? 999));

      clearEl(wrap);
      items.forEach((rec, idx) => {
        const card = document.createElement('div');
        card.className = 'bg-purple-100 p-6 rounded-xl shadow-md border border-gray-200';

        const h3 = document.createElement('h3');
        h3.className = 'text-lg font-semibold text-gray-800 mb-2';
        const title = rec.recommendation_title || rec.title || rec.heading || 'Recommendation';
        const num = (rec.priority != null ? rec.priority : (idx + 1));
        h3.textContent = `${num}. ${title}`;

        const p1 = document.createElement('p');
        p1.className = 'text-gray-700';
        p1.innerHTML = `<span class="font-bold">Rationale:</span> ${rec.rationale || rec.reason || ''}`;

        const p2 = document.createElement('p');
        // Keep the green styling specifically for the business outcome line
        p2.className = 'text-green-600 font-semibold mt-2';
        p2.innerHTML = `<span class="font-bold text-gray-800">Business Outcome:</span> ${rec.business_outcome || rec.outcome || ''}`;

        card.append(h3, p1, p2);
        wrap.appendChild(card);
      });

      if (!items.length) {
        const div = document.createElement('div');
        div.className = 'text-gray-600';
        div.textContent = 'No recommendations found.';
        wrap.appendChild(div);
      }
    } catch (err) {
      console.warn('Recommendations:', err.message);
    }
  }

  // --- Explainability (single-row strip; only messages) --------------------
  async function bindExplainability() {
    const grid = document.getElementById('explainability-grid');
    if (!grid) return;
    try {
      const data = await fetchJSON(`${OUT_DIR}/explainability_samples.json`);
      clearEl(grid);

      // Normalized shape: { themeName: [{message, sentiment, specific_theme}, ...], ... }
      const entries = Array.isArray(data)
        ? data // maybe already an array of {theme, message}
        : Object.entries(data).flatMap(([theme, items]) => items.map((it) => ({ theme, ...it })));

      const byTheme = {};
      entries.forEach((row) => {
        const theme = row.theme || row.theme_category || row.category || 'Misc';
        const msg = row.message || row.text || row.sample || '';
        if (!byTheme[theme]) byTheme[theme] = [];
        if (msg) byTheme[theme].push(msg);
      });

      Object.entries(byTheme).forEach(([theme, msgs]) => {
        const card = document.createElement('div');
        card.className = 'bg-white p-4 rounded-xl shadow-md border border-gray-200 min-w-[260px] max-w-[320px]';
        const title = document.createElement('h4');
        title.className = 'text-sm font-semibold text-gray-800 mb-2';
        title.textContent = theme;
        const list = document.createElement('ul');
        list.className = 'space-y-2';
        msgs.slice(0, 6).forEach((m) => {
          const li = document.createElement('li');
          li.className = 'text-xs text-gray-600 bg-gray-50 rounded-lg p-2';
          li.textContent = m;
          list.appendChild(li);
        });
        card.append(title, list);
        grid.appendChild(card);
      });
    } catch (err) {
      console.warn('Explainability:', err.message);
    }
  }

  // --- Boot ---------------------------------------------------------------
  async function boot() {
    showFileWarningIfNeeded();
    await Promise.all([
      bindRecentFeedback(),
      bindTrendSummary(),
      bindStatisticalSummary(),
      bindSegmentation('negative_sentiment_ratio.csv', 'neg-sentiment-thead', 'neg-sentiment-tbody', 'neg-empty'),
      bindSegmentation('segmentation_by_approval_ratio.csv', 'seg-approval-thead', 'seg-approval-tbody', 'app-empty'),
      bindSegmentation('segmentation_by_spend.csv', 'seg-spend-thead', 'seg-spend-tbody', 'spend-empty'),
      bindSegmentation('segmentation_by_tier.csv', 'seg-tier-thead', 'seg-tier-tbody', 'tier-empty'),
      bindRecommendations(),
      bindExplainability(),
    ]);
  }

  window.addEventListener('DOMContentLoaded', boot);
})();
