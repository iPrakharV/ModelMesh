from __future__ import annotations


DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ModelMesh</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #09090f;
      color: #e5e7eb;
    }
    body {
      margin: 0;
      min-height: 100vh;
      background: #09090f;
    }
    main {
      width: min(1120px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 36px 0;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 24px;
      align-items: flex-end;
      margin-bottom: 24px;
    }
    h1 {
      margin: 0;
      font-size: 40px;
      line-height: 1;
      letter-spacing: 0;
    }
    p {
      color: #94a3b8;
      line-height: 1.6;
      margin: 10px 0 0;
      max-width: 620px;
    }
    .pill {
      border: 1px solid #334155;
      border-radius: 999px;
      padding: 8px 12px;
      color: #cbd5e1;
      font: 12px ui-monospace, SFMono-Regular, Menlo, monospace;
      white-space: nowrap;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    .card, .panel {
      border: 1px solid #1e293b;
      border-radius: 8px;
      background: #0f172a;
    }
    .card {
      padding: 18px;
    }
    .label {
      color: #64748b;
      font: 11px ui-monospace, SFMono-Regular, Menlo, monospace;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    .value {
      margin-top: 12px;
      font: 28px ui-monospace, SFMono-Regular, Menlo, monospace;
      color: #f8fafc;
    }
    .panel {
      margin-top: 12px;
      overflow: hidden;
    }
    .panel h2 {
      margin: 0;
      padding: 16px 18px;
      border-bottom: 1px solid #1e293b;
      font-size: 15px;
      letter-spacing: 0;
    }
    table {
      width: 100%;
      border-collapse: collapse;
    }
    th, td {
      padding: 12px 18px;
      border-bottom: 1px solid #1e293b;
      text-align: left;
      font-size: 14px;
    }
    th {
      color: #64748b;
      font: 11px ui-monospace, SFMono-Regular, Menlo, monospace;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    td.num {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      color: #e2e8f0;
      text-align: right;
    }
    .ok {
      color: #86efac;
    }
    .bad {
      color: #fca5a5;
    }
    @media (max-width: 760px) {
      header {
        display: block;
      }
      h1 {
        font-size: 32px;
      }
      .pill {
        display: inline-block;
        margin-top: 16px;
      }
      .grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      th, td {
        padding: 10px;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>ModelMesh</h1>
        <p>Gateway metrics and worker state for the local inference mesh.</p>
      </div>
      <div class="pill" id="status">loading</div>
    </header>

    <section class="grid">
      <div class="card">
        <div class="label">Requests</div>
        <div class="value" id="requests">0</div>
      </div>
      <div class="card">
        <div class="label">Cache hit rate</div>
        <div class="value" id="cache">0%</div>
      </div>
      <div class="card">
        <div class="label">p95 latency</div>
        <div class="value" id="p95">0 ms</div>
      </div>
      <div class="card">
        <div class="label">Worker errors</div>
        <div class="value" id="errors">0</div>
      </div>
    </section>

    <section class="panel">
      <h2>Workers</h2>
      <table>
        <thead>
          <tr>
            <th>URL</th>
            <th>Strategy</th>
            <th class="num">In flight</th>
            <th class="num">Requests</th>
            <th class="num">Failures</th>
            <th class="num">Latency EMA</th>
          </tr>
        </thead>
        <tbody id="workers"></tbody>
      </table>
    </section>
  </main>

  <script>
    const ids = {
      status: document.getElementById("status"),
      requests: document.getElementById("requests"),
      cache: document.getElementById("cache"),
      p95: document.getElementById("p95"),
      errors: document.getElementById("errors"),
      workers: document.getElementById("workers"),
    };

    function pct(value) {
      return `${Math.round(value * 100)}%`;
    }

    async function refresh() {
      try {
        const [health, metrics, workers] = await Promise.all([
          fetch("/health").then((response) => response.json()),
          fetch("/metrics").then((response) => response.json()),
          fetch("/workers").then((response) => response.json()),
        ]);
        ids.status.textContent = `${health.status} / ${health.router_strategy}`;
        ids.status.className = "pill ok";
        ids.requests.textContent = metrics.requests;
        ids.cache.textContent = pct(metrics.cache_hit_rate);
        ids.p95.textContent = `${metrics.p95_latency_ms} ms`;
        ids.errors.textContent = metrics.worker_errors;
        ids.workers.innerHTML = workers.map((worker) => `
          <tr>
            <td>${worker.url}</td>
            <td>${worker.strategy}</td>
            <td class="num">${worker.in_flight}</td>
            <td class="num">${worker.requests}</td>
            <td class="num ${worker.failures ? "bad" : ""}">${worker.failures}</td>
            <td class="num">${worker.latency_ema_ms} ms</td>
          </tr>
        `).join("");
      } catch (error) {
        ids.status.textContent = "offline";
        ids.status.className = "pill bad";
      }
    }

    refresh();
    setInterval(refresh, 1500);
  </script>
</body>
</html>
"""
