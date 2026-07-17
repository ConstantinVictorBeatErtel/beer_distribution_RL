(() => {
  const ROLES = ["retailer_a", "retailer_b", "wholesaler", "distributor", "factory"];
  const RETAILERS = ["retailer_a", "retailer_b"];
  const TRUNK = ["wholesaler", "distributor", "factory"];
  const COLORS = {
    retailer_a: "#5ec4a0",
    retailer_b: "#3dd6c6",
    wholesaler: "#6aa8e8",
    distributor: "#c4a0e8",
    factory: "#e8a54b",
  };
  const DEMAND_COLORS = {
    retailer_a: "#f3f6fa",
    retailer_b: "#ffd38a",
  };

  const els = {
    retailers: document.getElementById("retailers"),
    trunk: document.getElementById("trunk"),
    week: document.getElementById("week"),
    horizon: document.getElementById("horizon"),
    demand: document.getElementById("demand"),
    demandA: document.getElementById("demand-a"),
    demandB: document.getElementById("demand-b"),
    demandACard: document.getElementById("demand-a-card"),
    demandBCard: document.getElementById("demand-b-card"),
    demandChange: document.getElementById("demand-change"),
    topology: document.getElementById("topology-label"),
    demandModel: document.getElementById("demand-model"),
    weekCost: document.getElementById("week-cost"),
    cumCost: document.getElementById("cum-cost"),
    legend: document.getElementById("legend"),
    canvas: document.getElementById("order-chart"),
    play: document.getElementById("btn-play"),
    pause: document.getElementById("btn-pause"),
    step: document.getElementById("btn-step"),
    reset: document.getElementById("btn-reset"),
    speed: document.getElementById("speed"),
    speedVal: document.getElementById("speed-val"),
    conn: document.getElementById("conn-status"),
  };

  /** @type {Record<string, HTMLElement>} */
  const nodes = {};
  /** @type {Record<string, {orders?: HTMLElement, ships?: HTMLElement}>} */
  const edges = {};

  let history = [];
  let playing = false;
  let terminated = false;
  let reconnectTimer = null;
  let ws = null;

  function edgeHtml() {
    return `
      <div class="edge-flow orders">
        <span class="qty" data-kind="orders">0</span>
        <span>order →</span>
      </div>
      <div class="edge-flow ships">
        <span class="qty" data-kind="ships">0</span>
        <span>← shipment</span>
      </div>
    `;
  }

  function nodeHtml(role) {
    const label = role.replace("_", " ");
    return `
      <h3 class="node-title">${label}</h3>
      <div class="stats">
        <div class="stat"><span>Inventory</span><strong data-k="inv">0</strong></div>
        <div class="stat backlog"><span>Backlog</span><strong data-k="bl">0</strong></div>
        <div class="stat"><span>Order</span><strong data-k="ord">0</strong></div>
        <div class="stat"><span>Received</span><strong data-k="ship">0</strong></div>
        <div class="cost-stat">
          <span>Player cost</span>
          <strong data-k="cost">0.0</strong>
          <small data-k="total-cost">total 0.0</small>
        </div>
      </div>
    `;
  }

  function makeNode(role) {
    const node = document.createElement("div");
    node.className = "node";
    node.dataset.role = role;
    node.innerHTML = nodeHtml(role);
    nodes[role] = node;
    return node;
  }

  function bindEdge(el, key) {
    edges[key] = {
      orders: el.querySelector('[data-kind="orders"]'),
      ships: el.querySelector('[data-kind="ships"]'),
    };
  }

  function buildBoard() {
    els.retailers.innerHTML = "";
    els.trunk.innerHTML = "";
    els.legend.innerHTML = "";

    for (const role of RETAILERS) {
      const wrap = document.createElement("div");
      wrap.className = "retailer-slot";
      wrap.appendChild(makeNode(role));
      els.retailers.appendChild(wrap);
    }

    document.querySelectorAll(".fork-edge").forEach((el) => {
      const key = el.dataset.edge;
      el.className = "edge fork-edge";
      el.innerHTML = edgeHtml();
      bindEdge(el, key);
    });

    TRUNK.forEach((role, i) => {
      const wrap = document.createElement("div");
      wrap.className = "node-wrap";
      if (i > 0) {
        const edge = document.createElement("div");
        edge.className = "edge";
        edge.innerHTML = edgeHtml();
        wrap.appendChild(edge);
        bindEdge(edge, role);
      }
      wrap.appendChild(makeNode(role));
      els.trunk.appendChild(wrap);
    });

    for (const role of ROLES) {
      const li = document.createElement("li");
      li.innerHTML = `<span class="swatch" style="background:${COLORS[role]}"></span>${role.replace("_", " ")}`;
      els.legend.appendChild(li);
    }
    for (const role of RETAILERS) {
      const li = document.createElement("li");
      li.innerHTML = `<span class="swatch demand-swatch" style="border-color:${DEMAND_COLORS[role]}"></span>demand ${role.slice(-1)}`;
      els.legend.appendChild(li);
    }
  }

  function fmt(n, digits = 1) {
    if (n == null || Number.isNaN(n)) return "—";
    return Number(n).toFixed(digits);
  }

  function fillNode(role, frame, flash) {
    const node = nodes[role];
    if (!node) return;
    node.querySelector('[data-k="inv"]').textContent = String(frame.inventories?.[role] ?? 0);
    const blEl = node.querySelector('[data-k="bl"]');
    const bl = frame.backlogs?.[role] ?? 0;
    blEl.textContent = String(bl);
    blEl.classList.toggle("warn", bl > 0);
    node.querySelector('[data-k="ord"]').textContent = String(frame.orders?.[role] ?? 0);
    node.querySelector('[data-k="ship"]').textContent = String(
      frame.shipments_received?.[role] ?? 0,
    );
    node.querySelector('[data-k="cost"]').textContent = fmt(frame.local_costs?.[role] ?? 0);
    node.querySelector('[data-k="total-cost"]').textContent = `total ${fmt(
      frame.cumulative_local_costs?.[role] ?? 0,
    )}`;
    if (flash && frame.t > 0) {
      node.classList.remove("flash");
      void node.offsetWidth;
      node.classList.add("flash");
    }
  }

  function applyFrame(frame, { flash = true } = {}) {
    if (!frame) return;
    terminated = !!frame.terminated;

    els.week.textContent = String(frame.t ?? 0);
    els.horizon.textContent = `/ ${frame.horizon ?? 52}`;

    const demand = frame.customer_demand;
    const da = frame.customer_demands?.retailer_a;
    const db = frame.customer_demands?.retailer_b;
    els.demand.textContent = demand == null ? "—" : String(demand);
    els.demandA.textContent = da == null ? "—" : String(da);
    els.demandB.textContent = db == null ? "—" : String(db);

    const previous = [...history]
      .reverse()
      .find((item) => item.t < frame.t && item.customer_demand != null);
    const demandDelta = demand == null || !previous
      ? null
      : demand - previous.customer_demand;
    els.demandChange.textContent = demandDelta == null || demandDelta === 0
      ? ""
      : `${demandDelta > 0 ? "▲" : "▼"} ${Math.abs(demandDelta)}`;
    els.demandChange.className = `demand-change ${
      demandDelta > 0 ? "up" : demandDelta < 0 ? "down" : ""
    }`;

    const prevA = previous?.customer_demands?.retailer_a;
    const prevB = previous?.customer_demands?.retailer_b;
    els.demandACard.classList.toggle("changed", da != null && prevA != null && da !== prevA);
    els.demandBCard.classList.toggle("changed", db != null && prevB != null && db !== prevB);

    els.weekCost.textContent = fmt(frame.system_cost);
    els.cumCost.textContent = fmt(frame.cumulative_cost);

    for (const role of ROLES) fillNode(role, frame, flash);

    // Retailer → wholesaler fork edges
    for (const role of RETAILERS) {
      const edge = edges[role];
      if (!edge) continue;
      edge.orders.textContent = String(frame.orders?.[role] ?? 0);
      const alloc = frame.allocations?.wholesaler?.[role];
      edge.ships.textContent = String(alloc ?? frame.shipments?.[role] ?? 0);
    }

    // Trunk edges: wholesaler ← distributor ← factory
    // Between distributor and wholesaler: orders from wholesaler, ships from distributor
    if (edges.distributor) {
      edges.distributor.orders.textContent = String(frame.orders?.wholesaler ?? 0);
      edges.distributor.ships.textContent = String(frame.shipments?.distributor ?? 0);
    }
    if (edges.factory) {
      edges.factory.orders.textContent = String(frame.orders?.distributor ?? 0);
      edges.factory.ships.textContent = String(frame.shipments?.factory ?? 0);
    }

    updateButtons();
    drawChart();
  }

  function applyStatus(msg) {
    if (typeof msg.playing === "boolean") playing = msg.playing;
    if (msg.speed_ms != null) {
      els.speed.value = String(msg.speed_ms);
      els.speedVal.textContent = `${msg.speed_ms} ms`;
    }
    if (msg.horizon != null) {
      els.horizon.textContent = `/ ${msg.horizon}`;
    }
    if (msg.topology) {
      els.topology.textContent = `${msg.topology} topology`;
    }
    if (msg.demand_model) {
      const pretty = {
        correlated_y: "Correlated Y",
        ar1: "Dynamic AR(1)",
        classic_step: "Classic step",
      };
      const name = pretty[msg.demand_model] || msg.demand_model;
      els.demandModel.textContent = `${name} demand`;
    }
    updateButtons();
  }

  function updateButtons() {
    els.play.disabled = playing || terminated;
    els.pause.disabled = !playing;
    els.step.disabled = playing || terminated;
  }

  function drawChart() {
    const canvas = els.canvas;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || 960;
    const cssH = 220;
    canvas.width = Math.floor(cssW * dpr);
    canvas.height = Math.floor(cssH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.clearRect(0, 0, cssW, cssH);
    ctx.fillStyle = "rgba(26, 35, 50, 0.5)";
    ctx.fillRect(0, 0, cssW, cssH);

    const pad = { l: 36, r: 12, t: 16, b: 28 };
    const plotW = cssW - pad.l - pad.r;
    const plotH = cssH - pad.t - pad.b;

    const frames = history.filter((f) => f.t > 0);
    let maxY = 1;
    for (const f of frames) {
      maxY = Math.max(maxY, f.customer_demand ?? 0);
      for (const role of RETAILERS) {
        maxY = Math.max(maxY, f.customer_demands?.[role] ?? 0);
      }
      for (const role of ROLES) {
        maxY = Math.max(maxY, f.orders?.[role] ?? 0);
      }
    }
    maxY = Math.ceil(maxY * 1.1) || 1;

    ctx.strokeStyle = "rgba(157, 173, 196, 0.2)";
    ctx.lineWidth = 1;
    ctx.font = "11px IBM Plex Mono, monospace";
    ctx.fillStyle = "#9aadc4";
    for (let i = 0; i <= 4; i++) {
      const y = pad.t + (plotH * i) / 4;
      const val = Math.round(maxY * (1 - i / 4));
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(pad.l + plotW, y);
      ctx.stroke();
      ctx.fillText(String(val), 4, y + 4);
    }

    if (frames.length < 1) return;

    const horizon = frames[frames.length - 1].horizon || 52;
    const maxT = Math.max(horizon, ...frames.map((f) => f.t));
    ctx.fillText("week", pad.l + plotW - 28, cssH - 8);

    for (const role of ROLES) {
      ctx.beginPath();
      ctx.strokeStyle = COLORS[role];
      ctx.lineWidth = 2;
      ctx.setLineDash([]);
      frames.forEach((f, idx) => {
        const x = pad.l + ((f.t - 1) / Math.max(1, maxT - 1)) * plotW;
        const y = pad.t + plotH - ((f.orders?.[role] ?? 0) / maxY) * plotH;
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    for (const role of RETAILERS) {
      ctx.beginPath();
      ctx.strokeStyle = DEMAND_COLORS[role];
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 5]);
      frames.forEach((f, idx) => {
        const x = pad.l + ((f.t - 1) / Math.max(1, maxT - 1)) * plotW;
        const y = pad.t + plotH - ((f.customer_demands?.[role] ?? 0) / maxY) * plotH;
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  async function control(action, extra = {}) {
    const body = { action, ...extra };
    if (action !== "reset" && els.speed.value) {
      body.speed_ms = Number(els.speed.value);
    }
    const res = await fetch("/api/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`control failed: ${res.status}`);
    const data = await res.json();
    if (data.history) history = data.history;
    if (data.frame) applyFrame(data.frame, { flash: action === "step" });
    applyStatus(data);
    return data;
  }

  function setConn(text, cls) {
    els.conn.textContent = text;
    els.conn.className = `status-line ${cls || ""}`;
  }

  function connect() {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);

    ws.onopen = () => setConn("Live", "ok");
    ws.onclose = () => {
      setConn("Reconnecting…", "bad");
      reconnectTimer = setTimeout(connect, 1000);
    };
    ws.onerror = () => setConn("Connection error", "bad");

    ws.onmessage = (ev) => {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (msg.type === "snapshot") {
        if (Array.isArray(msg.history)) history = msg.history;
        applyStatus(msg);
        if (msg.frame) applyFrame(msg.frame, { flash: false });
        else if (history.length) applyFrame(history[history.length - 1], { flash: false });
        return;
      }
      if (msg.type === "status") {
        applyStatus(msg);
        return;
      }
      if (msg.type === "frame") {
        const { type: _t, ...frame } = msg;
        if (frame.t === 0) history = [frame];
        else {
          const last = history[history.length - 1];
          if (!last || last.t !== frame.t) history.push(frame);
          else history[history.length - 1] = frame;
        }
        applyFrame(frame);
      }
    };
  }

  els.play.addEventListener("click", () => control("play").catch(console.error));
  els.pause.addEventListener("click", () => control("pause").catch(console.error));
  els.step.addEventListener("click", () => control("step").catch(console.error));
  els.reset.addEventListener("click", () => {
    control("reset", { speed_ms: Number(els.speed.value) }).catch(console.error);
  });

  els.speed.addEventListener("input", () => {
    els.speedVal.textContent = `${els.speed.value} ms`;
  });
  els.speed.addEventListener("change", () => {
    const speed_ms = Number(els.speed.value);
    fetch("/api/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: playing ? "play" : "pause", speed_ms }),
    }).catch(console.error);
  });

  window.addEventListener("resize", () => drawChart());

  buildBoard();
  connect();
})();
