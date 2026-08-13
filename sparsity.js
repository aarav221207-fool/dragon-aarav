function renderSparsity(inputIdx) {
  const container = document.getElementById('vizContainer');
  const data = BDH_DATA.inputs[inputIdx];

  if (!container || !data) return;

  const captures = data.captures.filter(c => c && c.output);

  if (!captures.length) {
    container.innerHTML = `
      <div class="placeholder">
        No sparsity data available for this inference.
      </div>
    `;
    return;
  }

  const avgSparsity =
    captures.reduce((sum, cap) => sum + (cap.output.sparsity || 0), 0) /
    captures.length;

  const maxSparsity =
    Math.max(...captures.map(cap => cap.output.sparsity || 0));

  const activeRatio = 1 - avgSparsity;

  let html = `
    <div class="sparsity-dashboard">

      <div class="viz-title">
        Sparsity & Inference Analysis
      </div>

      <p class="viz-note">
        Live tensor sparsity across the current inference path.
        Select a layer to inspect its activation statistics.
      </p>

      <div class="sparsity-summary">

        <div class="sparsity-stat">
          <span>Average sparsity</span>
          <strong>${pct(avgSparsity)}</strong>
        </div>

        <div class="sparsity-stat">
          <span>Active ratio</span>
          <strong>${pct(activeRatio)}</strong>
        </div>

        <div class="sparsity-stat">
          <span>Peak sparsity</span>
          <strong>${pct(maxSparsity)}</strong>
        </div>

        <div class="sparsity-stat">
          <span>Tokens</span>
          <strong>${data.token_count}</strong>
        </div>

      </div>

      <div class="chart-section">

        <div class="chart-heading">
          <span>Layer sparsity</span>
          <span class="chart-help">Click a layer</span>
        </div>

        <div class="sparsity-chart">
  `;

  captures.forEach((cap, index) => {
    const sp = cap.output.sparsity || 0;
    const type = cap.module_path || 'unknown';

    html += `
      <button
        class="sparsity-layer"
        type="button"
        onclick="selectSparsityLayer(${index})"
      >

        <div class="sparsity-layer-header">
          <span class="sparsity-layer-name">
            ${index + 1}. ${type}
          </span>

          <span class="sparsity-layer-value">
            ${pct(sp)}
          </span>
        </div>

        <div class="sparsity-track">
          <div
            class="sparsity-fill ${type}"
            style="width:${Math.max(0, Math.min(100, sp * 100))}%"
          ></div>
        </div>

        <div class="sparsity-meta">
          <span>${cap.type}</span>
          <span>${cap.output.numel.toLocaleString()} elements</span>
        </div>

      </button>
    `;
  });

  html += `
        </div>
      </div>

      <div class="chart-section">

        <div class="chart-heading">
          <span>Inference trace</span>
          <span class="chart-help">
            activation density through the network
          </span>
        </div>

        <div class="inference-trace">
  `;

  captures.forEach((cap, index) => {
    const sp = cap.output.sparsity || 0;
    const density = 1 - sp;
    const type = cap.module_path || 'unknown';

    html += `
      <button
        type="button"
        class="trace-node ${type}"
        onclick="selectSparsityLayer(${index})"
        title="${cap.type} — ${pct(sp)} sparse"
      >
        <span class="trace-index">${index + 1}</span>

        <span class="trace-name">
          ${type}
        </span>

        <span class="trace-density">
          ${pct(density)} active
        </span>
      </button>
    `;

    if (index < captures.length - 1) {
      html += `<span class="trace-arrow">→</span>`;
    }
  });

  html += `
        </div>
      </div>

      <div id="sparsityDetail" class="sparsity-detail">
        <div class="sparsity-detail-empty">
          Select a layer above to inspect the inference.
        </div>
      </div>

    </div>
  `;

  container.innerHTML = html;

  selectSparsityLayer(0);
}


function selectSparsityLayer(index) {
  const data = BDH_DATA.inputs[currentInput];

  if (!data || !data.captures[index]) return;

  const cap = data.captures[index];
  const out = cap.output;

  document
    .querySelectorAll('.sparsity-layer')
    .forEach((el, i) => {
      el.classList.toggle('selected', i === index);
    });

  document
    .querySelectorAll('.trace-node')
    .forEach((el, i) => {
      el.classList.toggle('selected', i === index);
    });

  const detail = document.getElementById('sparsityDetail');

  if (!detail) return;

  const density = 1 - (out.sparsity || 0);

  detail.innerHTML = `
    <div class="detail-header">

      <div>
        <span class="detail-kicker">
          INFERENCE LAYER ${index + 1}
        </span>

        <h3>${cap.module_path}</h3>
      </div>

      <span class="detail-type">
        ${cap.type}
      </span>

    </div>

    <div class="detail-grid">

      <div>
        <span>Shape</span>
        <strong>${shapeStr(out.shape)}</strong>
      </div>

      <div>
        <span>Elements</span>
        <strong>${out.numel.toLocaleString()}</strong>
      </div>

      <div>
        <span>Sparsity</span>
        <strong class="highlight">${pct(out.sparsity)}</strong>
      </div>

      <div>
        <span>Active</span>
        <strong>${pct(density)}</strong>
      </div>

      <div>
        <span>Mean</span>
        <strong>${fmt(out.mean)}</strong>
      </div>

      <div>
        <span>Std</span>
        <strong>${fmt(out.std)}</strong>
      </div>

    </div>

    <div class="activation-preview">

      <div class="chart-heading">
        <span>Activation sample</span>
        <span class="chart-help">first ${out.sample.length} values</span>
      </div>

      <div class="activation-bars">

        ${out.sample.map((value, i) => {

          const max =
            Math.max(
              Math.abs(out.min || 0),
              Math.abs(out.max || 0),
              0.000001
            );

          const height =
            Math.max(
              4,
              Math.min(100, Math.abs(value) / max * 100)
            );

          const zero = value === 0;

          return `
            <div
              class="activation-bar ${zero ? 'zero' : ''}"
              style="height:${height}%"
              title="Index ${i}: ${fmt(value)}"
            ></div>
          `;
        }).join('')}

      </div>

    </div>
  `;
}
