function renderSparsity(inputIdx) {
  const container = document.getElementById('vizContainer');
  const data = BDH_DATA.inputs[inputIdx];

  if (!container || !data) return;

  const captures = data.captures.filter(c => c && c.output);

  if (!captures.length) {
    container.innerHTML = `
      <div class="visual-empty">
        <h2>No sparsity data</h2>
        <p>No captured tensors are available for this inference.</p>
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

      <div class="view-heading">
        <div>
          <span class="eyebrow">02 · SPARSITY</span>
          <h2>Sparsity Analysis</h2>
          <p>Measure inactive computation across the current inference path.</p>
        </div>
      </div>

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
          <span>LAYER SPARSITY</span>
          <span class="chart-help">click any layer</span>
        </div>

        <div class="sparsity-chart">
  `;

  captures.forEach((cap, index) => {
    const sp = cap.output.sparsity || 0;
    const density = 1 - sp;
    const type = cap.module_path || 'unknown';

    html += `
      <button
        class="sparsity-layer ${selectedLayer === index ? 'selected' : ''}"
        type="button"
        onclick="selectSparsityLayer(${index})"
      >

        <div class="sparsity-layer-header">
          <span class="sparsity-layer-name">
            ${String(index + 1).padStart(2, '0')} · ${type}
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
          <span>${Number(cap.output.numel).toLocaleString()} elements · ${pct(density)} active</span>
        </div>

      </button>
    `;
  });

  html += `
        </div>
      </div>

      <div class="chart-section">
        <div class="chart-heading">
          <span>INFERENCE DENSITY TRACE</span>
          <span class="chart-help">active computation</span>
        </div>

        <div class="inference-trace">
  `;

  captures.forEach((cap, index) => {
    const sp = cap.output.sparsity || 0;
    const density = 1 - sp;

    html += `
      <button
        type="button"
        class="trace-node ${cap.module_path} ${selectedLayer === index ? 'selected' : ''}"
        onclick="selectSparsityLayer(${index})"
      >
        <span class="trace-index">${index + 1}</span>
        <span class="trace-name">${cap.module_path}</span>
        <span class="trace-density">${pct(density)} active</span>
      </button>
    `;

    if (index < captures.length - 1) {
      html += `<span class="trace-arrow">→</span>`;
    }
  });

  html += `
        </div>
      </div>

      <div id="sparsityDetail" class="sparsity-detail"></div>

    </div>
  `;

  container.innerHTML = html;

  selectSparsityLayer(
    selectedLayer < captures.length ? selectedLayer : 0
  );
}

function selectSparsityLayer(index) {
  const data = BDH_DATA.inputs[currentInput];

  if (!data || !data.captures[index]) return;

  selectedLayer = index;

  const cap = data.captures[index];
  const out = cap.output;
  const density = 1 - (out.sparsity || 0);
  const detail = document.getElementById('sparsityDetail');

  document.querySelectorAll('.sparsity-layer').forEach((el, i) => {
    el.classList.toggle('selected', i === index);
  });

  document.querySelectorAll('.trace-node').forEach((el, i) => {
    el.classList.toggle('selected', i === index);
  });

  document.querySelectorAll('.layer-node').forEach((el, i) => {
    el.classList.toggle('active', i === index);
  });

  if (!detail) return;

  detail.innerHTML = `
    <div class="detail-header">

      <div>
        <span class="detail-kicker">INFERENCE LAYER ${index + 1}</span>
        <h3>${cap.module_path}</h3>
      </div>

      <span class="detail-type">${cap.type}</span>

    </div>

    <div class="detail-grid">

      <div>
        <span>Shape</span>
        <strong>${shapeStr(out.shape)}</strong>
      </div>

      <div>
        <span>Elements</span>
        <strong>${Number(out.numel).toLocaleString()}</strong>
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

      <div>
        <span>Min</span>
        <strong>${fmt(out.min)}</strong>
      </div>

      <div>
        <span>Max</span>
        <strong>${fmt(out.max)}</strong>
      </div>

    </div>

    <div class="activation-preview">

      <div class="chart-heading">
        <span>ACTIVATION SAMPLE</span>
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
