function renderComparison() {
  const container = document.getElementById('vizContainer');
  const data = BDH_DATA.inputs[currentInput];

  if (!container || !data) return;

  const captures = data.captures || [];
  const cap = captures[selectedLayer] || captures[0];

  if (!cap) return;

  const out = cap.output;
  const sample = out.sample || [];

  const max =
    Math.max(
      Math.abs(out.min || 0),
      Math.abs(out.max || 0),
      0.000001
    );

  container.innerHTML = `
    <div class="comparison-view">

      <div class="view-heading">
        <div>
          <span class="eyebrow">03 · ACTIVATIONS</span>
          <h2>Activation Inspector</h2>
          <p>Inspect tensor statistics and the captured activation sample.</p>
        </div>

        <select
          class="layer-select"
          onchange="selectLayer(Number(this.value)); renderComparison()"
        >
          ${captures.map((item, index) => `
            <option value="${index}" ${index === selectedLayer ? 'selected' : ''}>
              ${index + 1}. ${item.module_path}
            </option>
          `).join('')}
        </select>
      </div>

      <div class="activation-header">
        <div>
          <span class="eyebrow">SELECTED TENSOR</span>
          <h3>${cap.module_path.toUpperCase()}</h3>
          <p>${cap.type}</p>
        </div>

        <code>${shapeStr(out.shape)}</code>
      </div>

      <div class="activation-stat-grid">

        <div>
          <span>MEAN</span>
          <strong>${fmt(out.mean)}</strong>
        </div>

        <div>
          <span>STD</span>
          <strong>${fmt(out.std)}</strong>
        </div>

        <div>
          <span>MIN</span>
          <strong>${fmt(out.min)}</strong>
        </div>

        <div>
          <span>MAX</span>
          <strong>${fmt(out.max)}</strong>
        </div>

        <div>
          <span>SPARSITY</span>
          <strong>${pct(out.sparsity)}</strong>
        </div>

        <div>
          <span>ELEMENTS</span>
          <strong>${Number(out.numel).toLocaleString()}</strong>
        </div>

      </div>

      <div class="chart-section">

        <div class="chart-heading">
          <span>ACTIVATION SAMPLE</span>
          <span class="chart-help">hover for exact value</span>
        </div>

        <div class="activation-visual">
          ${sample.map((value, index) => {

            const height =
              Math.max(
                3,
                Math.min(100, Math.abs(value) / max * 100)
              );

            return `
              <div
                class="activation-column ${value === 0 ? 'zero' : ''}"
                style="height:${height}%"
                title="index ${index}: ${Number(value).toFixed(6)}"
              ></div>
            `;
          }).join('')}
        </div>

      </div>

      <div class="chart-section">

        <div class="chart-heading">
          <span>RAW VALUES</span>
          <span class="chart-help">${sample.length} values</span>
        </div>

        <div class="raw-values">
          ${sample.map((value, index) => `
            <div>
              <span>${index}</span>
              <strong>${Number(value).toFixed(5)}</strong>
            </div>
          `).join('')}
        </div>

      </div>

    </div>
  `;
}
