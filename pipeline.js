function renderPipeline(inputIdx) {
  const container = document.getElementById('vizContainer');
  const data = BDH_DATA.inputs[inputIdx];

  if (!container || !data) return;

  const captures = data.captures || [];

  const avgSparsity = captures.length
    ? captures.reduce((sum, cap) => sum + (cap.output.sparsity || 0), 0) / captures.length
    : 0;

  let html = `
    <div class="pipeline-view">

      <div class="view-heading">
        <div>
          <span class="eyebrow">01 · PATHWAY</span>
          <h2>Inference Pipeline</h2>
          <p>Follow the tensor through every captured stage of the Dragon Hatchling pathway.</p>
        </div>

        <div class="pipeline-health">
          <span>AVG SPARSITY</span>
          <strong>${pct(avgSparsity)}</strong>
        </div>
      </div>

      <div class="pipeline-flow">
  `;

  captures.forEach((cap, index) => {
    const output = cap.output;
    const density = 1 - (output.sparsity || 0);

    html += `
      <button
        type="button"
        class="flow-node ${cap.module_path} ${index === selectedLayer ? 'selected' : ''}"
        onclick="selectLayer(${index})"
      >
        <div class="flow-number">${String(index + 1).padStart(2, '0')}</div>

        <div class="flow-content">
          <div class="flow-top">
            <strong>${cap.module_path.toUpperCase()}</strong>
            <span>${cap.type}</span>
          </div>

          <code>${shapeStr(output.shape)}</code>

          <div class="flow-metrics">
            <span>ACTIVE <b>${pct(density)}</b></span>
            <span>SPARSE <b>${pct(output.sparsity || 0)}</b></span>
            <span>N <b>${Number(output.numel || 0).toLocaleString()}</b></span>
          </div>

          <div class="flow-meter">
            <i style="width:${density * 100}%"></i>
          </div>
        </div>
      </button>
    `;

    if (index < captures.length - 1) {
      html += `<div class="flow-connector">↓</div>`;
    }
  });

  html += `
      </div>

      <div class="pipeline-footer">
        <div>
          <span>INPUT</span>
          <strong>${data.token_count} TOKENS</strong>
        </div>
        <div>
          <span>FINAL OUTPUT</span>
          <strong>${shapeStr(data.final_output.shape)}</strong>
        </div>
        <div>
          <span>OUTPUT STD</span>
          <strong>${fmt(data.final_output.std)}</strong>
        </div>
      </div>

    </div>
  `;

  container.innerHTML = html;
}
