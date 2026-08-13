let currentInput = 0;
let currentSignal = 'pipeline';

function renderTree() {
  const data = BDH_DATA.inputs[currentInput];
  const blocks = groupCaptures(data.captures);
  let html = '';
  
  blocks.forEach((block, bi) => {
    html += `<div class="block-group expanded">
      <div class="block-header" onclick="toggleBlock(this)">
        <span class="block-toggle">▶</span> Block ${bi + 1}
      </div>
      <div class="block-items">`;
    
    block.forEach((cap, ci) => {
      const type = cap.module_path;
      const idx = data.captures.indexOf(cap);
      html += `
        <div class="layer-node" onclick="selectLayer(${idx})" data-idx="${idx}">
          <span class="node-dot ${type}"></span>
          <span class="node-name">${type}</span>
          <span class="node-shape">${shapeStr(cap.output.shape)}</span>
        </div>`;
    });
    
    html += '</div></div>';
  });
  
  document.getElementById('treeContent').innerHTML = html;
}

function toggleBlock(el) {
  el.parentElement.classList.toggle('expanded');
}

function selectLayer(idx) {
  document.querySelectorAll('.layer-node').forEach(n => n.classList.remove('active'));
  document.querySelector(`[data-idx="${idx}"]`)?.classList.add('active');
  
  const cap = BDH_DATA.inputs[currentInput].captures[idx];
  const out = cap.output;
  
  document.getElementById('statsContent').innerHTML = `
    <div class="stat-row"><span class="stat-label">Module</span><span class="stat-value">${cap.module_path}</span></div>
    <div class="stat-row"><span class="stat-label">Type</span><span class="stat-value">${cap.type}</span></div>
    <div class="stat-row"><span class="stat-label">Shape</span><span class="stat-value">${shapeStr(out.shape)}</span></div>
    <div class="stat-row"><span class="stat-label">Elements</span><span class="stat-value">${out.numel.toLocaleString()}</span></div>
    <div class="stat-row"><span class="stat-label">Sparsity</span><span class="stat-value highlight">${pct(out.sparsity)}</span></div>
    <div class="stat-row"><span class="stat-label">Mean</span><span class="stat-value">${fmt(out.mean)}</span></div>
    <div class="stat-row"><span class="stat-label">Std</span><span class="stat-value">${fmt(out.std)}</span></div>
    <div class="stat-row"><span class="stat-label">Min</span><span class="stat-value">${fmt(out.min)}</span></div>
    <div class="stat-row"><span class="stat-label">Max</span><span class="stat-value">${fmt(out.max)}</span></div>
    <div class="sparkline">${out.sample.map(v => 
      `<div class="spark-bar" style="height:${Math.max(2, Math.abs(v)/Math.max(Math.abs(out.min), out.max)*100)}%;opacity:${v===0?0.2:0.7}"></div>`
    ).join('')}</div>`;
}

function renderSignal() {
  if (currentSignal === 'pipeline') renderPipeline(currentInput);
  else if (currentSignal === 'sparsity') renderSparsity(currentInput);
  else if (currentSignal === 'activations') renderComparison();
  else if (currentSignal === 'attention') renderLocked('Attention heatmaps require [4,seq,seq] weight matrices. Hook the Q·Kᵀ intermediate tensor in the Attention module to unlock.');
  else if (currentSignal === 'rope') renderLocked('RoPE visualization requires inv_freq, cos_cached, and sin_cached tensors from the position embedding module.');
}

function renderLocked(msg) {
  document.getElementById('vizContainer').innerHTML = `
    <div class="locked-overlay">
      <div class="lock-icon">🔒</div>
      <h3>Signal Not Available</h3>
      <p>${msg}</p>
    </div>`;
}

// Event listeners
document.getElementById('inputSelect').addEventListener('change', e => {
  currentInput = parseInt(e.target.value);
  renderTree();
  renderSignal();
});

document.querySelectorAll('.signal-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.signal-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentSignal = btn.dataset.signal;
    renderSignal();
  });
});

// Init
renderTree();
renderSignal();
