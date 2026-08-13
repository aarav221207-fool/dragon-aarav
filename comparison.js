function renderComparison() {
  const container = document.getElementById('vizContainer');
  let html = '<div class="comparison-grid">';
  
  BDH_DATA.inputs.forEach(inp => {
    const blocks = groupCaptures(inp.captures);
    const avgSparsity = blocks.flat().reduce((a,c) => a + c.output.sparsity, 0) / inp.captures.length;
    const dropSparsity = inp.captures.find(c => c.module_path === 'drop')?.output.sparsity || 0;
    
    html += `
      <div class="comparison-card">
        <h4>"${inp.text.substring(0,20)}..."</h4>
        <div class="comparison-metric"><span>Tokens</span><span>${inp.token_count}</span></div>
        <div class="comparison-metric"><span>Layers</span><span>${inp.captures.length}</span></div>
        <div class="comparison-metric"><span>Avg Sparsity</span><span>${pct(avgSparsity)}</span></div>
        <div class="comparison-metric"><span>Dropout Sparsity</span><span class="highlight">${pct(dropSparsity)}</span></div>
      </div>`;
  });
  
  html += '</div>';
  container.innerHTML = html;
}
