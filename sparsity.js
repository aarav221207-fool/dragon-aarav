function renderSparsity(inputIdx) {
  const container = document.getElementById('vizContainer');
  const data = BDH_DATA.inputs[inputIdx];
  const blocks = groupCaptures(data.captures);
  
  let html = '<div class="sparsity-chart">';
  blocks.forEach((block, bi) => {
    block.forEach(cap => {
      const sp = cap.output.sparsity;
      const type = cap.module_path;
      html += `
        <div class="sparsity-bar-group">
          <div class="sparsity-label">
            <span>Block ${bi+1} ${type}</span>
            <span>${pct(sp)}</span>
          </div>
          <div class="sparsity-track">
            <div class="sparsity-fill ${type}" style="width:${sp*100}%"></div>
          </div>
        </div>`;
    });
  });
  html += '</div>';
  container.innerHTML = html;
}
