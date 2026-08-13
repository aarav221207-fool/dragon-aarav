function renderPipeline(inputIdx) {
  const container = document.getElementById('vizContainer');
  const data = BDH_DATA.inputs[inputIdx];
  const blocks = groupCaptures(data.captures);
  
  let html = '<div class="pipeline-flow">';
  blocks.forEach((block, bi) => {
    html += `<div class="block-label">Block ${bi + 1}</div>`;
    block.forEach((cap, ci) => {
      const type = cap.module_path;
      const shape = shapeStr(cap.output.shape);
      html += `<div class="flow-node ${type}">${type}<br><small>${shape}</small></div>`;
      if (ci < block.length - 1) html += '<div class="flow-connector"></div>';
    });
    if (bi < blocks.length - 1) html += '<div class="flow-connector dashed"></div>';
  });
  html += '</div>';
  container.innerHTML = html;
}
