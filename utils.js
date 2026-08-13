const fmt = n => typeof n === 'number' ? n.toExponential(2) : n;
const pct = n => (n * 100).toFixed(1) + '%';
const shapeStr = s => '[' + s.join(',') + ']';

function colorFor(type) {
  const map = { embed: '#58a6ff', ln: '#3fb950', attn: '#d29922', drop: '#f85149' };
  return map[type] || '#8b949e';
}

function groupCaptures(captures) {
  const blocks = [];
  let current = [];
  captures.forEach((c, i) => {
    if (c.module_path === 'embed' && i > 0) {
      blocks.push(current);
      current = [];
    }
    current.push(c);
  });
  if (current.length) blocks.push(current);
  return blocks;
}
