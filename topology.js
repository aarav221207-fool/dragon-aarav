
(function () {
  function draw() {
    var host = document.getElementById('vizContainer');
    var sel = document.getElementById('inputSelect');
    if (!host || !sel || typeof BDH_DATA === 'undefined') return;
    var data = BDH_DATA.inputs[parseInt(sel.value) || 0];
    var levels = [];
    data.captures.forEach(function (c) {
      if (c.type === 'Attention') levels.push({ attn: c, drop: null });
      else if (c.type === 'Dropout' && levels.length) levels[levels.length - 1].drop = c;
    });

    var nodes = [{ label: 'in', sp: 0, amp: 0, loop: false }];
    levels.forEach(function (L, i) {
      nodes.push({ label: 'L' + (i + 1), sp: L.drop ? L.drop.output.sparsity : 0, amp: L.attn.output.std, loop: true });
    });
    nodes.push({ label: 'out', sp: 0, amp: 0, loop: false });

    var maxAmp = 1;
    nodes.forEach(function (n) { if (n.amp > maxAmp) maxAmp = n.amp; });

    var step = 84, x0 = 50, y = 110;
    var W = x0 * 2 + step * (nodes.length - 1);
    var s = '<div style="width:100%;overflow-x:auto"><svg width="' + W + '" height="200" viewBox="0 0 ' + W + ' 200" style="display:block;margin:0 auto">';

    for (var i = 0; i < nodes.length - 1; i++) {
      s += '<line x1="' + (x0 + i * step) + '" y1="' + y + '" x2="' + (x0 + (i + 1) * step) + '" y2="' + y + '" stroke="#30363d" stroke-width="2"/>';
    }

    nodes.forEach(function (n, i) {
      var x = x0 + i * step;
      var r = 12 + n.sp * 16;
      var heat = n.amp / maxAmp;
      if (n.loop) {
        s += '<path d="M ' + (x - 12) + ' ' + (y - r) + ' q 12 -30 24 0" fill="none" stroke="#58a6ff" stroke-width="1.5" stroke-dasharray="4 3"/>';
      }
      s += '<circle cx="' + x + '" cy="' + y + '" r="' + r + '" fill="rgba(210,153,34,' + (0.15 + 0.65 * heat).toFixed(2) + ')" stroke="#d29922" stroke-width="1.5"/>';
      s += '<text x="' + x + '" y="' + (y + 4) + '" text-anchor="middle" fill="#c9d1d9" font-size="11" font-family="monospace">' + n.label + '</text>';
      if (n.sp) {
        s += '<text x="' + x + '" y="' + (y + r + 16) + '" text-anchor="middle" fill="#8b949e" font-size="10">' + Math.round(n.sp * 100) + '%</text>';
      }
    });

    s += '</svg></div>';
    s += '<p style="color:#8b949e;font-size:12px;max-width:560px;margin:12px auto 0;line-height:1.5">Structural topology preview: node size = gated sparsity per level, orange heat = attention std (amplification), dashed loop = residual connection (x = ln(x+y)). Full TDA on activation clouds = bdh_tda.py (v2).</p>';
    host.innerHTML = s;
  }

  var bar = document.querySelector('.bottom-bar');
  if (bar && !document.getElementById('topoBtn')) {
    var b = document.createElement('button');
    b.type = 'button';
    b.className = 'signal-btn';
    b.id = 'topoBtn';
    b.textContent = 'Topology';
    b.addEventListener('click', function () {
      document.querySelectorAll('.signal-btn').forEach(function (x) { x.classList.remove('active'); });
      b.classList.add('active');
      draw();
    });
    bar.appendChild(b);

    var sel = document.getElementById('inputSelect');
    if (sel) {
      sel.addEventListener('change', function () {
        if (b.classList.contains('active')) draw();
      });
    }
  }
})();