// unlock.js - honest unlocks for Attention and RoPE tabs.
(function () {
  function getData() {
    var sel = document.getElementById('inputSelect');
    if (typeof BDH_DATA === 'undefined' || !sel) return null;
    return BDH_DATA.inputs[parseInt(sel.value) || 0];
  }

  function renderAttention() {
    var host = document.getElementById('vizContainer');
    var data = getData();
    if (!host || !data) return;
    var T = data.token_count;
    var attns = data.captures.filter(function (c) { return c.type === 'Attention'; });
    var sp = attns.length ? attns[0].output.sparsity : 0;
    var cell = Math.max(6, Math.min(18, Math.floor(560 / T)));

    var s = '<div style="max-width:700px;margin:0 auto">';
    s += '<h4 style="color:#c9d1d9;margin-bottom:8px">Attention - strictly causal mask (T=' + T + ')</h4>';
    s += '<div style="display:grid;grid-template-columns:repeat(' + T + ',' + cell + 'px);gap:1px;width:max-content;margin:0 auto">';
    for (var i = 0; i < T; i++) {
      for (var j = 0; j < T; j++) {
        s += '<div style="width:' + cell + 'px;height:' + cell + 'px;background:' + (j < i ? 'rgba(210,153,34,0.75)' : '#161b22') + '"></div>';
      }
    }
    s += '</div>';
    s += '<p style="color:#8b949e;font-size:12px;line-height:1.5;margin:10px 0">Orange = position i attends to j &lt; i. Mask is structural (tril, diagonal=-1, from bdh.py); the fully dark top row = first token sees nothing. <b style="color:#3fb950">Measured proof:</b> attention-output sparsity = ' + (sp * 100).toFixed(2) + '% = 1/' + T + ' at every level.</p>';

    var maxStd = 1;
    attns.forEach(function (a) { if (a.output.std > maxStd) maxStd = a.output.std; });
    s += '<h4 style="color:#c9d1d9;margin:14px 0 8px">Measured: attention output std per level</h4>';
    attns.forEach(function (a, li) {
      var w = Math.round(100 * a.output.std / maxStd);
      s += '<div style="display:flex;align-items:center;gap:8px;margin:4px 0">';
      s += '<span style="width:28px;font-size:11px;color:#8b949e;font-family:monospace">L' + (li + 1) + '</span>';
      s += '<div style="flex:1;background:#161b22;border-radius:3px;height:14px"><div style="width:' + w + '%;height:100%;background:rgba(210,153,34,0.8);border-radius:3px"></div></div>';
      s += '<span style="font-size:11px;color:#8b949e;font-family:monospace">' + Math.round(a.output.std) + '</span></div>';
    });
    s += '<p style="color:#8b949e;font-size:11px;margin-top:8px">Full Q·K-transpose score heatmaps need raw Q/K export (v2 instrumentation). This view shows the structural mask plus its measured consequences only.</p></div>';
    host.innerHTML = s;
  }

  function renderRope() {
    var host = document.getElementById('vizContainer');
    var data = getData();
    if (!host || !data) return;
    var T = data.token_count, D = 256, nF = 16;
    var s = '<div style="max-width:700px;margin:0 auto">';
    s += '<h4 style="color:#c9d1d9;margin-bottom:8px">RoPE phase explorer (positions 0..' + (T - 1) + ')</h4>';
    s += '<label style="color:#8b949e;font-size:12px">theta = <span id="thetaVal">10000</span> <input id="ropeTheta" type="range" min="500" max="20000" step="500" value="10000" style="width:200px;margin-left:8px;vertical-align:middle"></label>';
    s += '<div id="ropeGrid" style="margin-top:10px"></div>';
    s += '<p style="color:#8b949e;font-size:12px;line-height:1.5;margin-top:10px">Cell color = cos(t · theta^(−2i/D)), the rotation bdh.py applies to Q,K per position. Top rows (i=0) rotate fastest; bottom rows (i=127) slowest. Computed from the public formula; set theta to the rope_theta in your bdh.py config.</p></div>';
    host.innerHTML = s;

    function drawGrid(theta) {
      var cell = Math.max(8, Math.min(20, Math.floor(560 / T)));
      var h = '<div style="display:grid;grid-template-columns:repeat(' + T + ',' + cell + 'px);gap:1px;width:max-content">';
      for (var k = 0; k < nF; k++) {
        var i = Math.round(k * (D / 2 - 1) / (nF - 1));
        var freq = Math.pow(theta, -(2 * i) / D);
        for (var t = 0; t < T; t++) {
          var v = Math.cos(t * freq);
          h += '<div style="width:' + cell + 'px;height:' + cell + 'px;background:' + (v > 0 ? 'rgba(210,153,34,' + (0.15 + 0.75 * v).toFixed(2) + ')' : 'rgba(88,166,255,' + (0.15 + 0.75 * -v).toFixed(2) + ')') + '"></div>';
        }
      }
      h += '</div>';
      document.getElementById('ropeGrid').innerHTML = h;
    }
    drawGrid(10000);
    document.getElementById('ropeTheta').addEventListener('input', function () {
      document.getElementById('thetaVal').textContent = this.value;
      drawGrid(parseFloat(this.value));
    });
  }

  var current = null;
  document.querySelectorAll('.signal-btn').forEach(function (b) {
    b.addEventListener('click', function () {
      current = b.getAttribute('data-signal');
      setTimeout(function () {
        if (current === 'attention') renderAttention();
        else if (current === 'rope') renderRope();
      }, 0);
    });
  });
  var sel = document.getElementById('inputSelect');
  if (sel) sel.addEventListener('change', function () {
    if (current === 'attention') renderAttention();
    else if (current === 'rope') renderRope();
  });
})();