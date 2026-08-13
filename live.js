
(function () {
  var D = 256, NH = 4, N = 8192, LEVELS = 6, THETA = 65536, MAXT = 48;
  var W = null;

  function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; var t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function gaussF(rng) { return function () { var u = 0, v = 0; while (u === 0) u = rng(); while (v === 0) v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }; }

  function initWeights() {
    var rng = mulberry32(20260813), g = gaussF(rng), i;
    W = {};
    W.embed = new Float32Array(256 * D); for (i = 0; i < W.embed.length; i++) W.embed[i] = g() * 0.02;
    W.enc = new Float32Array(NH * D * N); for (i = 0; i < W.enc.length; i++) W.enc[i] = g() * 0.02;
    W.encV = new Float32Array(NH * D * N); for (i = 0; i < W.encV.length; i++) W.encV[i] = g() * 0.02;
    W.dec = new Float32Array(NH * N * D); for (i = 0; i < W.dec.length; i++) W.dec[i] = g() * 0.02;
    W.freqs = new Float32Array(N);
    for (i = 0; i < N; i++) { var q = Math.floor(i / 2) * 2; W.freqs[i] = 1 / Math.pow(THETA, q / N) / (2 * Math.PI); }
  }

  function lnRows(a, rows, cols) {
    for (var r = 0; r < rows; r++) {
      var o = r * cols, m = 0, i; for (i = 0; i < cols; i++) m += a[o + i]; m /= cols;
      var v = 0; for (i = 0; i < cols; i++) { var d = a[o + i] - m; v += d * d; } v /= cols;
      var s = 1 / Math.sqrt(v + 1e-5);
      for (i = 0; i < cols; i++) a[o + i] = (a[o + i] - m) * s;
    }
  }

  function statsOf(arr, shape, name) {
    var n = arr.length, m = 0, i, mn = Infinity, mx = -Infinity, z = 0;
    for (i = 0; i < n; i++) { var v = arr[i]; m += v; if (v < mn) mn = v; if (v > mx) mx = v; if (v === 0) z++; }
    m /= n; var va = 0; for (i = 0; i < n; i++) { var d = arr[i] - m; va += d * d; } va /= n;
    return { name: name, shape: shape, numel: n, mean: m, std: Math.sqrt(va), min: mn, max: mx, sparsity: z / n, sample: Array.prototype.slice.call(arr, 0, 10) };
  }

  function run(text) {
    if (!W) initWeights();
    var toks = [], ci; for (ci = 0; ci < text.length && ci < MAXT; ci++) toks.push(text.charCodeAt(ci) & 255);
    var T = toks.length, t, d, h, n;
    var captures = [];

    var x = new Float32Array(T * D);
    for (t = 0; t < T; t++) for (d = 0; d < D; d++) x[t * D + d] = W.embed[toks[t] * D + d];
    captures.push({ module_path: 'embed', type: 'Embedding', output: statsOf(x, [1, T, D], 'embed_out'), input: { name: 'embed_in', shape: [1, T], numel: T, mean: 0, std: 0, min: 0, max: 0, sparsity: 0, sample: toks.slice(0, 10) } });
    lnRows(x, T, D);
    captures.push({ module_path: 'ln', type: 'LayerNorm', output: statsOf(x, [1, 1, T, D], 'ln_out') });

    for (var L = 0; L < LEVELS; L++) {
      var xs = new Float32Array(NH * T * N);
      for (h = 0; h < NH; h++) {
        var b1 = h * D * N;
        for (t = 0; t < T; t++) {
          var acc = new Float32Array(N);
          for (d = 0; d < D; d++) { var xd = x[t * D + d]; if (xd === 0) continue; var o1 = b1 + d * N; for (n = 0; n < N; n++) acc[n] += xd * W.enc[o1 + n]; }
          var xo = (h * T + t) * N;
          for (n = 0; n < N; n++) xs[xo + n] = acc[n] > 0 ? acc[n] : 0;
        }
      }

      var qr = new Float32Array(xs.length);
      for (h = 0; h < NH; h++) for (t = 0; t < T; t++) {
        var off = (h * T + t) * N;
        for (n = 0; n < N; n++) {
          var ph = ((t * W.freqs[n]) % 1) * 2 * Math.PI;
          var vr = (n % 2 === 0) ? -xs[off + n + 1] : xs[off + n - 1];
          qr[off + n] = xs[off + n] * Math.cos(ph) + vr * Math.sin(ph);
        }
      }

      var ykv = new Float32Array(NH * T * D);
      for (h = 0; h < NH; h++) for (var i2 = 1; i2 < T; i2++) {
        var qo = (h * T + i2) * N, sc = new Float32Array(i2), j;
        for (j = 0; j < i2; j++) { var ko = (h * T + j) * N, dot = 0; for (n = 0; n < N; n++) dot += qr[qo + n] * qr[ko + n]; sc[j] = dot; }
        var yo = (h * T + i2) * D;
        for (j = 0; j < i2; j++) { var sv = sc[j]; if (sv === 0) continue; var vo = j * D; for (d = 0; d < D; d++) ykv[yo + d] += sv * x[vo + d]; }
      }
      captures.push({ module_path: 'attn', type: 'Attention', output: statsOf(ykv, [1, NH, T, D], 'attn_out') });
      lnRows(ykv, NH * T, D);
      captures.push({ module_path: 'ln', type: 'LayerNorm', output: statsOf(ykv, [1, NH, T, D], 'ln_out') });

      var xy = new Float32Array(NH * T * N);
      for (h = 0; h < NH; h++) {
        var b2 = h * D * N;
        for (t = 0; t < T; t++) {
          var acc2 = new Float32Array(N);
          for (d = 0; d < D; d++) { var yd = ykv[(h * T + t) * D + d]; if (yd === 0) continue; var o2 = b2 + d * N; for (n = 0; n < N; n++) acc2[n] += yd * W.encV[o2 + n]; }
          var xo2 = (h * T + t) * N;
          for (n = 0; n < N; n++) xy[xo2 + n] = (acc2[n] > 0 ? acc2[n] : 0) * xs[xo2 + n];
        }
      }
      captures.push({ module_path: 'drop', type: 'Dropout', output: statsOf(xy, [1, NH, T, N], 'drop_out') });

      var ymlp = new Float32Array(T * D);
      for (t = 0; t < T; t++) for (h = 0; h < NH; h++) {
        var mo = h * N, xyo = (h * T + t) * N;
        for (n = 0; n < N; n++) { var xv = xy[xyo + n]; if (xv === 0) continue; var doff = (mo + n) * D; for (d = 0; d < D; d++) ymlp[t * D + d] += xv * W.dec[doff + d]; }
      }
      lnRows(ymlp, T, D);
      captures.push({ module_path: 'ln', type: 'LayerNorm', output: statsOf(ymlp, [1, 1, T, D], 'ln_out') });
      for (var k = 0; k < x.length; k++) x[k] += ymlp[k];
      lnRows(x, T, D);
      captures.push({ module_path: 'ln', type: 'LayerNorm', output: statsOf(x, [1, 1, T, D], 'ln_out') });
    }
    return { text: text, tokens: toks, token_count: T, captures: captures, final_output: statsOf(x, [1, T, D], 'final') };
  }

  var probe = document.getElementById('probeText');
  var sel = document.getElementById('inputSelect');
  var holder = document.querySelector('.probe-input') || document.querySelector('.probe-bar');
  if (!probe || !sel || !holder || typeof BDH_DATA === 'undefined') return;
  probe.removeAttribute('readonly');

  var note = document.createElement('p');
  note.style.cssText = 'font:400 10px "JetBrains Mono";color:#8b98ab;margin-top:6px';
  note.textContent = 'Live sandbox: architecture-faithful JS mirror of bdh.py with fresh deterministic init statistically equivalent, not byte-identical to captured presets.';
  document.querySelector('.probe-bar').appendChild(note);

  var btn = document.createElement('button');
  btn.type = 'button'; btn.className = 'preset-btn'; btn.textContent = '▶ Run inference';
  btn.addEventListener('click', function () {
    var text = (probe.value || '').trim(); if (!text) return;
    btn.textContent = '… computing'; btn.disabled = true;
    setTimeout(function () {
      try {
        var res = run(text);
        BDH_DATA.inputs.push(res);
        var idx = BDH_DATA.inputs.length - 1;
        var opt = document.createElement('option'); opt.value = String(idx); opt.textContent = '"live: ' + text + '"';
        sel.appendChild(opt); sel.value = String(idx); sel.dispatchEvent(new Event('change'));
        var chip = document.createElement('button'); chip.type = 'button'; chip.className = 'preset-btn'; chip.textContent = '"live: ' + text + '"';
        chip.addEventListener('click', function () { sel.value = String(idx); sel.dispatchEvent(new Event('change')); });
        document.getElementById('presetRow').appendChild(chip);
      } catch (e) { btn.textContent = 'error: ' + e.message; }
      btn.textContent = '▶ Run live (JS mirror)'; btn.disabled = false;
    }, 30);
  });
  holder.appendChild(btn);
})();
