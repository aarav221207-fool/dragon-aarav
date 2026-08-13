(function () {
  var D = 256;
  var NH = 4;
  var N = 8192;
  var LEVELS = 6;
  var THETA = 65536;
  var MAXT = 48;

  var W = null;

  function mulberry32(a) {
    return function () {
      a |= 0;
      a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  function gaussF(rng) {
    return function () {
      var u = 0;
      var v = 0;

      while (u === 0) u = rng();
      while (v === 0) v = rng();

      return Math.sqrt(-2 * Math.log(u)) *
        Math.cos(2 * Math.PI * v);
    };
  }

  function initWeights() {
    var rng = mulberry32(20260813);
    var g = gaussF(rng);
    var i;

    W = {};

    W.embed = new Float32Array(256 * D);

    for (i = 0; i < W.embed.length; i++) {
      W.embed[i] = g() * 0.02;
    }

    W.enc = new Float32Array(NH * D * N);

    for (i = 0; i < W.enc.length; i++) {
      W.enc[i] = g() * 0.02;
    }

    W.encV = new Float32Array(NH * D * N);

    for (i = 0; i < W.encV.length; i++) {
      W.encV[i] = g() * 0.02;
    }

    W.dec = new Float32Array(NH * N * D);

    for (i = 0; i < W.dec.length; i++) {
      W.dec[i] = g() * 0.02;
    }

    W.freqs = new Float32Array(N);

    for (i = 0; i < N; i++) {
      var q = Math.floor(i / 2) * 2;

      W.freqs[i] =
        1 /
        Math.pow(THETA, q / N) /
        (2 * Math.PI);
    }
  }

  function lnRows(a, rows, cols) {
    for (var r = 0; r < rows; r++) {
      var o = r * cols;
      var m = 0;

      for (var i = 0; i < cols; i++) {
        m += a[o + i];
      }

      m /= cols;

      var v = 0;

      for (var j = 0; j < cols; j++) {
        var d = a[o + j] - m;
        v += d * d;
      }

      v /= cols;

      var s = 1 / Math.sqrt(v + 1e-5);

      for (var k = 0; k < cols; k++) {
        a[o + k] = (a[o + k] - m) * s;
      }
    }
  }

  function statsOf(arr, shape, name) {
    var n = arr.length;

    if (!n) {
      return {
        name: name,
        shape: shape,
        numel: 0,
        mean: 0,
        std: 0,
        min: 0,
        max: 0,
        sparsity: 0,
        sample: []
      };
    }

    var m = 0;
    var mn = Infinity;
    var mx = -Infinity;
    var z = 0;

    for (var i = 0; i < n; i++) {
      var value = arr[i];

      m += value;

      if (value < mn) mn = value;
      if (value > mx) mx = value;
      if (value === 0) z++;
    }

    m /= n;

    var variance = 0;

    for (var j = 0; j < n; j++) {
      var diff = arr[j] - m;
      variance += diff * diff;
    }

    variance /= n;

    return {
      name: name,
      shape: shape,
      numel: n,
      mean: m,
      std: Math.sqrt(variance),
      min: mn,
      max: mx,
      sparsity: z / n,
      sample: Array.prototype.slice.call(arr, 0, 32)
    };
  }

  function run(text) {
    if (!W) {
      initWeights();
    }

    var toks = [];

    for (
      var ci = 0;
      ci < text.length && ci < MAXT;
      ci++
    ) {
      toks.push(text.charCodeAt(ci) & 255);
    }

    if (!toks.length) {
      toks.push(32);
    }

    var T = toks.length;

    var t;
    var d;
    var h;
    var n;

    var captures = [];

    var x = new Float32Array(T * D);

    for (t = 0; t < T; t++) {
      for (d = 0; d < D; d++) {
        x[t * D + d] =
          W.embed[toks[t] * D + d];
      }
    }

    captures.push({
      module_path: 'embed',
      type: 'Embedding',
      output: statsOf(
        x,
        [1, T, D],
        'embed_out'
      ),
      input: {
        name: 'embed_in',
        shape: [1, T],
        numel: T,
        mean: 0,
        std: 0,
        min: 0,
        max: 0,
        sparsity: 0,
        sample: toks.slice(0, 32)
      }
    });

    lnRows(x, T, D);

    captures.push({
      module_path: 'ln',
      type: 'LayerNorm',
      output: statsOf(
        x,
        [1, 1, T, D],
        'ln_out'
      )
    });

    var attentionMaps = [];

    for (var L = 0; L < LEVELS; L++) {

      var xs =
        new Float32Array(NH * T * N);

      for (h = 0; h < NH; h++) {

        var b1 = h * D * N;

        for (t = 0; t < T; t++) {

          var acc = new Float32Array(N);

          for (d = 0; d < D; d++) {

            var xd =
              x[t * D + d];

            if (xd === 0) continue;

            var o1 =
              b1 + d * N;

            for (n = 0; n < N; n++) {
              acc[n] +=
                xd * W.enc[o1 + n];
            }
          }

          var xo =
            (h * T + t) * N;

          for (n = 0; n < N; n++) {
            xs[xo + n] =
              acc[n] > 0
                ? acc[n]
                : 0;
          }
        }
      }

      var qr =
        new Float32Array(xs.length);

      for (h = 0; h < NH; h++) {

        for (t = 0; t < T; t++) {

          var off =
            (h * T + t) * N;

          for (n = 0; n < N; n++) {

            var ph =
              ((t * W.freqs[n]) % 1) *
              2 *
              Math.PI;

            var vr =
              (n % 2 === 0)
                ? -xs[off + n + 1]
                : xs[off + n - 1];

            qr[off + n] =
              xs[off + n] *
                Math.cos(ph) +
              vr *
                Math.sin(ph);
          }
        }
      }

      var ykv =
        new Float32Array(NH * T * D);

      var layerAttention = [];

      for (h = 0; h < NH; h++) {

        var headMatrix = [];

        for (t = 0; t < T; t++) {

          var row =
            new Array(T).fill(0);

          if (t > 0) {

            var qo =
              (h * T + t) * N;

            var scores =
              new Float32Array(t);

            var scoreMax =
              -Infinity;

            for (var j = 0; j < t; j++) {

              var ko =
                (h * T + j) * N;

              var dot = 0;

              for (n = 0; n < N; n++) {
                dot +=
                  qr[qo + n] *
                  qr[ko + n];
              }

              scores[j] = dot;

              if (dot > scoreMax) {
                scoreMax = dot;
              }
            }

            var denominator = 0;

            for (j = 0; j < t; j++) {
              var expScore =
                Math.exp(
                  Math.max(
                    -30,
                    Math.min(
                      30,
                      scores[j] - scoreMax
                    )
                  )
                );

              scores[j] = expScore;
              denominator += expScore;
            }

            if (denominator === 0) {
              denominator = 1;
            }

            var yo =
              (h * T + t) * D;

            for (j = 0; j < t; j++) {

              var weight =
                scores[j] /
                denominator;

              row[j] = weight;

              var vo =
                j * D;

              for (d = 0; d < D; d++) {

                ykv[yo + d] +=
                  weight *
                  x[vo + d];
              }
            }
          }

          headMatrix.push(row);
        }

        layerAttention.push(headMatrix);
      }

      attentionMaps.push(layerAttention);

      captures.push({
        module_path: 'attn',
        type: 'Attention',
        output: statsOf(
          ykv,
          [1, NH, T, D],
          'attn_out'
        )
      });

      lnRows(
        ykv,
        NH * T,
        D
      );

      captures.push({
        module_path: 'ln',
        type: 'LayerNorm',
        output: statsOf(
          ykv,
          [1, NH, T, D],
          'ln_out'
        )
      });

      var xy =
        new Float32Array(
          NH * T * N
        );

      for (h = 0; h < NH; h++) {

        var b2 =
          h * D * N;

        for (t = 0; t < T; t++) {

          var acc2 =
            new Float32Array(N);

          for (d = 0; d < D; d++) {

            var yd =
              ykv[
                (h * T + t) * D + d
              ];

            if (yd === 0) continue;

            var o2 =
              b2 + d * N;

            for (n = 0; n < N; n++) {
              acc2[n] +=
                yd *
                W.encV[o2 + n];
            }
          }

          var xo2 =
            (h * T + t) * N;

          for (n = 0; n < N; n++) {

            xy[xo2 + n] =
              (
                acc2[n] > 0
                  ? acc2[n]
                  : 0
              ) *
              xs[xo2 + n];
          }
        }
      }

      captures.push({
        module_path: 'drop',
        type: 'Dropout',
        output: statsOf(
          xy,
          [1, NH, T, N],
          'drop_out'
        )
      });

      var ymlp =
        new Float32Array(T * D);

      for (t = 0; t < T; t++) {

        for (h = 0; h < NH; h++) {

          var mo =
            h * N;

          var xyo =
            (h * T + t) * N;

          for (n = 0; n < N; n++) {

            var xv =
              xy[xyo + n];

            if (xv === 0) continue;

            var doff =
              (mo + n) * D;

            for (d = 0; d < D; d++) {

              ymlp[t * D + d] +=
                xv *
                W.dec[doff + d];
            }
          }
        }
      }

      lnRows(
        ymlp,
        T,
        D
      );

      captures.push({
        module_path: 'ln',
        type: 'LayerNorm',
        output: statsOf(
          ymlp,
          [1, 1, T, D],
          'ln_out'
        )
      });

      for (
        var k = 0;
        k < x.length;
        k++
      ) {
        x[k] += ymlp[k];
      }

      lnRows(
        x,
        T,
        D
      );

      captures.push({
        module_path: 'ln',
        type: 'LayerNorm',
        output: statsOf(
          x,
          [1, 1, T, D],
          'ln_out'
        )
      });
    }

    var ropePhases = [];

    for (
      var position = 0;
      position < T;
      position++
    ) {

      var phaseRow = [];

      for (
        var dimension = 0;
        dimension < 64;
        dimension++
      ) {

        phaseRow.push(
          position *
          W.freqs[dimension]
        );
      }

      ropePhases.push(phaseRow);
    }

    return {
      text: text,
      tokens: toks,
      token_count: T,
      captures: captures,
      attentionMap:
        attentionMaps.length
          ? attentionMaps[attentionMaps.length - 1]
          : [],
      attentionByLayer:
        attentionMaps,
      rope: {
        frequencies:
          Array.prototype.slice.call(
            W.freqs,
            0,
            64
          ),
        phases: ropePhases
      },
      selectedHead: 0,
      selectedPosition: 0,
      final_output: statsOf(
        x,
        [1, T, D],
        'final'
      )
    };
  }

  function setupLiveInference() {

    var probe =
      document.getElementById(
        'probeText'
      );

    var sel =
      document.getElementById(
        'inputSelect'
      );

    var row =
      document.getElementById(
        'presetRow'
      );

    var runButton =
      document.getElementById(
        'runInference'
      );

    var status =
      document.getElementById(
        'runStatus'
      );

    var chip =
      document.getElementById(
        'statusChip'
      );

    if (
      !probe ||
      !sel ||
      !row ||
      !runButton ||
      typeof BDH_DATA === 'undefined'
    ) {
      return;
    }

    function rebuildInputs() {

      sel.innerHTML = '';

      BDH_DATA.inputs.forEach(
        function (input, index) {

          var option =
            document.createElement(
              'option'
            );

          option.value =
            String(index);

          option.textContent =
            input.text;

          sel.appendChild(option);
        }
      );

      sel.value =
        String(currentInput);
    }

    function rebuildPresets() {

      row.innerHTML = '';

      BDH_DATA.inputs.forEach(
        function (input, index) {

          var button =
            document.createElement(
              'button'
            );

          button.type = 'button';

          button.className =
            'preset-btn';

          button.textContent =
            '"' +
            input.text +
            '"';

          button.addEventListener(
            'click',
            function () {

              currentInput =
                index;

              sel.value =
                String(index);

              probe.value =
                input.text;

              selectedLayer = 0;

              if (
                typeof refreshApplication ===
                'function'
              ) {
                refreshApplication();
              }
            }
          );

          row.appendChild(
            button
          );
        }
      );
    }

    function sync() {

      var index =
        parseInt(
          sel.value,
          10
        ) || 0;

      currentInput =
        index;

      var input =
        BDH_DATA.inputs[index];

      if (!input) return;

      probe.value =
        input.text;

      var inputChip =
        document.getElementById(
          'inputChip'
        );

      var captureChip =
        document.getElementById(
          'captureChip'
        );

      if (inputChip) {
        inputChip.textContent =
          'Inputs: ' +
          BDH_DATA.inputs.length;
      }

      if (captureChip) {
        captureChip.textContent =
          'Captures: ' +
          input.captures.length;
      }

      row
        .querySelectorAll(
          '.preset-btn'
        )
        .forEach(
          function (button, i) {
            button.classList.toggle(
              'active',
              i === index
            );
          }
        );
    }

    sel.addEventListener(
      'change',
      function () {

        currentInput =
          parseInt(
            sel.value,
            10
          ) || 0;

        selectedLayer = 0;

        sync();

        if (
          typeof refreshApplication ===
          'function'
        ) {
          refreshApplication();
        }
      }
    );

    runButton.addEventListener(
      'click',
      function () {

        var text =
          (
            probe.value || ''
          )
            .trim()
            .slice(
              0,
              MAXT
            );

        if (!text) {
          status.textContent =
            'Enter a sentence first.';

          return;
        }

        runButton.disabled = true;
        runButton.textContent =
          '◌ Computing…';

        if (chip) {
          chip.textContent =
            '◌ RUNNING';
          chip.classList.remove(
            'chip-ok'
          );
          chip.classList.add(
            'chip-running'
          );
        }

        if (status) {
          status.textContent =
            'Running live BDH mirror…';
        }

        setTimeout(
          function () {

            try {

              var result =
                run(text);

              BDH_DATA.inputs.push(
                result
              );

              currentInput =
                BDH_DATA.inputs.length - 1;

              selectedLayer = 0;

              rebuildInputs();
              rebuildPresets();

              sel.value =
                String(currentInput);

              sync();

              if (
                typeof refreshApplication ===
                'function'
              ) {
                refreshApplication();
              }

              if (chip) {
                chip.textContent =
                  '✓ READY';

                chip.classList.remove(
                  'chip-running'
                );

                chip.classList.add(
                  'chip-ok'
                );
              }

              if (status) {
                status.textContent =
                  'Inference complete · ' +
                  result.token_count +
                  ' tokens';
              }

            } catch (error) {

              console.error(
                error
              );

              if (chip) {
                chip.textContent =
                  '× ERROR';

                chip.classList.remove(
                  'chip-running'
                );
              }

              if (status) {
                status.textContent =
                  error.message ||
                  'Inference failed.';
              }

            }

            runButton.disabled =
              false;

            runButton.textContent =
              '▶ Run inference';

          },
          40
        );
      }
    );

    rebuildInputs();
    rebuildPresets();
    sync();
  }

  if (
    document.readyState ===
    'loading'
  ) {
    document.addEventListener(
      'DOMContentLoaded',
      setupLiveInference
    );
  } else {
    setupLiveInference();
  }

})();
