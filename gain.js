// gain.js - Calculates and visualizes Signal Gain (Output Std / Input Std)
(function () {
  function renderGain() {
    var host = document.getElementById('vizContainer');
    var sel = document.getElementById('inputSelect');
    if (!host || !sel || typeof BDH_DATA === 'undefined') return;
    var data = BDH_DATA.inputs[parseInt(sel.value) || 0];

    var s = '<div style="max-width:700px;margin:0 auto">';
    s += '<h4 style="color:#c9d1d9;margin-bottom:12px">Signal Gain Map (Output Std / Input Std)</h4>';
    s += '<p style="color:#8b949e;font-size:12px;margin-bottom:16px">Shows how much each module amplifies or attenuates the signal. High gain = explosion risk; Low gain = vanishing risk. LayerNorm acts as the stabilizer.</p>';
    
    s += '<div style="display:flex;flex-direction:column;gap:6px">';
    
    data.captures.forEach(function (cap) {
      if (cap.input && cap.input.std > 0 && cap.output && cap.output.std > 0) {
        var gain = cap.output.std / cap.input.std;
        var logGain = Math.log10(gain + 1e-9); // Log scale for visualization
        var color = '#3fb950'; // Green (Stable)
        if (gain > 100) color = '#f85149'; // Red (Explosion risk)
        else if (gain < 0.1) color = '#58a6ff'; // Blue (Attenuation)
        
        var barWidth = Math.min(100, Math.max(5, Math.abs(logGain) * 20));
        
        s += '<div style="display:flex;align-items:center;gap:12px;font-family:monospace;font-size:12px">';
        s += '<div style="width:100px;color:#8b949e">' + cap.module_path + '</div>';
        s += '<div style="flex:1;background:#161b22;border-radius:4px;height:16px;position:relative">';
        s += '<div style="width:' + barWidth + '%;height:100%;background:' + color + ';border-radius:4px;opacity:0.8"></div>';
        s += '</div>';
        s += '<div style="width:80px;text-align:right;color:' + color + '">x' + gain.toFixed(2) + '</div>';
        s += '</div>';
      }
    });
    
    s += '</div>';
    s += '<p style="color:#8b949e;font-size:11px;margin-top:16px;border-top:1px solid #30363d;padding-top:8px">Note: Attention modules show massive raw gain, but subsequent LayerNorm modules immediately compress the std back to ~1.0, proving the architecture\'s stability.</p>';
    s += '</div>';
    host.innerHTML = s;
  }

  var bar = document.querySelector('.bottom-bar');
  if (bar && !document.getElementById('gainBtn')) {
    var b = document.createElement('button');
    b.type = 'button'; b.className = 'signal-btn'; b.id = 'gainBtn'; b.textContent = 'Signal Gain';
    b.addEventListener('click', function () {
      document.querySelectorAll('.signal-btn').forEach(function (x) { x.classList.remove('active'); });
      b.classList.add('active');
      renderGain();
    });
    bar.appendChild(b);
    
    var sel = document.getElementById('inputSelect');
    if (sel) sel.addEventListener('change', function () { if (b.classList.contains('active')) renderGain(); });
  }
})();