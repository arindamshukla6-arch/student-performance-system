/* Student Performance Early Warning System - frontend helpers.
   The server re-checks every value, so this file is only for convenience. */

// Same colours as the CSS risk palette
window.RISK_COLORS = {
  'Low Risk': '#17855b',
  'Moderate Risk': '#e0a526',
  'High Risk': '#c43a3a'
};

// Shared Chart.js options (called after Chart.js has loaded)
window.chartOptions = function (opts) {
  opts = opts || {};
  if (window.Chart) {
    Chart.defaults.font.family = '"Figtree", "Segoe UI", system-ui, sans-serif';
    Chart.defaults.font.size = 13;
    Chart.defaults.color = '#5a6a85';
  }
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { position: 'bottom', labels: { usePointStyle: true, boxWidth: 8 } } },
    scales: {
      x: { stacked: !!opts.stacked, grid: { display: false } },
      y: {
        stacked: !!opts.stacked,
        beginAtZero: true,
        max: opts.max,
        ticks: { precision: 0 },
        title: opts.yTitle ? { display: true, text: opts.yTitle } : undefined,
        grid: { color: '#e8eef8' }
      }
    }
  };
};

document.addEventListener('DOMContentLoaded', function () {
  // ----- Mobile sidebar -----
  var sidebar = document.getElementById('sidebar');
  var menuBtn = document.getElementById('menu-btn');
  if (sidebar && menuBtn) {
    menuBtn.addEventListener('click', function () {
      var open = sidebar.classList.toggle('open');
      menuBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    document.addEventListener('click', function (e) {
      if (sidebar.classList.contains('open') && !sidebar.contains(e.target) && !menuBtn.contains(e.target)) {
        sidebar.classList.remove('open');
        menuBtn.setAttribute('aria-expanded', 'false');
      }
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') sidebar.classList.remove('open');
    });
  }

  // ----- Confirm before destructive actions -----
  document.querySelectorAll('form[data-confirm]').forEach(function (form) {
    form.addEventListener('submit', function (e) {
      if (!window.confirm(form.getAttribute('data-confirm'))) e.preventDefault();
    });
  });

  // ----- Print button on the report page -----
  document.querySelectorAll('[data-print]').forEach(function (btn) {
    btn.addEventListener('click', function () { window.print(); });
  });

  // ----- Student / prediction form helpers -----
  var form = document.querySelector('form[data-validate]');
  if (!form) return;

  var obtained = form.querySelector('#internal_obtained');
  var maximum = form.querySelector('#internal_max');
  var previous = form.querySelector('#previous_value');
  var scale = form.querySelector('#previous_scale');
  var attendance = form.querySelector('#attendance');
  var internalPreview = form.querySelector('#internal-preview');
  var previousPreview = form.querySelector('#previous-preview');

  function num(el) {
    var v = parseFloat(el.value);
    return isFinite(v) ? v : null;
  }

  function updatePreviews() {
    var o = num(obtained), m = num(maximum);
    if (o !== null && m !== null && m > 0) {
      internalPreview.textContent = o > m
        ? 'Marks obtained cannot be more than the maximum.'
        : 'That is ' + (o / m * 100).toFixed(1) + '% of the maximum.';
    } else {
      internalPreview.textContent = '';
    }

    var top = scale.value === 'cgpa' ? 10 : 100;
    previous.max = top;
    var p = num(previous);
    if (p !== null) {
      if (p > top) {
        previousPreview.textContent = 'This scale goes up to ' + top + '.';
      } else {
        previousPreview.textContent = scale.value === 'cgpa'
          ? 'Treated as about ' + (p * 10).toFixed(1) + '% on the percentage scale.'
          : '';
      }
    } else {
      previousPreview.textContent = '';
    }
  }

  [obtained, maximum, previous, scale].forEach(function (el) {
    el.addEventListener('input', updatePreviews);
    el.addEventListener('change', updatePreviews);
  });
  updatePreviews();

  // Cross-field checks that HTML attributes cannot express
  function setRules() {
    var o = num(obtained), m = num(maximum), p = num(previous);
    obtained.setCustomValidity(o !== null && m !== null && o > m ? 'Marks obtained cannot be more than the maximum.' : '');
    var top = scale.value === 'cgpa' ? 10 : 100;
    previous.setCustomValidity(p !== null && p > top ? 'Enter a value up to ' + top + ' for this scale.' : '');
  }

  form.addEventListener('submit', function (e) {
    setRules();
    if (!form.checkValidity()) {
      e.preventDefault();
      form.reportValidity();
    }
  });
  [obtained, maximum, previous, scale, attendance].forEach(function (el) {
    el.addEventListener('input', setRules);
  });
});
