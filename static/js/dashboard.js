const $ = id => document.getElementById(id);
let allAlerts = [];
let alertSeverityFilter = 'ALL';
let alertDateFilter = 'all';

async function post(url, body) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  const text = await response.text();
  let data;
  try { data = JSON.parse(text); } catch { data = { error: text || 'Request failed' }; }
  if (!response.ok) throw Error(data.error || `Request failed (${response.status})`);
  return data;
}

function setBusy(button, busy, label) {
  button.disabled = busy;
  button.textContent = busy ? 'Analyzing…' : label;
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;').replaceAll("'", '&#039;');
}

async function loadWeather() {
  const location = $('location').value.trim();
  if (!location) return;
  const button = $('analyze');
  setBusy(button, true, 'Analyze');
  $('riskText').textContent = 'Analyzing live forecast…';
  $('severity').textContent = '—';
  $('alertCount').textContent = 'Checking…';
  $('alerts').innerHTML = '<div class="muted">Loading hazard analysis…</div>';
  $('hourlyForecast').innerHTML = '<div class="muted">Loading hourly forecast…</div>';

  try {
    const data = await post('/weather/current', { location });
    const current = data.current || {};
    const displayLocation = data.location?.display_name || location;
    const cityName = location.split(',')[0].trim() || location;
    $('place').textContent = `${cityName} Weather`;
    $('locationDetail').textContent = displayLocation;
    $('temp').textContent = `${current.temperature_2m ?? '—'}°C`;
    $('feels').textContent = `${current.apparent_temperature ?? '—'}°C`;
    $('tempStat').textContent = `${current.temperature_2m ?? '—'}°C`;
    $('feelsStat').textContent = `${current.apparent_temperature ?? '—'}°C`;
    $('humidity').textContent = `${current.relative_humidity_2m ?? '—'}%`;
    $('wind').textContent = `${current.wind_speed_10m ?? '—'} km/h`;
    renderHourly(data.hourly || {}, current.time);
    renderForecast(data.daily || {});
    $('question').placeholder = `Ask about ${cityName} weather…`;
    try {
      const alerts = await post('/weather/alerts', { location });
      renderAlerts(alerts);
    } catch (error) {
      $('severity').textContent = 'UNAVAILABLE';
      $('severity').style.color = 'var(--warn)';
      $('riskMeter').style.width = '45%';
      $('alertCount').textContent = 'Unavailable';
      $('riskText').textContent = 'Weather loaded, but hazard analysis is temporarily unavailable.';
      $('alerts').innerHTML = `<div class="alert"><div class="alert-main"><b>Hazard analysis unavailable</b><div class="alert-detail">${escapeHtml(error.message)}</div></div></div>`;
    }
  } catch (error) {
    $('severity').textContent = 'ERROR';
    $('severity').style.color = 'var(--danger)';
    $('riskMeter').style.width = '100%';
    $('riskText').textContent = error.message;
    $('alertCount').textContent = 'Error';
    $('alerts').innerHTML = '<div class="alert high"><div class="alert-main"><b>Unable to load weather</b><div class="alert-detail">Check the city name and try again.</div></div></div>';
    $('hourlyForecast').innerHTML = '<div class="muted">Hourly forecast unavailable.</div>';
    $('forecast').innerHTML = '<div class="muted">Forecast unavailable.</div>';
  } finally { setBusy(button, false, 'Analyze'); }
}

function weatherIcon(code) {
  const icons = {0:'☀️',1:'🌤️',2:'⛅',3:'☁️',45:'🌫️',48:'🌫️',51:'🌦️',53:'🌦️',55:'🌧️',56:'🌧️',57:'🌧️',61:'🌧️',63:'🌧️',65:'🌧️',66:'🌧️',67:'🌧️',71:'🌨️',73:'🌨️',75:'❄️',77:'❄️',80:'🌦️',81:'🌧️',82:'🌧️',85:'🌨️',86:'❄️',95:'⛈️',96:'⛈️',99:'⛈️'};
  return icons[code] || '🌤️';
}

function renderHourly(data, currentTime) {
  const times = Array.isArray(data.time) ? data.time : [];
  const temps = Array.isArray(data.temperature_2m) ? data.temperature_2m : [];
  const rainChance = Array.isArray(data.precipitation_probability) ? data.precipitation_probability : [];
  const rain = Array.isArray(data.precipitation) ? data.precipitation : [];
  const codes = Array.isArray(data.weather_code) ? data.weather_code : [];
  const winds = Array.isArray(data.wind_speed_10m) ? data.wind_speed_10m : [];
  if (!times.length) { $('hourlyForecast').innerHTML = '<div class="muted">Hourly forecast unavailable.</div>'; return; }
  const startIndex = currentTime ? Math.max(0, times.findIndex(time => time >= currentTime)) : 0;
  const first = startIndex < 0 ? 0 : startIndex;
  $('hourlyForecast').innerHTML = times.slice(first, first + 12).map((time, offset) => {
    const index = first + offset;
    const date = new Date(time);
    const label = offset === 0 ? 'Now' : date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
    const temp = temps[index] == null ? '—' : `${Math.round(temps[index])}°`;
    const chance = rainChance[index] == null ? '—' : `${Math.round(rainChance[index])}%`;
    const amount = rain[index] == null ? '0' : Number(rain[index]).toFixed(1);
    const wind = winds[index] == null ? '—' : `${Math.round(winds[index])}`;
    return `<div class="hour"><small>${label}</small><b>${weatherIcon(codes[index])}</b><strong>${temp}</strong><span class="hour-rain">💧 ${chance}</span><span class="hour-meta">${amount} mm · ${wind} km/h</span></div>`;
  }).join('');
}

function formatAlertDate(date) {
  const d = new Date(`${date}T12:00:00`);
  return Number.isNaN(d.getTime()) ? date : d.toLocaleDateString([], { weekday:'short', month:'short', day:'numeric' });
}

function populateAlertDates() {
  const select = $('alertDateFilter');
  const dates = [...new Set(allAlerts.map(alert => alert.date).filter(Boolean))].sort();
  const current = alertDateFilter;
  select.innerHTML = '<option value="all">All forecast dates</option>' + dates.map(date => `<option value="${escapeHtml(date)}">${escapeHtml(formatAlertDate(date))}</option>`).join('');
  select.value = dates.includes(current) ? current : 'all';
  alertDateFilter = select.value;
}

function setAlertSeverity(severity) {
  alertSeverityFilter = severity;
  document.querySelectorAll('.alert-filter').forEach(button => button.classList.toggle('active', button.dataset.severity === severity));
  renderAlertList();
}

function renderAlerts(data) {
  allAlerts = Array.isArray(data.alerts) ? data.alerts : [];
  alertSeverityFilter = 'ALL';
  alertDateFilter = 'all';
  populateAlertDates();
  document.querySelectorAll('.alert-filter').forEach(button => button.classList.toggle('active', button.dataset.severity === 'ALL'));
  const severity = data.highest_severity || 'NONE';
  $('severity').textContent = severity;
  $('severity').style.color = severity === 'HIGH' ? 'var(--danger)' : severity === 'MODERATE' ? 'var(--warn)' : 'var(--ok)';
  $('alertCount').textContent = allAlerts.length ? `${allAlerts.length} alert${allAlerts.length === 1 ? '' : 's'} detected` : 'No active alerts';
  $('riskMeter').style.width = severity === 'HIGH' ? '100%' : severity === 'MODERATE' ? '60%' : '18%';
  $('riskText').textContent = data.alert_count ? `${data.alert_count} forecast hazard${data.alert_count === 1 ? '' : 's'} detected. Review the advisories below for details.` : 'No major hazards detected by the application thresholds.';
  renderAlertList();
}

function renderAlertList() {
  const box = $('alerts');
  let filtered = allAlerts.filter(alert => alertSeverityFilter === 'ALL' || alert.severity === alertSeverityFilter);
  if (alertDateFilter !== 'all') filtered = filtered.filter(alert => alert.date === alertDateFilter);
  if (!filtered.length) {
    box.innerHTML = '<div class="alert-empty">No alerts match the selected filters.</div>';
    return;
  }
  const grouped = filtered.reduce((groups, alert) => {
    const date = alert.date || 'Unknown date';
    (groups[date] ||= []).push(alert);
    return groups;
  }, {});
  const dates = Object.keys(grouped).sort();
  box.innerHTML = dates.map(date => `
    <div class="alert-day">
      <div class="alert-day-heading"><span>${escapeHtml(formatAlertDate(date))}</span><span class="alert-day-count">${grouped[date].length} alert${grouped[date].length === 1 ? '' : 's'}</span></div>
      ${grouped[date].map(alert => `
        <div class="alert ${alert.severity === 'HIGH' ? 'high' : ''}">
          <div class="alert-main">
            <div class="alert-title"><b>${escapeHtml(alert.hazard)}</b><span class="badge">${escapeHtml(alert.severity)}</span></div>
            <div class="alert-detail">${escapeHtml(alert.details)} ${escapeHtml(alert.recommendation)}</div>
          </div>
          <div class="alert-meta">${escapeHtml(alert.date)}</div>
        </div>
      `).join('')}
    </div>
  `).join('');
}

function renderForecast(data) {
  const dates = Array.isArray(data.time) ? data.time : [];
  const codes = Array.isArray(data.weather_code) ? data.weather_code : [];
  const max = Array.isArray(data.temperature_2m_max) ? data.temperature_2m_max : [];
  const min = Array.isArray(data.temperature_2m_min) ? data.temperature_2m_min : [];
  if (!dates.length) { $('forecast').innerHTML = '<div class="muted">Forecast unavailable.</div>'; return; }
  const labels = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  $('forecast').innerHTML = dates.slice(0, 7).map((date, index) => {
    const day = new Date(`${date}T12:00:00`);
    const high = max[index] == null ? '—' : `${Math.round(max[index])}°`;
    const low = min[index] == null ? '—' : `${Math.round(min[index])}°`;
    return `<div class="day"><small>${index === 0 ? 'Today' : labels[day.getDay()]}</small><b>${weatherIcon(codes[index])}</b><strong>${high}</strong><small>${low}</small></div>`;
  }).join('');
}

function askAgent() {
  const question = $('question').value.trim();
  if (!question) return;
  const location = $('location').value.trim();
  askAgentRequest(location ? `${question}\nLocation: ${location}` : question);
}

async function askAgentRequest(query) {
  const button = $('askButton');
  button.disabled = true;
  button.textContent = 'Thinking…';
  $('askStatus').textContent = 'Weather Agent is using live tools and grounded context…';
  $('answer').classList.remove('hidden');
  $('answer').textContent = '';
  try {
    const data = await post('/weather/agent', { query });
    $('answer').textContent = data.answer || data.response || data.final_answer || JSON.stringify(data, null, 2);
    $('askStatus').textContent = data.success ? 'Completed with live tools and grounded context.' : 'Agent returned an unsuccessful result.';
  } catch (error) {
    $('askStatus').textContent = error.message;
    $('answer').textContent = 'Unable to generate an answer. Try the question again after a moment.';
  } finally { button.disabled = false; button.textContent = 'Ask Agent'; }
}

function useQuestion(text) { $('question').value = text; $('question').focus(); }
$('alertDateFilter').addEventListener('change', event => { alertDateFilter = event.target.value; renderAlertList(); });
$('question').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); askAgent(); } });
loadWeather();
