const $ = id => document.getElementById(id);
let allAlerts = [];

async function post(url, body) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  const text = await response.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    data = { error: text || 'Request failed' };
  }
  if (!response.ok) throw Error(data.error || `Request failed (${response.status})`);
  return data;
}

function setBusy(button, busy, label) {
  button.disabled = busy;
  button.textContent = busy ? 'Analyzing…' : label;
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
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
      $('alerts').innerHTML = `<div class="alert"><b>Hazard analysis unavailable</b><span class="muted">${escapeHtml(error.message)}</span></div>`;
    }
  } catch (error) {
    $('severity').textContent = 'ERROR';
    $('severity').style.color = 'var(--danger)';
    $('riskMeter').style.width = '100%';
    $('riskText').textContent = error.message;
    $('alertCount').textContent = 'Error';
    $('alerts').innerHTML = '<div class="alert high"><b>Unable to load weather</b><span class="muted">Check the city name and try again.</span></div>';
    $('hourlyForecast').innerHTML = '<div class="muted">Hourly forecast unavailable.</div>';
    $('forecast').innerHTML = '<div class="muted">Forecast unavailable.</div>';
  } finally {
    setBusy(button, false, 'Analyze');
  }
}

function weatherIcon(code) {
  const icons = {
    0:'☀️', 1:'🌤️', 2:'⛅', 3:'☁️', 45:'🌫️', 48:'🌫️',
    51:'🌦️', 53:'🌦️', 55:'🌧️', 56:'🌧️', 57:'🌧️',
    61:'🌧️', 63:'🌧️', 65:'🌧️', 66:'🌧️', 67:'🌧️',
    71:'🌨️', 73:'🌨️', 75:'❄️', 77:'❄️', 80:'🌦️',
    81:'🌧️', 82:'🌧️', 85:'🌨️', 86:'❄️', 95:'⛈️',
    96:'⛈️', 99:'⛈️'
  };
  return icons[code] || '🌤️';
}

function renderHourly(data, currentTime) {
  const times = Array.isArray(data.time) ? data.time : [];
  const temps = Array.isArray(data.temperature_2m) ? data.temperature_2m : [];
  const rainChance = Array.isArray(data.precipitation_probability) ? data.precipitation_probability : [];
  const rain = Array.isArray(data.precipitation) ? data.precipitation : [];
  const codes = Array.isArray(data.weather_code) ? data.weather_code : [];
  const winds = Array.isArray(data.wind_speed_10m) ? data.wind_speed_10m : [];

  if (!times.length) {
    $('hourlyForecast').innerHTML = '<div class="muted">Hourly forecast unavailable.</div>';
    return;
  }

  const startIndex = currentTime
    ? Math.max(0, times.findIndex(time => time >= currentTime))
    : 0;
  const first = startIndex < 0 ? 0 : startIndex;
  const items = times.slice(first, first + 12);

  $('hourlyForecast').innerHTML = items.map((time, offset) => {
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

function renderAlerts(data) {
  allAlerts = Array.isArray(data.alerts) ? data.alerts : [];
  const severity = data.highest_severity || 'NONE';
  $('severity').textContent = severity;
  $('severity').style.color = severity === 'HIGH'
    ? 'var(--danger)'
    : severity === 'MODERATE' ? 'var(--warn)' : 'var(--ok)';
  $('alertCount').textContent = allAlerts.length
    ? `${allAlerts.length} alert${allAlerts.length === 1 ? '' : 's'} detected`
    : 'No active alerts';
  const meterWidth = severity === 'HIGH' ? '100%' : severity === 'MODERATE' ? '60%' : '18%';
  $('riskMeter').style.width = meterWidth;
  $('riskText').textContent = data.alert_count
    ? `${data.alert_count} forecast hazard${data.alert_count === 1 ? '' : 's'} detected. Review the advisories below for details.`
    : 'No major hazards detected by the application thresholds.';
  renderAlertList(false);
}

function renderAlertList(expanded) {
  const box = $('alerts');
  if (!allAlerts.length) {
    box.innerHTML = '<div class="alert none"><div class="alert-title"><b>✓ No major hazards detected</b><span class="badge">CLEAR</span></div><span class="muted">Conditions are below the application alert thresholds.</span></div>';
    return;
  }

  const visible = expanded ? allAlerts : allAlerts.slice(0, 3);
  box.innerHTML = visible.map(alert => `
    <div class="alert ${alert.severity === 'HIGH' ? 'high' : ''}">
      <div class="alert-title"><b>${escapeHtml(alert.hazard)}</b><span class="badge">${escapeHtml(alert.severity)}</span></div>
      <span class="muted">${escapeHtml(alert.date)} · ${escapeHtml(alert.details)}</span>
      <span class="muted">${escapeHtml(alert.recommendation)}</span>
    </div>
  `).join('') + (allAlerts.length > 3
    ? `<button class="more-alerts" onclick="renderAlertList(${!expanded})">${expanded ? 'Show fewer' : `+${allAlerts.length - 3} more alerts`}</button>`
    : '');
}

function renderForecast(data) {
  const dates = Array.isArray(data.time) ? data.time : [];
  const codes = Array.isArray(data.weather_code) ? data.weather_code : [];
  const max = Array.isArray(data.temperature_2m_max) ? data.temperature_2m_max : [];
  const min = Array.isArray(data.temperature_2m_min) ? data.temperature_2m_min : [];
  if (!dates.length) {
    $('forecast').innerHTML = '<div class="muted">Forecast unavailable.</div>';
    return;
  }

  const labels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
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
  const query = location ? `${question}\nLocation: ${location}` : question;
  askAgentRequest(query);
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
  } finally {
    button.disabled = false;
    button.textContent = 'Ask Agent';
  }
}

function useQuestion(text) {
  $('question').value = text;
  $('question').focus();
}

$('question').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    askAgent();
  }
});

loadWeather();
