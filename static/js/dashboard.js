const $ = id => document.getElementById(id);
let allAlerts = [];
let alertSeverityFilter = 'ALL';
let alertDateFilter = 'all';
let currentLocation = 'Kolkata';
let suggestionCategory = 'forecast';

const suggestionSets = {
  forecast: [
    city => `Will it rain in ${city} tomorrow?`,
    city => `How will the weather change in ${city} this week?`,
    city => `What will the temperature be in ${city} tomorrow?`
  ],
  travel: [
    city => `Is it a good day to travel in ${city}?`,
    city => `Should I carry an umbrella in ${city} tomorrow?`,
    city => `What weather conditions should I expect while travelling in ${city}?`
  ],
  outdoor: [
    city => `Is ${city} weather good for outdoor activities today?`,
    city => `Can I go for a run in ${city} this evening?`,
    city => `What is the best time to be outdoors in ${city} today?`
  ],
  safety: [
    city => `What weather hazards should I watch for in ${city}?`,
    city => `Is it safe to be outdoors in ${city} today?`,
    city => `Are there any severe weather risks in ${city}?`
  ]
};

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
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function setSuggestionCategory(category) {
  suggestionCategory = category;
  document.querySelectorAll('.suggestion-tab').forEach(button => {
    button.classList.toggle('active', button.dataset.category === category);
  });
  renderSuggestions();
}

function renderSuggestions() {
  const city = currentLocation.split(',')[0].trim() || 'your city';
  const items = suggestionSets[suggestionCategory] || suggestionSets.forecast;
  const container = $('suggestions');
  if (!container) return;
  container.innerHTML = items.map((makeQuestion, index) => {
    const question = makeQuestion(city);
    return `<button type="button" class="suggestion" data-question="${escapeHtml(question)}"><span class="suggestion-icon">${['↗','◷','✦'][index]}</span><span>${escapeHtml(question)}</span><span class="suggestion-arrow">→</span></button>`;
  }).join('');
}

function conditionLabel(code) {
  const labels = {0:'Clear sky',1:'Mostly clear',2:'Partly cloudy',3:'Overcast',45:'Foggy',48:'Foggy',51:'Light drizzle',53:'Drizzle',55:'Heavy drizzle',61:'Light rain',63:'Rain',65:'Heavy rain',66:'Freezing rain',67:'Heavy freezing rain',71:'Light snow',73:'Snow',75:'Heavy snow',77:'Snow grains',80:'Rain showers',81:'Rain showers',82:'Heavy showers',85:'Snow showers',86:'Heavy snow showers',95:'Thunderstorm',96:'Thunderstorm with hail',99:'Thunderstorm with hail'};
  return labels[code] || 'Mixed conditions';
}

function weatherIcon(code) {
  const icons = {0:'☀️',1:'🌤️',2:'⛅',3:'☁️',45:'🌫️',48:'🌫️',51:'🌦️',53:'🌦️',55:'🌧️',56:'🌧️',57:'🌧️',61:'🌧️',63:'🌧️',65:'🌧️',66:'🌧️',67:'🌧️',71:'🌨️',73:'🌨️',75:'❄️',77:'❄️',80:'🌦️',81:'🌧️',82:'🌧️',85:'🌨️',86:'❄️',95:'⛈️',96:'⛈️',99:'⛈️'};
  return icons[code] || '🌤️';
}

function renderInsights(current, hourly, daily) {
  const rain = Array.isArray(hourly.precipitation_probability) ? hourly.precipitation_probability : [];
  const times = Array.isArray(hourly.time) ? hourly.time : [];
  const finiteRain = rain.map(Number).filter(Number.isFinite);
  const maxRain = finiteRain.length ? Math.max(...finiteRain) : 0;
  const maxTemp = Array.isArray(daily.temperature_2m_max) && daily.temperature_2m_max.length
    ? Number(daily.temperature_2m_max[0])
    : Number(current.temperature_2m);
  const apparent = Number(current.apparent_temperature);
  const temperature = Number(current.temperature_2m);
  const wind = Number(current.wind_speed_10m);
  const rainIndex = rain.findIndex(value => Number(value) >= 60);
  const rainTime = rainIndex >= 0 && times[rainIndex]
    ? new Date(times[rainIndex]).toLocaleTimeString([], { hour: 'numeric' })
    : null;
  const cards = [];

  if (Number.isFinite(apparent) && Number.isFinite(temperature) && apparent - temperature >= 4) {
    cards.push(['🌡️', 'Feels warmer', 'The apparent temperature is noticeably higher than the actual temperature.']);
  } else if (Number.isFinite(maxTemp) && maxTemp >= 35) {
    cards.push(['☀️', 'Hot day ahead', `Temperatures may reach around ${Math.round(maxTemp)}°C today.`]);
  } else {
    cards.push(['🌡️', 'Comfort check', `Currently ${current.temperature_2m ?? '—'}°C with a ${conditionLabel(current.weather_code)}.`]);
  }

  if (maxRain >= 60) {
    cards.push(['🌧️', 'Rain likely', `${Math.round(maxRain)}% rain probability${rainTime ? ` around ${rainTime}` : ' later today'}.`]);
  } else {
    cards.push(['☁️', 'Rain check', 'No strong rainfall signal in the next few hours.']);
  }

  if (Number.isFinite(wind) && wind >= 30) {
    cards.push(['💨', 'Breezy conditions', `Winds are around ${Math.round(wind)} km/h right now.`]);
  } else {
    cards.push(['✨', 'Good to know', 'Conditions look relatively calm right now.']);
  }

  $('insightCards').innerHTML = cards.map(([icon, title, text]) =>
    `<div class="insight"><span class="insight-icon">${icon}</span><div><b>${escapeHtml(title)}</b><p>${escapeHtml(text)}</p></div></div>`
  ).join('');
}

function showWeatherError(message) {
  $('insightCards').innerHTML = `<div class="insight muted-card">${escapeHtml(message)}</div>`;
  $('hourlyForecast').innerHTML = '<div class="muted">Hourly forecast unavailable.</div>';
  $('forecast').innerHTML = '<div class="muted">Forecast unavailable.</div>';
}

async function loadWeather() {
  const location = $('location').value.trim();
  if (!location) return;

  currentLocation = location;
  renderSuggestions();
  const button = $('analyze');
  setBusy(button, true, 'Analyze');
  $('alertCount').textContent = 'Checking…';
  $('alerts').innerHTML = '<div class="muted">Checking forecast hazards…</div>';
  $('hourlyForecast').innerHTML = '<div class="muted">Loading hourly forecast…</div>';

  try {
    const [weatherResult, alertsResult] = await Promise.allSettled([
      post('/weather/current', { location }),
      post('/weather/alerts', { location })
    ]);

    if (weatherResult.status === 'rejected') {
      showWeatherError('Unable to load weather. Check the city name and try again.');
      $('alerts').innerHTML = '<div class="alert high"><div class="alert-main"><b>Unable to load weather</b><div class="alert-detail">Check the city name and try again.</div></div></div>';
      $('alertCount').textContent = 'Error';
      return;
    }

    const data = weatherResult.value;
    const current = data.current || {};
    const displayLocation = data.location?.display_name || location;
    const cityName = location.split(',')[0].trim() || location;
    currentLocation = cityName;

    $('place').textContent = `${cityName} Weather`;
    $('locationDetail').textContent = displayLocation;
    $('temp').textContent = `${current.temperature_2m ?? '—'}°C`;
    $('feels').textContent = `${current.apparent_temperature ?? '—'}°C`;
    $('tempStat').textContent = `${current.temperature_2m ?? '—'}°C`;
    $('feelsStat').textContent = `${current.apparent_temperature ?? '—'}°C`;
    $('humidity').textContent = `${current.relative_humidity_2m ?? '—'}%`;
    $('wind').textContent = `${current.wind_speed_10m ?? '—'} km/h`;
    $('currentIcon').textContent = weatherIcon(current.weather_code);
    $('condition').textContent = conditionLabel(current.weather_code);
    renderInsights(current, data.hourly || {}, data.daily || {});
    renderHourly(data.hourly || {}, current.time);
    renderForecast(data.daily || {});
    $('question').placeholder = `Ask about ${cityName} weather…`;
    $('suggestionContext').textContent = cityName;
    renderSuggestions();

    if (alertsResult.status === 'fulfilled') {
      renderAlerts(alertsResult.value);
    } else {
      $('alertCount').textContent = 'Unavailable';
      $('alerts').innerHTML = '<div class="alert"><div class="alert-main"><b>Hazard analysis unavailable</b><div class="alert-detail">Weather is available, but hazard analysis could not be loaded.</div></div></div>';
    }
  } catch (error) {
    showWeatherError(error.message || 'Unable to analyze this location.');
  } finally {
    setBusy(button, false, 'Analyze');
  }
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

  const found = currentTime ? times.findIndex(time => time >= currentTime) : 0;
  const first = found < 0 ? 0 : found;
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
  return Number.isNaN(d.getTime()) ? date : d.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric' });
}

function populateAlertDates() {
  const select = $('alertDateFilter');
  const dates = [...new Set(allAlerts.map(a => a.date).filter(Boolean))].sort();
  select.innerHTML = '<option value="all">All forecast dates</option>' + dates.map(date =>
    `<option value="${escapeHtml(date)}">${escapeHtml(formatAlertDate(date))}</option>`
  ).join('');
  select.value = dates.includes(alertDateFilter) ? alertDateFilter : 'all';
  alertDateFilter = select.value;
}

function setAlertSeverity(severity) {
  alertSeverityFilter = severity;
  document.querySelectorAll('.alert-filter').forEach(button => {
    button.classList.toggle('active', button.dataset.severity === severity);
  });
  renderAlertList();
}

function renderAlerts(data) {
  allAlerts = Array.isArray(data.alerts) ? data.alerts : [];
  alertSeverityFilter = 'ALL';
  alertDateFilter = 'all';
  populateAlertDates();
  document.querySelectorAll('.alert-filter').forEach(button => {
    button.classList.toggle('active', button.dataset.severity === 'ALL');
  });
  $('alertCount').textContent = allAlerts.length
    ? `${allAlerts.length} alert${allAlerts.length === 1 ? '' : 's'} detected`
    : 'No active alerts';
  renderAlertList();
}

function renderAlertList() {
  const box = $('alerts');
  let filtered = allAlerts.filter(a => alertSeverityFilter === 'ALL' || a.severity === alertSeverityFilter);
  if (alertDateFilter !== 'all') filtered = filtered.filter(a => a.date === alertDateFilter);
  if (!filtered.length) {
    box.innerHTML = '<div class="alert-empty">No alerts match the selected filters.</div>';
    return;
  }

  const grouped = filtered.reduce((groups, alert) => {
    const date = alert.date || 'Unknown date';
    (groups[date] ||= []).push(alert);
    return groups;
  }, {});

  box.innerHTML = Object.keys(grouped).sort().map(date =>
    `<div class="alert-day"><div class="alert-day-heading"><span>${escapeHtml(formatAlertDate(date))}</span><span class="alert-day-count">${grouped[date].length} alert${grouped[date].length === 1 ? '' : 's'}</span></div>${grouped[date].map(alert =>
      `<div class="alert ${alert.severity === 'HIGH' ? 'high' : ''}"><div class="alert-main"><div class="alert-title"><b>${escapeHtml(alert.hazard)}</b><span class="badge">${escapeHtml(alert.severity)}</span></div><div class="alert-detail">${escapeHtml(alert.details)} ${escapeHtml(alert.recommendation)}</div></div><div class="alert-meta">${escapeHtml(alert.date)}</div></div>`
    ).join('')}</div>`
  ).join('');
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
  $('askStatus').textContent = 'Finding the best answer…';
  $('answer').classList.remove('hidden');
  $('answer').innerHTML = '<div class="answer-loading"><span></span><span></span><span></span></div>';
  try {
    const data = await post('/weather/agent', { query });
    const citations = Array.isArray(data.citations) && data.citations.length
      ? `<div class="answer-meta"><span>Grounded answer</span><span>Sources ${data.citations.map(c => escapeHtml(c)).join(' · ')}</span></div>`
      : '';
    const confidence = typeof data.confidence === 'number'
      ? `<span>Confidence ${Math.round(data.confidence * 100)}%</span>`
      : '';
    const warnings = Array.isArray(data.warnings) && data.warnings.length
      ? `<div class="answer-warning">⚠️ ${escapeHtml(data.warnings.join(' '))}</div>`
      : '';
    $('answer').innerHTML = `<div class="answer-label">✨ Weather Intelligence</div><div class="answer-text">${escapeHtml(data.answer || data.response || data.final_answer || 'No answer returned.')}</div>${warnings}${confidence || citations ? `<div class="answer-meta">${confidence}${citations ? `<span>${citations.replace('<div class="answer-meta">','').replace('</div>','')}</span>` : ''}</div>` : ''}`;
    $('askStatus').textContent = data.success ? 'Here’s what the weather suggests.' : 'The agent returned an incomplete result.';
  } catch (error) {
    $('askStatus').textContent = error.message || 'Unable to answer right now.';
    $('answer').innerHTML = '<div class="answer-label">Unable to answer</div><div class="answer-text">Try the question again in a moment.</div>';
  } finally {
    button.disabled = false;
    button.textContent = 'Ask Agent';
  }
}

function useQuestion(text) {
  $('question').value = text;
  $('question').focus();
}

$('analyze').addEventListener('click', loadWeather);
$('askButton').addEventListener('click', askAgent);
$('location').addEventListener('keydown', event => {
  if (event.key === 'Enter') {
    event.preventDefault();
    loadWeather();
  }
});
document.querySelectorAll('.suggestion-tab').forEach(button => {
  button.addEventListener('click', () => setSuggestionCategory(button.dataset.category));
});
$('suggestions').addEventListener('click', event => {
  const button = event.target.closest('.suggestion');
  if (!button) return;
  const question = button.dataset.question;
  if (!question) return;
  useQuestion(question);
  askAgentRequest(question + `\nLocation: ${$('location').value.trim() || currentLocation}`);
});
$('alertDateFilter').addEventListener('change', event => {
  alertDateFilter = event.target.value;
  renderAlertList();
});
document.querySelectorAll('.alert-filter').forEach(button => {
  button.addEventListener('click', () => setAlertSeverity(button.dataset.severity));
});
$('question').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    askAgent();
  }
});

renderSuggestions();
loadWeather();