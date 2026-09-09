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
  const response = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const text = await response.text();
  let data;
  try { data = JSON.parse(text); } catch { data = { error: text || 'Request failed' }; }
  if (!response.ok) throw Error(data.error || `Request failed (${response.status})`);
  return data;
}

function setBusy(button, busy, label) { button.disabled = busy; button.textContent = busy ? 'Analyzing…' : label; }
function escapeHtml(value) { return String(value ?? '').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#039;'); }

function setSuggestionCategory(category) {
  suggestionCategory = category;
  document.querySelectorAll('.suggestion-tab').forEach(button => button.classList.toggle('active', button.dataset.category === category));
  renderSuggestions();
}

function renderSuggestions() {
  const city = currentLocation.split(',')[0].trim() || 'your city';
  const items = suggestionSets[suggestionCategory] || suggestionSets.forecast;
  $('suggestions').innerHTML = items.map((makeQuestion, index) => {
    const question = makeQuestion(city);
    return `<button class="suggestion" onclick="useQuestion(${JSON.stringify(question)})"><span class="suggestion-icon">${['↗','◷','✦'][index]}</span><span>${escapeHtml(question)}</span><span class="suggestion-arrow">→</span></button>`;
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
  const maxRain = rain.length ? Math.max(...rain.map(Number).filter(Number.isFinite)) : 0;
  const maxTemp = Array.isArray(daily.temperature_2m_max) && daily.temperature_2m_max.length ? Number(daily.temperature_2m_max[0]) : Number(current.temperature_2m);
  const apparent = Number(current.apparent_temperature);
  const wind = Number(current.wind_speed_10m);
  const rainIndex = rain.findIndex(value => Number(value) >= 60);
  const rainTime = rainIndex >= 0 && times[rainIndex] ? new Date(times[rainIndex]).toLocaleTimeString([], {hour:'numeric'}) : null;
  const cards = [];
  if (Number.isFinite(apparent) && Number.isFinite(Number(current.temperature_2m)) && apparent - Number(current.temperature_2m) >= 4) {
    cards.push(['🌡️','Feels warmer','The apparent temperature is noticeably higher than the actual temperature.']);
  } else if (Number.isFinite(maxTemp) && maxTemp >= 35) {
    cards.push(['☀️','Hot day ahead',`Temperatures may reach around ${Math.round(maxTemp)}°C today.`]);
  } else {
    cards.push(['🌡️','Comfort check',`Currently ${current.temperature_2m ?? '—'}°C with a ${conditionLabel(current.weather_code)}.`]);
  }
  if (maxRain >= 60) cards.push(['🌧️','Rain likely',`${Math.round(maxRain)}% rain probability${rainTime ? ` around ${rainTime}` : ' later today'}.`]);
  else cards.push(['☁️','Rain check','No strong rainfall signal in the next few hours.']);
  if (Number.isFinite(wind) && wind >= 30) cards.push(['💨','Breezy conditions',`Winds are around ${Math.round(wind)} km/h right now.`]);
  else cards.push(['✨','Good to know','Conditions look relatively calm right now.']);
  $('insightCards').innerHTML = cards.map(([icon,title,text]) => `<div class="insight"><span class="insight-icon">${icon}</span><div><b>${escapeHtml(title)}</b><p>${escapeHtml(text)}</p></div></div>`).join('');
}

async function loadWeather() {
  const location = $('location').value.trim();
  if (!location) return;
  currentLocation = location;
  renderSuggestions();
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
    try { renderAlerts(await post('/weather/alerts', { location })); }
    catch (error) {
      $('severity').textContent = 'UNAVAILABLE'; $('severity').style.color = 'var(--warn)'; $('riskMeter').style.width = '45%'; $('alertCount').textContent = 'Unavailable';
      $('riskText').textContent = 'Weather loaded, but hazard analysis is temporarily unavailable.';
      $('alerts').innerHTML = `<div class="alert"><div class="alert-main"><b>Hazard analysis unavailable</b><div class="alert-detail">${escapeHtml(error.message)}</div></div></div>`;
    }
  } catch (error) {
    $('severity').textContent = 'ERROR'; $('severity').style.color = 'var(--danger)'; $('riskMeter').style.width = '100%'; $('riskText').textContent = error.message; $('alertCount').textContent = 'Error';
    $('insightCards').innerHTML = '<div class="insight muted-card">Unable to analyze this location.</div>';
    $('alerts').innerHTML = '<div class="alert high"><div class="alert-main"><b>Unable to load weather</b><div class="alert-detail">Check the city name and try again.</div></div></div>';
    $('hourlyForecast').innerHTML = '<div class="muted">Hourly forecast unavailable.</div>'; $('forecast').innerHTML = '<div class="muted">Forecast unavailable.</div>';
  } finally { setBusy(button, false, 'Analyze'); }
}

function renderHourly(data, currentTime) {
  const times = Array.isArray(data.time) ? data.time : [], temps = Array.isArray(data.temperature_2m) ? data.temperature_2m : [], rainChance = Array.isArray(data.precipitation_probability) ? data.precipitation_probability : [], rain = Array.isArray(data.precipitation) ? data.precipitation : [], codes = Array.isArray(data.weather_code) ? data.weather_code : [], winds = Array.isArray(data.wind_speed_10m) ? data.wind_speed_10m : [];
  if (!times.length) { $('hourlyForecast').innerHTML = '<div class="muted">Hourly forecast unavailable.</div>'; return; }
  const found = currentTime ? times.findIndex(time => time >= currentTime) : 0, first = found < 0 ? 0 : found;
  $('hourlyForecast').innerHTML = times.slice(first, first + 12).map((time, offset) => {
    const index = first + offset, date = new Date(time), label = offset === 0 ? 'Now' : date.toLocaleTimeString([], {hour:'numeric',minute:'2-digit'}), temp = temps[index] == null ? '—' : `${Math.round(temps[index])}°`, chance = rainChance[index] == null ? '—' : `${Math.round(rainChance[index])}%`, amount = rain[index] == null ? '0' : Number(rain[index]).toFixed(1), wind = winds[index] == null ? '—' : `${Math.round(winds[index])}`;
    return `<div class="hour"><small>${label}</small><b>${weatherIcon(codes[index])}</b><strong>${temp}</strong><span class="hour-rain">💧 ${chance}</span><span class="hour-meta">${amount} mm · ${wind} km/h</span></div>`;
  }).join('');
}

function formatAlertDate(date) { const d = new Date(`${date}T12:00:00`); return Number.isNaN(d.getTime()) ? date : d.toLocaleDateString([], {weekday:'short',month:'short',day:'numeric'}); }
function populateAlertDates() { const select = $('alertDateFilter'), dates = [...new Set(allAlerts.map(a => a.date).filter(Boolean))].sort(); select.innerHTML = '<option value="all">All forecast dates</option>' + dates.map(date => `<option value="${escapeHtml(date)}">${escapeHtml(formatAlertDate(date))}</option>`).join(''); select.value = dates.includes(alertDateFilter) ? alertDateFilter : 'all'; alertDateFilter = select.value; }
function setAlertSeverity(severity) { alertSeverityFilter = severity; document.querySelectorAll('.alert-filter').forEach(button => button.classList.toggle('active', button.dataset.severity === severity)); renderAlertList(); }
function renderAlerts(data) {
  allAlerts = Array.isArray(data.alerts) ? data.alerts : []; alertSeverityFilter = 'ALL'; alertDateFilter = 'all'; populateAlertDates(); document.querySelectorAll('.alert-filter').forEach(button => button.classList.toggle('active', button.dataset.severity === 'ALL'));
  const severity = data.highest_severity || 'NONE'; $('severity').textContent = severity; $('severity').style.color = severity === 'HIGH' ? 'var(--danger)' : severity === 'MODERATE' ? 'var(--warn)' : 'var(--ok)'; $('alertCount').textContent = allAlerts.length ? `${allAlerts.length} alert${allAlerts.length === 1 ? '' : 's'} detected` : 'No active alerts'; $('riskMeter').style.width = severity === 'HIGH' ? '100%' : severity === 'MODERATE' ? '60%' : '18%'; $('riskText').textContent = data.alert_count ? `${data.alert_count} forecast hazard${data.alert_count === 1 ? '' : 's'} detected. Review the advisories below for details.` : 'No major hazards detected by the application thresholds.'; renderAlertList();
}
function renderAlertList() {
  const box = $('alerts'); let filtered = allAlerts.filter(a => alertSeverityFilter === 'ALL' || a.severity === alertSeverityFilter); if (alertDateFilter !== 'all') filtered = filtered.filter(a => a.date === alertDateFilter);
  if (!filtered.length) { box.innerHTML = '<div class="alert-empty">No alerts match the selected filters.</div>'; return; }
  const grouped = filtered.reduce((groups, alert) => { const date = alert.date || 'Unknown date'; (groups[date] ||= []).push(alert); return groups; }, {});
  box.innerHTML = Object.keys(grouped).sort().map(date => `<div class="alert-day"><div class="alert-day-heading"><span>${escapeHtml(formatAlertDate(date))}</span><span class="alert-day-count">${grouped[date].length} alert${grouped[date].length === 1 ? '' : 's'}</span></div>${grouped[date].map(alert => `<div class="alert ${alert.severity === 'HIGH' ? 'high' : ''}"><div class="alert-main"><div class="alert-title"><b>${escapeHtml(alert.hazard)}</b><span class="badge">${escapeHtml(alert.severity)}</span></div><div class="alert-detail">${escapeHtml(alert.details)} ${escapeHtml(alert.recommendation)}</div></div><div class="alert-meta">${escapeHtml(alert.date)}</div></div>`).join('')}</div>`).join('');
}
function renderForecast(data) {
  const dates = Array.isArray(data.time) ? data.time : [], codes = Array.isArray(data.weather_code) ? data.weather_code : [], max = Array.isArray(data.temperature_2m_max) ? data.temperature_2m_max : [], min = Array.isArray(data.temperature_2m_min) ? data.temperature_2m_min : [];
  if (!dates.length) { $('forecast').innerHTML = '<div class="muted">Forecast unavailable.</div>'; return; }
  const labels = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  $('forecast').innerHTML = dates.slice(0,7).map((date,index) => { const day = new Date(`${date}T12:00:00`), high = max[index] == null ? '—' : `${Math.round(max[index])}°`, low = min[index] == null ? '—' : `${Math.round(min[index])}°`; return `<div class="day"><small>${index === 0 ? 'Today' : labels[day.getDay()]}</small><b>${weatherIcon(codes[index])}</b><strong>${high}</strong><small>${low}</small></div>`; }).join('');
}
function askAgent() { const question = $('question').value.trim(); if (!question) return; const location = $('location').value.trim(); askAgentRequest(location ? `${question}\nLocation: ${location}` : question); }
async function askAgentRequest(query) {
  const button = $('askButton'); button.disabled = true; button.textContent = 'Thinking…'; $('askStatus').textContent = 'Finding the best answer…'; $('answer').classList.remove('hidden'); $('answer').innerHTML = '<div class="answer-loading"><span></span><span></span><span></span></div>';
  try {
    const data = await post('/weather/agent', { query });
    const citations = Array.isArray(data.citations) && data.citations.length ? `<div class="answer-meta"><span>Grounded answer</span><span>Sources ${data.citations.map(c => escapeHtml(c)).join(' · ')}</span></div>` : '';
    const confidence = typeof data.confidence === 'number' ? `<span>Confidence ${Math.round(data.confidence * 100)}%</span>` : '';
    const warnings = Array.isArray(data.warnings) && data.warnings.length ? `<div class="answer-warning">⚠️ ${escapeHtml(data.warnings.join(' '))}</div>` : '';
    $('answer').innerHTML = `<div class="answer-label">✨ Weather Intelligence</div><div class="answer-text">${escapeHtml(data.answer || data.response || data.final_answer || 'No answer returned.')}</div>${warnings}${citations || confidence ? `<div class="answer-meta">${confidence}${citations ? `<span>${citations.replace('<div class="answer-meta">','').replace('</div>','')}</span>` : ''}</div>` : ''}`;
    $('askStatus').textContent = data.success ? 'Here’s what the weather suggests.' : 'The agent returned an incomplete result.';
  } catch (error) { $('askStatus').textContent = error.message; $('answer').innerHTML = '<div class="answer-label">Unable to answer</div><div class="answer-text">Try the question again in a moment.</div>'; }
  finally { button.disabled = false; button.textContent = 'Ask Agent'; }
}
function useQuestion(text) { $('question').value = text; $('question').focus(); }
$('alertDateFilter').addEventListener('change', event => { alertDateFilter = event.target.value; renderAlertList(); });
$('question').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); askAgent(); } });
renderSuggestions();
loadWeather();
