"""Indian Weather Intelligence MCP server (MCP Python SDK v2)."""
from __future__ import annotations
from datetime import datetime
from typing import Any
import lakebase, weather_client
from mcp.server.mcpserver import MCPServer
from rag.pipeline import RAGPipeline
from observability import emit, span

mcp=MCPServer("indian-weather-intelligence")
_rag=RAGPipeline()

def _target_date(value:str|None,daily:dict[str,Any])->str:
    dates=daily.get("time") or []
    if not dates: raise ValueError("Forecast contains no dates")
    text=(value or "tomorrow").strip().lower()
    if text=="today": return dates[0]
    if text=="tomorrow": return dates[1] if len(dates)>1 else dates[0]
    try: return datetime.strptime(text,"%Y-%m-%d").date().isoformat()
    except ValueError as exc: raise ValueError("date must be 'today', 'tomorrow', or YYYY-MM-DD") from exc

def _daily_summary(weather:dict[str,Any],target:str)->dict[str,Any]:
    daily=weather.get("daily",{}); dates=daily.get("time") or []
    if target not in dates: raise ValueError(f"No forecast available for {target}")
    i=dates.index(target)
    def value(name:str,default:Any=None):
        values=daily.get(name) or []; return values[i] if i<len(values) else default
    code=value("weather_code")
    return {"date":target,"condition":weather_client.weather_description(code),"weather_code":code,"severity":weather_client.weather_severity(code),"temperature_min_c":value("temperature_2m_min"),"temperature_max_c":value("temperature_2m_max"),"apparent_temperature_max_c":value("apparent_temperature_max"),"precipitation_mm":value("precipitation_sum"),"rain_mm":value("rain_sum"),"precipitation_probability_pct":value("precipitation_probability_max"),"max_wind_kmh":value("wind_speed_10m_max"),"sunrise":value("sunrise"),"sunset":value("sunset")}

def _current_time_of_day(current_time:str|None,daily:dict[str,Any])->str:
    if not current_time:return "unknown"
    try:
        current=datetime.fromisoformat(current_time); dates=daily.get("time") or []; i=dates.index(current_time[:10]) if current_time[:10] in dates else 0; sunrise=(daily.get("sunrise") or [None])[i]; sunset=(daily.get("sunset") or [None])[i]
        if sunrise and sunset:return "day" if sunrise<=current_time<=sunset else "night"
    except (ValueError,IndexError,TypeError):pass
    return "unknown"

def _current_summary(current:dict[str,Any],daily:dict[str,Any],timezone:str|None)->dict[str,Any]:
    code=current.get("weather_code")
    return {"observation_time":current.get("time"),"timezone":timezone,"time_of_day":_current_time_of_day(current.get("time"),daily),"condition":weather_client.weather_description(code),"weather_code":code,"temperature_c":current.get("temperature_2m"),"apparent_temperature_c":current.get("apparent_temperature"),"relative_humidity_pct":current.get("relative_humidity_2m"),"cloud_cover_pct":current.get("cloud_cover"),"precipitation_mm":current.get("precipitation"),"wind_speed_kmh":current.get("wind_speed_10m"),"wind_direction_deg":current.get("wind_direction_10m")}

@mcp.tool()
def get_weather(location:str)->dict[str,Any]:
    """Get current weather and a 7-day forecast."""
    location=location.strip() if location else ""
    if not location:raise ValueError("location cannot be empty")
    details=weather_client.geocode_location_details(location)
    if not details:return {"success":False,"error":f"Could not resolve location: {location}"}
    weather=weather_client.fetch_weather(details["latitude"],details["longitude"]); current=weather.get("current",{}); daily=weather.get("daily",{})
    return {"success":True,"location":details,"timezone":weather.get("timezone"),"current":current,"current_summary":_current_summary(current,daily,weather.get("timezone")),"daily":daily,"forecast_summary":[_daily_summary(weather,d) for d in daily.get("time") or []]}

@mcp.tool()
def get_forecast(location:str,date:str="tomorrow")->dict[str,Any]:
    """Get a forecast for today, tomorrow, or YYYY-MM-DD."""
    details=weather_client.geocode_location_details(location.strip())
    if not details:return {"success":False,"error":f"Could not resolve location: {location}"}
    weather=weather_client.fetch_weather(details["latitude"],details["longitude"]); target=_target_date(date,weather.get("daily",{}))
    return {"success":True,"location":details,"forecast":_daily_summary(weather,target),"source":"open-meteo"}

@mcp.tool()
def get_weather_alerts(location:str)->dict[str,Any]:
    """Detect actionable hazards from the live 7-day forecast."""
    weather=get_weather(location)
    if not weather.get("success"):return weather
    daily=weather.get("daily",{}); alerts=[]
    for i,target in enumerate(daily.get("time",[])):
        probability=(daily.get("precipitation_probability_max") or [0])[i] if i<len(daily.get("precipitation_probability_max") or []) else 0; rain=(daily.get("precipitation_sum") or [0])[i] if i<len(daily.get("precipitation_sum") or []) else 0; wind=(daily.get("wind_speed_10m_max") or [0])[i] if i<len(daily.get("wind_speed_10m_max") or []) else 0; apparent=(daily.get("apparent_temperature_max") or [0])[i] if i<len(daily.get("apparent_temperature_max") or []) else 0; code=(daily.get("weather_code") or [0])[i] if i<len(daily.get("weather_code") or []) else 0
        if probability>=70 and rain>=15:alerts.append({"date":target,"severity":"HIGH","hazard":"Heavy rain","probability":probability,"details":f"{rain:.1f} mm expected with {probability}% precipitation probability."})
        elif probability>=70:alerts.append({"date":target,"severity":"MODERATE","hazard":"High rain probability","probability":probability,"details":f"Precipitation probability is {probability}%."})
        if wind>=40:alerts.append({"date":target,"severity":"HIGH","hazard":"Strong wind","details":f"Maximum forecast wind is {wind:.1f} km/h."})
        if apparent>=40:alerts.append({"date":target,"severity":"HIGH","hazard":"Extreme heat","details":f"Maximum apparent temperature is {apparent:.1f}°C."})
        elif apparent>=35:alerts.append({"date":target,"severity":"MODERATE","hazard":"High heat","details":f"Maximum apparent temperature is {apparent:.1f}°C."})
        if code>=95:alerts.append({"date":target,"severity":"HIGH","hazard":"Thunderstorm","details":f"Forecast weather code is {code}."})
    alerts.sort(key=lambda x:(0 if x["severity"]=="HIGH" else 1,x["date"]))
    return {"success":True,"location":weather["location"],"alerts":alerts,"alert_count":len(alerts),"highest_severity":alerts[0]["severity"] if alerts else "NONE","disclaimer":"Application-level forecast hazard detection; not an official government warning."}

@mcp.tool()
def assess_weather_risk(location:str,activity:str="outdoor activity",date:str="tomorrow")->dict[str,Any]:
    """Assess activity risk for a specific forecast date."""
    weather=get_weather(location)
    if not weather.get("success"):return weather
    target=_target_date(date,weather.get("daily",{})); f=_daily_summary(weather,target); rain_probability=f["precipitation_probability_pct"] or 0; precipitation=f["precipitation_mm"] or 0; wind=f["max_wind_kmh"] or 0; apparent=f["apparent_temperature_max_c"] or 0; code=f["weather_code"] or 0; score=0; factors=[]
    if rain_probability>=70:score+=2; factors.append(f"high precipitation probability ({rain_probability}%)")
    elif rain_probability>=40:score+=1; factors.append(f"moderate precipitation probability ({rain_probability}%)")
    if precipitation>=10:score+=2; factors.append(f"heavy expected precipitation ({precipitation:.1f} mm)")
    elif precipitation>=3:score+=1; factors.append(f"expected precipitation ({precipitation:.1f} mm)")
    if wind>=30:score+=2; factors.append(f"strong wind ({wind:.1f} km/h)")
    elif wind>=20:score+=1; factors.append(f"elevated wind ({wind:.1f} km/h)")
    if apparent>=40:score+=2; factors.append(f"very high apparent temperature ({apparent:.1f}°C)")
    elif apparent>=35:score+=1; factors.append(f"high apparent temperature ({apparent:.1f}°C)")
    if code>=95:score+=2; factors.append("thunderstorm risk in the forecast")
    risk="HIGH" if score>=5 else "MODERATE" if score>=2 else "LOW"; recommendation=f"Avoid or postpone the {activity} if possible." if risk=="HIGH" else f"The {activity} is possible with precautions and a backup plan." if risk=="MODERATE" else f"Conditions look generally suitable for the {activity}."
    return {"success":True,"location":weather["location"],"activity":activity,"date":target,"forecast":f,"risk_level":risk,"score":score,"recommendation":recommendation,"factors":factors or ["no major weather risk detected"]}

@mcp.tool()
def search_weather(query:str,top_k:int=5,location:str|None=None,state:str|None=None)->dict[str,Any]:
    """Search weather knowledge through the modular RAG pipeline."""
    trace_id="unknown"
    if not query or not query.strip():return {"success":False,"error":"query cannot be empty"}
    with span("mcp.search_weather",trace_id=trace_id,tool="search_weather") as info:
        result=_rag.retrieve(query.strip(),location=location,state=state); sources=result.sources; payload={"success":True,"query":query.strip(),"intent":result.plan.intent,"documents":result.documents,"context":result.context,"sources":sources}; info["success"]=True; info["documents"]=len(result.documents); info["sources"]=len(sources); emit("mcp.tool.result",trace_id=trace_id,tool="search_weather",success=True,documents=len(result.documents),sources=len(sources)); return payload

@mcp.tool()
def ask_weather(query:str,top_k:int=5,location:str|None=None,state:str|None=None)->dict[str,Any]:
    """Retrieve grounded weather knowledge evidence; generation is handled by the agent synthesizer."""
    return search_weather(query,top_k,location,state)

@mcp.tool()
def sync_weather(locations:list[str])->dict[str,Any]:
    """Fetch and store fresh weather data for Indian locations."""
    cleaned=[x.strip() for x in locations if x and x.strip()] if locations else []
    if not cleaned:raise ValueError("locations cannot be empty")
    return {"success":True,"locations":cleaned,"documents_synced":weather_client.sync_locations(cleaned)}

@mcp.tool()
def database_health()->dict[str,Any]:
    """Check PostgreSQL/Lakebase reachability."""
    try:
        connected=lakebase.check_connection(); return {"success":connected,"backend":lakebase.DATABASE_BACKEND,"status":"ok" if connected else "unavailable"}
    except Exception as exc:return {"success":False,"backend":lakebase.DATABASE_BACKEND,"status":"error","error":str(exc)}

if __name__=="__main__":mcp.run()
