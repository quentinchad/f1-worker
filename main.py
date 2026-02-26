"""
F1 Analytics - Python FastF1 Worker
FastAPI microservice called by the PHP backend
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import fastf1
import pandas as pd
import numpy as np
import requests
import json
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable FastF1 cache
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'fastf1_cache')
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# Désactiver ergast dans FastF1 — ergast.com est inaccessible sur Render
# et cause des timeouts inutiles lors du chargement des sessions
try:
    fastf1.ergast.interface.Ergast._get = lambda *a, **kw: None
except Exception:
    pass

# Configurer le niveau de log FastF1 pour réduire le bruit
import logging as _logging
_logging.getLogger('fastf1').setLevel(_logging.WARNING)

app = FastAPI(title="F1 Analytics Worker", version="1.0.0")


# Ajoute ce dict en haut de main.py après les imports
LOCATION_TO_CIRCUIT_KEY: dict = {
    'bahrain':          'bahrain',
    'jeddah':           'jeddah',
    'melbourne':        'albert_park',
    'suzuka':           'suzuka',
    'shanghai':         'shanghai',
    'miami':            'miami',
    'imola':            'imola',
    'monte-carlo':      'monaco',
    'montreal':         'villeneuve',
    'barcelona':        'catalunya',
    'spielberg':        'red_bull_ring',
    'silverstone':      'silverstone',
    'budapest':         'hungaroring',   # ← ton cas
    'spa-francorchamps':'spa',
    'zandvoort':        'zandvoort',
    'monza':            'monza',
    'baku':             'baku',
    'singapore':        'marina_bay',
    'austin':           'americas',
    'mexico city':      'rodriguez',
    'são paulo':        'interlagos',
    'las vegas':        'vegas',
    'lusail':           'losail',
    'abu dhabi':        'yas_marina',
    'portimão':         'portimao',
    'mugello':          'mugello',
    'sochi':            'sochi',
    'paul ricard':      'paul_ricard',
    'istanbul':         'istanbul',
    'nürburg':          'nurburgring',
    'hockenheim':       'hockenheimring',
    'magny-cours':      'magny_cours',
    'sepang':           'sepang',
    'valencia':         'valencia',
    'indianapolis':     'indianapolis',
}

# ─── Pydantic Models ─────────────────────────────────────────────────────────

class SeasonRequest(BaseModel):
    year: int

class RaceRequest(BaseModel):
    year: int
    round: int

class SessionRequest(BaseModel):
    year: int
    round: int
    session_type: str  # FP1, FP2, FP3, Q, R, S, SQ, SS

class DriverRequest(BaseModel):
    driver_id: str

class CircuitRequest(BaseModel):
    circuit_key: str

# ─── Helper Functions ─────────────────────────────────────────────────────────

def safe_val(val):
    """Convert numpy/pandas types to Python native."""
    if pd.isna(val) if hasattr(val, '__class__') and val.__class__.__name__ in ['float', 'float64'] else False:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return None if np.isnan(val) else float(val)
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    if isinstance(val, pd.Timedelta):
        total_ms = int(val.total_seconds() * 1000)
        return total_ms if total_ms > 0 else None
    return val


def timedelta_to_ms(td) -> Optional[int]:
    if pd.isna(td):
        return None
    try:
        return int(td.total_seconds() * 1000)
    except Exception:
        return None


def timedelta_to_str(td) -> Optional[str]:
    if pd.isna(td):
        return None
    try:
        total_s = td.total_seconds()
        minutes = int(total_s // 60)
        seconds = total_s % 60
        return f"{minutes}:{seconds:06.3f}"
    except Exception:
        return None


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.post("/load_seasons")
def load_seasons():
    """Return list of all available F1 seasons."""
    current_year = datetime.utcnow().year
    seasons = []
    # FastF1 supports from 2018 reliably; ergast for older
    for year in range(1950, current_year + 1):
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            seasons.append({
                "year": year,
                "total_rounds": len(schedule)
            })
        except Exception:
            # For very old seasons, just add them without round count
            if year >= 1950:
                seasons.append({"year": year, "total_rounds": 0})
    return {"seasons": seasons}


@app.post("/load_season")
def load_season(req: SeasonRequest):
    """Load full season calendar + championship standings."""
    year = req.year
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FastF1 error: {str(e)}")

    races = []
    for _, event in schedule.iterrows():
        raw_location = str(event.get('Location', '')).lower().strip()
        circuit_key  = LOCATION_TO_CIRCUIT_KEY.get(raw_location, raw_location.replace(' ', '_').replace('-', '_'))
        race_entry = {
            "year": year,
            "round": int(event.get('RoundNumber', 0)),
            "name": str(event.get('EventName', '')),
            "official_name": str(event.get('OfficialEventName', event.get('EventName', ''))),
            "circuit_key": circuit_key,
            "date": str(event.get('EventDate', ''))[:10] if event.get('EventDate') else None,
            "country": str(event.get('Country', '')),
            "sessions": _extract_event_sessions(event, year),
        }
        races.append(race_entry)

    # Championship standings via Ergast
    championship_drivers = _fetch_ergast_driver_standings(year)
    championship_constructors = _fetch_ergast_constructor_standings(year)

    return {
        "year": year,
        "races": races,
        "championship_drivers": championship_drivers,
        "championship_constructors": championship_constructors,
    }


def _extract_event_sessions(event, year: int) -> list:
    sessions = []
    session_map = {
        'Session1': 'FP1', 'Session2': 'FP2', 'Session3': 'FP3',
        'Session4': 'Q', 'Session5': 'R',
        'SprintQualifying': 'SQ', 'Sprint': 'S',
    }
    round_num = int(event.get('RoundNumber', 0))
    for col, stype in session_map.items():
        date_col = col + 'Date' if col in ['SprintQualifying', 'Sprint'] else col + 'Date'
        date_val = event.get(f'{col}Date') or event.get(col)
        if date_val and str(date_val) not in ('NaT', 'None', 'nan', ''):
            sessions.append({
                "year": year,
                "round": round_num,
                "type": stype,
                "name": _session_type_name(stype),
                "date": str(date_val)[:10] if date_val else None,
            })
    return sessions


def _session_type_name(stype: str) -> str:
    names = {'FP1': 'Practice 1', 'FP2': 'Practice 2', 'FP3': 'Practice 3',
             'Q': 'Qualifying', 'R': 'Race', 'S': 'Sprint', 'SQ': 'Sprint Qualifying', 'SS': 'Sprint Shootout'}
    return names.get(stype, stype)


def _fetch_ergast_driver_standings(year: int) -> list:
    try:
        url = f"https://ergast.com/api/f1/{year}/driverStandings.json"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        standings = data['MRData']['StandingsTable']['StandingsLists']
        if not standings:
            return []
        rows = []
        for s in standings[0]['DriverStandings']:
            rows.append({
                "year": year,
                "position": int(s['position']),
                "driver_id": s['Driver']['driverId'],
                "constructor_id": s['Constructors'][0]['constructorId'] if s['Constructors'] else None,
                "points": float(s['points']),
                "wins": int(s['wins']),
            })
        return rows
    except Exception as e:
        logger.warning(f"Ergast driver standings error {year}: {e}")
        return []


def _fetch_ergast_constructor_standings(year: int) -> list:
    try:
        url = f"https://ergast.com/api/f1/{year}/constructorStandings.json"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        standings = data['MRData']['StandingsTable']['StandingsLists']
        if not standings:
            return []
        rows = []
        for s in standings[0]['ConstructorStandings']:
            rows.append({
                "year": year,
                "position": int(s['position']),
                "constructor_id": s['Constructor']['constructorId'],
                "points": float(s['points']),
                "wins": int(s['wins']),
            })
        return rows
    except Exception as e:
        logger.warning(f"Ergast constructor standings error {year}: {e}")
        return []


@app.post("/load_race")
def load_race(req: RaceRequest):
    """Load race event info and sessions list."""
    try:
        event = fastf1.get_event(req.year, req.round)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FastF1 event error: {str(e)}")

    raw_location = str(event.get('Location', '')).lower().strip()
    circuit_key  = LOCATION_TO_CIRCUIT_KEY.get(raw_location, raw_location.replace(' ', '_').replace('-', '_'))
    sessions = _extract_event_sessions(event, req.year)

    race = {
        "year": req.year,
        "round": req.round,
        "name": str(event.get('EventName', '')),
        "official_name": str(event.get('OfficialEventName', '')),
        "circuit_key": circuit_key,
        "date": str(event.get('EventDate', ''))[:10] if event.get('EventDate') else None,
        "country": str(event.get('Country', '')),
    }
    return {"race": race, "sessions": sessions}


@app.post("/debug_session")
def debug_session(req: SessionRequest):
    """Debug: retourne les colonnes et la première ligne de session.results."""
    try:
        session = fastf1.get_session(req.year, req.round, req.session_type)
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        df = session.results
        if df is None or df.empty:
            return {"error": "results empty", "columns": []}
        first_row = {k: str(v) for k, v in df.iloc[0].to_dict().items()}
        return {
            "columns": list(df.columns),
            "first_row": first_row,
            "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/load_session")
def load_session(req: SessionRequest):
    """Load full session data: results, laps, strategy."""
    try:
        session = fastf1.get_session(req.year, req.round, req.session_type)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FastF1 session load error: {str(e)}")

    results = _extract_results(session, req.session_type)
    laps_data = _extract_laps(session)
    strategy = _extract_strategy(laps_data)

    return {
        "results": results,
        "laps": laps_data[:5000],  # cap at 5k rows
        "strategy": strategy,
    }


def _safe_driver_id(row) -> Optional[str]:
    """Extract a clean driver_id. Essaie plusieurs colonnes FastF1."""
    # Abbreviation est toujours présent (VER, HAM...) → meilleur identifiant
    for col in ('Abbreviation', 'DriverId', 'BroadcastName', 'Driver'):
        val = row.get(col)
        if val is None:
            continue
        s = str(val).strip().lower()
        if s and s not in ('nan', 'none', 'nat', ''):
            # BroadcastName peut être "Max Verstappen" → on ne veut pas ça
            if col == 'BroadcastName' and ' ' in s:
                continue
            return s
    # Dernier recours : numéro de pilote
    for col in ('DriverNumber', 'CarNumber'):
        val = row.get(col)
        if val is not None:
            s = str(val).strip()
            if s and s not in ('nan', 'none', ''):
                return f"driver_{s}"
    return None


def _safe_constructor_id(row) -> Optional[str]:
    """Extract a clean constructor_id. FastF1 2025 uses 'TeamId' or 'TeamName'."""
    for col in ('TeamId', 'ConstructorId', 'TeamName', 'Constructor'):
        val = row.get(col)
        if val is None:
            continue
        s = str(val).strip().lower().replace(' ', '_')
        if s and s not in ('nan', 'none', 'nat', ''):
            return s
    return None


def _safe_position(row) -> Optional[int]:
    """Try multiple FastF1 columns for finishing position."""
    NON_CLASSIFIED = {'nan', 'none', 'nat', '', 'nc', 'dsq', 'dnf', 'dns', 'dq', 'ret', 'wd', 'ex'}
    for col in ('Position', 'ClassifiedPosition', 'GridPosition', 'FinishingPosition'):
        val = row.get(col)
        if val is None:
            continue
        s = str(val).strip().lower()
        if s in NON_CLASSIFIED:
            continue
        try:
            pos = int(float(s))
            if pos > 0:
                return pos
        except (ValueError, TypeError):
            continue
    return None


def _extract_results(session, session_type: str) -> list:
    results = []
    try:
        df = session.results
        if df is None or df.empty:
            logger.warning("session.results is None or empty")
            return []

        # Log les colonnes disponibles pour debug
        logger.info(f"session.results columns: {list(df.columns)}")
        logger.info(f"session.results first row: {df.iloc[0].to_dict() if len(df) > 0 else 'empty'}")

        for _, row in df.iterrows():
            driver_id = _safe_driver_id(row)
            if not driver_id:
                logger.warning(f"Skipping result row with no driver_id: {dict(row)}")
                continue

            # Position : essayer aussi de la déduire depuis l'index si tout échoue
            position = _safe_position(row)

            r = {
                "position":       position,
                "driver_id":      driver_id,
                "driver_code":    str(row.get('Abbreviation', row.get('Driver', ''))).upper(),
                "first_name":     str(row.get('FirstName', '')),
                "last_name":      str(row.get('LastName', row.get('FullName', ''))),
                "constructor_id": _safe_constructor_id(row),
                "constructor_name": str(row.get('TeamName', '')),
                "laps_completed": safe_val(row.get('NumberOfLaps')),
                "status":         str(row.get('Status', '')),
                "points":         safe_val(row.get('Points', 0)),
                "fastest_lap":    bool(row.get('FastestLap', False)),
                "gap_to_leader":  str(row.get('Time', row.get('TimeDelta', ''))) if row.get('Time') or row.get('TimeDelta') else None,
                "best_lap_time":  timedelta_to_str(row.get('FastestLapTime')),
            }
            if session_type in ('Q', 'SQ', 'SS', 'Sprint Qualifying', 'Sprint Shootout'):
                r['q1_time'] = timedelta_to_str(row.get('Q1'))
                r['q2_time'] = timedelta_to_str(row.get('Q2'))
                r['q3_time'] = timedelta_to_str(row.get('Q3'))
            results.append(r)
    except Exception as e:
        logger.warning(f"Results extraction error: {e}", exc_info=True)
    return results


def _extract_laps(session) -> list:
    laps = []
    try:
        df = session.laps
        if df is None or df.empty:
            return []
        for _, row in df.iterrows():
            driver_id = _safe_driver_id(row)
            if not driver_id:
                continue
            lap = {
                "driver_id":       driver_id,
                "driver_code":     str(row.get('Driver', '')),
                "lap_number":      safe_val(row.get('LapNumber')),
                "lap_time_ms":     timedelta_to_ms(row.get('LapTime')),
                "sector1_ms":      timedelta_to_ms(row.get('Sector1Time')),
                "sector2_ms":      timedelta_to_ms(row.get('Sector2Time')),
                "sector3_ms":      timedelta_to_ms(row.get('Sector3Time')),
                "compound":        str(row.get('Compound', '')).upper() if row.get('Compound') else None,
                "tyre_life":       safe_val(row.get('TyreLife')),
                "stint":           safe_val(row.get('Stint')),
                "is_personal_best": bool(row.get('IsPersonalBest', False)),
                "speed_trap":      safe_val(row.get('SpeedFL')),
            }
            laps.append(lap)
    except Exception as e:
        logger.warning(f"Laps extraction error: {e}")
    return laps


def _extract_strategy(laps: list) -> list:
    """Build stint strategy from lap data."""
    strategy = []
    driver_laps: dict = {}
    for lap in laps:
        driver = lap['driver_id']
        if driver not in driver_laps:
            driver_laps[driver] = []
        driver_laps[driver].append(lap)

    for driver_id, dlaps in driver_laps.items():
        dlaps_sorted = sorted(dlaps, key=lambda x: x.get('lap_number') or 0)
        stint_num = 1
        current_compound = None
        stint_start = 1
        code = dlaps_sorted[0].get('driver_code', '') if dlaps_sorted else ''

        for lap in dlaps_sorted:
            compound = lap.get('compound') or 'UNKNOWN'
            lap_num = lap.get('lap_number') or 0
            if compound != current_compound:
                if current_compound is not None:
                    strategy.append({
                        "driver_id": driver_id,
                        "driver_code": code,
                        "stint_number": stint_num,
                        "compound": current_compound,
                        "start_lap": stint_start,
                        "end_lap": lap_num - 1,
                        "laps_on_tyre": lap_num - stint_start,
                    })
                    stint_num += 1
                current_compound = compound
                stint_start = lap_num

        if current_compound:
            last_lap = dlaps_sorted[-1].get('lap_number') or stint_start
            strategy.append({
                "driver_id": driver_id,
                "driver_code": code,
                "stint_number": stint_num,
                "compound": current_compound,
                "start_lap": stint_start,
                "end_lap": last_lap,
                "laps_on_tyre": last_lap - stint_start + 1,
            })
    return strategy


@app.post("/load_drivers")
def load_drivers():
    """Return driver metadata via Ergast."""
    drivers = _fetch_ergast_drivers()
    return {"drivers": drivers}


@app.post("/load_driver")
def load_driver(req: DriverRequest):
    """Return specific driver info."""
    drivers = _fetch_ergast_drivers(driver_id=req.driver_id)
    if not drivers:
        raise HTTPException(status_code=404, detail="Driver not found")
    return drivers[0]


def _fetch_ergast_drivers(driver_id: Optional[str] = None) -> list:
    try:
        url = f"https://ergast.com/api/f1/drivers/{driver_id if driver_id else ''}.json?limit=1000"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        raw = data['MRData']['DriverTable']['Drivers']
        drivers = []
        for d in raw:
            drivers.append({
                "driver_id": d['driverId'],
                "code": d.get('code'),
                "number": int(d['permanentNumber']) if d.get('permanentNumber') else None,
                "first_name": d.get('givenName'),
                "last_name": d.get('familyName'),
                "nationality": d.get('nationality'),
                "dob": d.get('dateOfBirth'),
                "photo_url": f"https://www.formula1.com/content/dam/fom-website/drivers/{d.get('givenName', [''])[0].upper()}/{d.get('driverId', '').upper()[:6]}01__{d.get('givenName','').lower()}-{d.get('familyName','').lower()}/driver-career/{d.get('driverId','')}.png",
            })
        return drivers
    except Exception as e:
        logger.warning(f"Ergast drivers error: {e}")
        return []


@app.post("/load_circuits")
def load_circuits():
    circuits = _fetch_ergast_circuits()
    return {"circuits": circuits}


@app.post("/load_circuit")
def load_circuit(req: CircuitRequest):
    circuits = _fetch_ergast_circuits(circuit_id=req.circuit_key)
    if not circuits:
        raise HTTPException(status_code=404, detail="Circuit not found")
    return circuits[0]


def _fetch_ergast_circuits(circuit_id: Optional[str] = None) -> list:
    try:
        path = f"circuits/{circuit_id}" if circuit_id else "circuits"
        url = f"https://ergast.com/api/f1/{path}.json?limit=200"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        raw = data['MRData']['CircuitTable']['Circuits']
        circuits = []
        for c in raw:
            key = c['circuitId']
            circuits.append({
                "circuit_key": key,
                "name": c.get('circuitName'),
                "short_name": c.get('circuitName', '').split('Grand Prix')[0].strip(),
                "country": c.get('Location', {}).get('country'),
                "city": c.get('Location', {}).get('locality'),
                "latitude": float(c.get('Location', {}).get('lat', 0)),
                "longitude": float(c.get('Location', {}).get('long', 0)),
                "image_url": f"https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Circuit%20maps%2016x9/{key.replace('_', '-')}.png",
                "svg_url": f"https://www.formula1.com/content/dam/fom-website/2018-redesign-assets/Racecourse%20illustrations/{key.replace('_', '-')}.png",
            })
        return circuits
    except Exception as e:
        logger.warning(f"Ergast circuits error: {e}")
        return []


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")