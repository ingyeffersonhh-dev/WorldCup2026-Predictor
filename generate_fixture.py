# -*- coding: utf-8 -*-
import json
import urllib.request
import pandas as pd
from pathlib import Path

# Official team name mapping to ELO history names
mapping = {
    'Congo DR': 'DR Congo',
    'Curacao': 'Curaçao',
    'IR Iran': 'Iran',
    'Korea Republic': 'South Korea',
    "Cote d'Ivoire": 'Ivory Coast',
    'Czechia': 'Czech Republic',
    'Turkiye': 'Turkey',
    'Cabo Verde': 'Cape Verde'
}

url = "https://www.thestatsapi.com/world-cup/data/fixtures.json"
req = urllib.request.Request(
    url, 
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
)

matches = []
fetched_online = False

try:
    print(f"Intento de descarga de calendario oficial desde {url}...")
    response = urllib.request.urlopen(req, timeout=10)
    data = json.loads(response.read().decode('utf-8'))
    raw_fixtures = data['fixtures']
    group_fixtures = [m for m in raw_fixtures if m['stage'] == 'group-stage']
    
    # Ordenar cronológicamente por fecha, hora de inicio UTC y número de partido
    group_fixtures.sort(key=lambda x: (x.get('date', ''), x.get('kickoffUtc', ''), x.get('matchNumber', 0)))
    
    for idx, f in enumerate(group_fixtures, 1):
        home = f['homeTeam']
        away = f['awayTeam']
        
        # Mapear nombres para ELO
        home = mapping.get(home, home)
        away = mapping.get(away, away)
        
        matches.append({
            'match_id': idx,
            'group': f['group'],
            'round': 'group',
            'date': f['date'],
            'home_team': home,
            'away_team': away,
            'neutral_venue': 1
        })
    fetched_online = True
    print("Calendario descargado y mapeado correctamente.")
    
    # Guardar copia de seguridad local en JSON
    with open('data/raw/fixture_2026_source.json', 'w', encoding='utf-8') as f:
        json.dump(matches, f, indent=2, ensure_ascii=False)
except Exception as e:
    print(f"No se pudo descargar el calendario online: {e}")
    print("Cargando desde copia de seguridad local (fixture_2026_source.json)...")
    
    # Fallback to local JSON source
    source_path = Path('data/raw/fixture_2026_source.json')
    if source_path.exists():
        with open(source_path, 'r', encoding='utf-8') as f:
            matches = json.load(f)
        print("Cargado correctamente desde archivo local.")
    else:
        print("ERROR: No se encontró la copia de seguridad local de fixtures.")
        raise FileNotFoundError("data/raw/fixture_2026_source.json no encontrado.")

if matches:
    df = pd.DataFrame(matches)
    df.to_csv('data/raw/fixture_2026.csv', index=False, encoding='utf-8')
    print(f"Creado fixture_2026.csv con {len(df)} partidos en orden oficial.")
