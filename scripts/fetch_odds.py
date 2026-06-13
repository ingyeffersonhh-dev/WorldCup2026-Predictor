"""
fetch_odds.py

Descarga cuotas del Mundial 2026 desde The Odds API
y las integra en el fixture. Incluye horarios en Venezuela (UTC-4).

Uso:
    python scripts/fetch_odds.py

Requiere:
    - API key de the-odds-api.com (plan gratis: 500 requests/mes)
    - Guardar la key en .env como ODDS_API_KEY
"""
import json
import csv
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import urllib.request
    import urllib.error
except ImportError:
    print("ERROR: urllib no disponible")
    sys.exit(1)

# ── Configuración ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = PROJECT_ROOT / "data" / "raw" / "fixture_2026.csv"
ODDS_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "odds_2026.csv"
FIXTURE_WITH_ODDS_PATH = PROJECT_ROOT / "data" / "raw" / "fixture_2026_with_odds.csv"
API_BASE = "https://api.the-odds-api.com/v4/sports"
# Venezuela: UTC-4
VZ_TZ = timezone(timedelta(hours=-4))


def load_api_key() -> str:
    """Cargar API key desde .env o variables de entorno."""
    # Intentar variable de entorno
    key = os.environ.get("ODDS_API_KEY")
    if key:
        return key.strip()

    # Intentar archivo .env
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("ODDS_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")

    print("ERROR: No se encontró ODDS_API_KEY")
    print("Creá un archivo .env en la raíz del proyecto con:")
    print("  ODDS_API_KEY=tu_api_key_aqui")
    print()
    print("Obtené tu key gratis en: https://the-odds-api.com/")
    sys.exit(1)


def load_fixture() -> list:
    """Cargar fixture CSV."""
    with open(FIXTURE_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def utc_to_ve(utc_str: str) -> datetime:
    """Convertir string UTC a datetime en Venezuela (UTC-4)."""
    dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
    return dt.astimezone(VZ_TZ)


def american_to_implied(price: float) -> float:
    """Convertir odds americanas a probabilidad implícita."""
    if price > 0:
        return 100 / (price + 100)
    else:
        return abs(price) / (abs(price) + 100)


def avg_implied(outcomes: list, team_name: str) -> float:
    """Obtener la probabilidad implícita promedio de un equipo entre bookmakers."""
    for outcome in outcomes:
        if outcome["name"] == team_name:
            return american_to_implied(outcome["price"])
    return 1.0 / 3.0


def fetch_sport_odds(api_key: str, sport_key: str) -> list:
    """Obtener cuotas de un deporte específico."""
    url = (
        f"{API_BASE}/{sport_key}/odds/"
        f"?apiKey={api_key}"
        f"&regions=eu,uk"
        f"&markets=h2h"
        f"&oddsFormat=decimal"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        remaining = resp.headers.get("x-requests-remaining", "?")
        used = resp.headers.get("x-requests-used", "?")
        data = json.loads(resp.read().decode("utf-8"))
        print(f"  API response: {len(data)} events (remaining: {remaining}, used: {used})")
        return data
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print("  ERROR: API key inválida o expirada")
        elif e.code == 404:
            print(f"  Deporte '{sport_key}' no encontrado o fuera de temporada")
        else:
            print(f"  HTTP Error {e.code}: {e.reason}")
        return []
    except Exception as e:
        print(f"  Error: {e}")
        return []


def find_world_cup_sport(api_key: str) -> str:
    """Buscar el sport key correcto para el Mundial 2026."""
    url = f"{API_BASE}/?apiKey={api_key}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        sports = json.loads(resp.read().decode("utf-8"))

        # Buscar mundiales
        wc_sports = []
        for sport in sports:
            key = sport.get("key", "")
            title = sport.get("title", "")
            desc = sport.get("description", "")
            if any(term in key.lower() for term in ["world_cup", "world_cup_winner"]):
                wc_sports.append(sport)
            elif any(term in title.lower() for term in ["world cup"]):
                wc_sports.append(sport)
            elif any(term in desc.lower() for term in ["world cup"]):
                wc_sports.append(sport)

        if wc_sports:
            print("  Mundiales encontrados:")
            for ws in wc_sports:
                print(f"    {ws['key']}: {ws['title']} (outrights: {ws.get('has_outrights', False)})")
            # Preferir el de outright si existe, sino el de h2h
            outright = [ws for ws in wc_sports if ws.get("has_outrights")]
            h2h = [ws for ws in wc_sports if not ws.get("has_outrights")]
            if h2h:
                return h2h[0]["key"]
            elif outright:
                return outright[0]["key"]

        # Fallback: probar nombres comunes
        fallbacks = [
            "soccer_fifa_world_cup",
            "soccer_world_cup",
            "soccer_fifa_world_cup_winner",
        ]
        for fb in fallbacks:
            url2 = f"{API_BASE}/{fb}/odds/?apiKey={api_key}&regions=eu&markets=h2h&oddsFormat=decimal"
            try:
                req2 = urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0"})
                resp2 = urllib.request.urlopen(req2, timeout=10)
                remaining = resp2.headers.get("x-requests-remaining", "?")
                print(f"  Fallback '{fb}' funciona (remaining: {remaining})")
                # Consumir la respuesta pero no contar si está vacía
                return fb
            except urllib.error.HTTPError as e:
                if e.code != 429:
                    continue
            except Exception:
                continue

        print("  No se encontró deporte Mundial 2026 en la API")
        return None

    except Exception as e:
        print(f"  Error listando deportes: {e}")
        return None


def normalize_team(name: str) -> str:
    """Normalizar nombre de equipo entre The Odds API y fixture."""
    aliases = {
        "Czechia": "Czech Republic",
        "South Korea": "South Korea",
        "Korea Republic": "South Korea",
        "Republic of Korea": "South Korea",
        "USA": "United States",
        "United States of America": "United States",
        "Côte d'Ivoire": "Ivory Coast",
        "Cote d'Ivoire": "Ivory Coast",
        "Cabo Verde": "Cape Verde",
        "Cape Verde Islands": "Cape Verde",
        "Netherlands": "Netherlands",
        "Holland": "Netherlands",
        "Bosnia and Herzegovina": "Bosnia and Herzegovina",
        "Bosnia-Herzegovina": "Bosnia and Herzegovina",
        "Saudi Arabia": "Saudi Arabia",
    }
    return aliases.get(name.strip(), name.strip())


def merge_odds_with_fixture(fixture: list, odds_data: list) -> tuple:
    """Mergeear cuotas del API con el fixture CSV.
    Retorna (fixture_con_odds, odds_csv_rows)
    """
    fixture_teams = {}
    for row in fixture:
        key = (row["home_team"].strip(), row["away_team"].strip())
        fixture_teams[key] = row

    odds_rows = []
    matched = 0
    unmatched = []

    for event in odds_data:
        api_home = event.get("home_team", "").strip()
        api_away = event.get("away_team", "").strip()
        commence = event.get("commence_time", "")
        ve_time = utc_to_ve(commence) if commence else None

        # Buscar en el fixture
        fixture_key = None
        for (fh, fa) in fixture_teams:
            if normalize_team(api_home) == fh and normalize_team(api_away) == fa:
                fixture_key = (fh, fa)
                break

        # Intentar al revés (API puede invertir local/visitante)
        if fixture_key is None:
            for (fh, fa) in fixture_teams:
                if normalize_team(api_home) == fa and normalize_team(api_away) == fh:
                    fixture_key = (fh, fa)
                    break

        if fixture_key is None:
            unmatched.append(f"{api_home} vs {api_away}")
            continue

        # Promediar cuotas entre todos los bookmakers
        all_home_probs = []
        all_draw_probs = []
        all_away_probs = []

        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") == "h2h":
                    outcomes = market.get("outcomes", [])
                    home_prob = avg_implied(outcomes, normalize_team(api_home))
                    draw_prob = avg_implied(outcomes, "Draw")
                    away_prob = avg_implied(outcomes, normalize_team(api_away))
                    all_home_probs.append(home_prob)
                    all_draw_probs.append(draw_prob)
                    all_away_probs.append(away_prob)

        if all_home_probs:
            avg_home = sum(all_home_probs) / len(all_home_probs)
            avg_draw = sum(all_draw_probs) / len(all_draw_probs)
            avg_away = sum(all_away_probs) / len(all_away_probs)

            # Normalizar para que sumen 1
            total = avg_home + avg_draw + avg_away
            avg_home /= total
            avg_draw /= total
            avg_away /= total

            odds_rows.append({
                "home_team": fixture_key[0],
                "away_team": fixture_key[1],
                "implied_home": round(avg_home, 4),
                "implied_draw": round(avg_draw, 4),
                "implied_away": round(avg_away, 4),
                "n_bookmakers": len(all_home_probs),
                "commence_time_utc": commence,
                "commence_time_ve": ve_time.strftime("%d/%m %H:%M") if ve_time else "",
            })
            matched += 1

    print(f"\n  Matches: {matched} con cuotas, {len(unmatched)} sin cuota")
    if unmatched:
        print(f"  Sin cuota: {', '.join(unmatched[:5])}")
        if len(unmatched) > 5:
            print(f"    ... y {len(unmatched)-5} más")

    return odds_rows


def save_odds(odds_rows: list):
    """Guardar odds en CSV."""
    if not odds_rows:
        print("  No hay odds para guardar")
        return

    with open(ODDS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "home_team", "away_team", "implied_home", "implied_draw",
            "implied_away", "n_bookmakers", "commence_time_utc",
            "commence_time_ve"
        ])
        writer.writeheader()
        writer.writerows(odds_rows)
    print(f"  Guardado: {ODDS_CSV_PATH}")


def update_fixture_with_odds(fixture: list, odds_rows: list):
    """Agregar columnas de odds al fixture original."""
    odds_map = {}
    for row in odds_rows:
        odds_map[(row["home_team"], row["away_team"])] = row

    # Leer fixture original
    with open(FIXTURE_PATH, newline="", encoding="utf-8") as f:
        original = list(csv.DictReader(f))

    # Guardar fixture con odds
    fieldnames = list(original[0].keys()) + [
        "implied_home", "implied_draw", "implied_away",
        "commence_time_ve"
    ]

    with open(FIXTURE_WITH_ODDS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in original:
            key = (row["home_team"].strip(), row["away_team"].strip())
            new_row = dict(row)
            if key in odds_map:
                new_row["implied_home"] = odds_map[key]["implied_home"]
                new_row["implied_draw"] = odds_map[key]["implied_draw"]
                new_row["implied_away"] = odds_map[key]["implied_away"]
                new_row["commence_time_ve"] = odds_map[key]["commence_time_ve"]
            else:
                new_row["implied_home"] = ""
                new_row["implied_draw"] = ""
                new_row["implied_away"] = ""
                new_row["commence_time_ve"] = ""
            writer.writerow(new_row)

    print(f"  Guardado: {FIXTURE_WITH_ODDS_PATH}")


def main():
    print("=" * 60)
    print(f"Fetcher de Cuotas — Mundial 2026 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # 1. Cargar API key
    print("\n[1] Cargando API key...")
    api_key = load_api_key()
    print(f"  Key: {api_key[:8]}...{api_key[-4:]}")

    # 2. Buscar deporte Mundial
    print("\n[2] Buscando Mundial 2026 en la API...")
    sport_key = find_world_cup_sport(api_key)
    if not sport_key:
        print("  No se encontró el Mundial. La API puede no tener cobertura aún.")
        print("  Intentá de nuevo más cerca del torneo.")
        sys.exit(1)

    # 3. Descargar cuotas
    print(f"\n[3] Descargando cuotas de '{sport_key}'...")
    odds_data = fetch_sport_odds(api_key, sport_key)
    if not odds_data:
        print("  No hay cuotas disponibles. La API puede no tener cobertura aún.")
        sys.exit(1)

    # 4. Cargar fixture
    print("\n[4] Cargando fixture...")
    fixture = load_fixture()
    print(f"  {len(fixture)} partidos en fixture")

    # 5. Merge
    print("\n[5] Mergeando cuotas con fixture...")
    odds_rows = merge_odds_with_fixture(fixture, odds_data)

    # 6. Guardar
    print("\n[6] Guardando...")
    save_odds(odds_rows)
    update_fixture_with_odds(fixture, odds_rows)

    # 7. Resumen
    print(f"\n{'=' * 60}")
    print("RESUMEN:")
    print(f"  Partidos con cuota: {len(odds_rows)}")
    if odds_rows:
        print(f"\n  Primeros 10 partidos con cuota:")
        for row in odds_rows[:10]:
            ve_time = row.get("commence_time_ve", "")
            print(f"    {ve_time} | {row['home_team']} vs {row['away_team']}")
            print(f"              P(Local)={row['implied_home']:.1%} "
                  f"P(Emp)={row['implied_draw']:.1%} "
                  f"P(Visit)={row['implied_away']:.1%}")
    print(f"\n{'=' * 60}")
    print("Próximos pasos:")
    print("  1. Actualizar feature_store.py para usar implied_home/draw/away")
    print("     en vez de los default 33.3%")
    print("  2. Recorrer el modelo con las nuevas features")
    print("  3. Correr backtesting para medir impacto en Kelly ROI")


if __name__ == "__main__":
    main()
