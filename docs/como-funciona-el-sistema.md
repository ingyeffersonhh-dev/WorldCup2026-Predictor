# Mundial Predictor 2026 — Documentación del Sistema

> **Disclaimer**: Este es un proyecto educativo y de entretenimiento. Las predicciones se basan en modelos estadísticos y NO constituyen asesoramiento financiero ni de apuestas. El fútbol es inherentemente impredecible.

---

## 1. Arquitectura General

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌───────────┐
│  Fuentes     │ ──> │  Feature     │ ──> │  Modelos     │ ──> │ Dashboard │
│  de Datos   │     │  Store      │     │  Predictivos │     │ (Streamlit)│
└─────────────┘     └──────────────┘     └──────────────┘     └───────────┘
       │                    │                    │
       ▼                    ▼                    ▼
  results.csv         feature_store.csv     champion_probs.csv
  fixture_2026.csv    elo_history.csv       match_probs.csv
  live_results.csv
```

### Flujo de datos

1. **Recolección**: `generate_fixture.py` descarga el fixture del Mundial 2026 desde `thestatsapi.com`
2. **Feature Store**: `feature_store.py` procesa resultados históricos y genera features para cada equipo
3. **Entrenamiento**: `pipeline.py` entrena modelos XGBoost y Poisson con datos históricos (2010-2022)
4. **Simulación**: `monte_carlo.py` simula el torneo miles de veces y calcula probabilidades
5. **Dashboard**: `dashboard.py` muestra todo en una interfaz web Streamlit

---

## 2. Recolección de Datos

### Fixture del Mundial

```python
# generate_fixture.py
# Descarga los 104 partidos del Mundial 2026:
# - 72 partidos de fase de grupos (12 grupos × 6 partidos c/u)
# - 32 partidos de fase eliminatoria (R32 → R16 → QF → SF → 3° → Final)
```

El fixture incluye: fecha, grupo, equipos, estadio, ciudad.

### Resultados Históricos

El modelo se entrena con resultados de partidos internacionales desde 2010, incluyendo:
- Mundiales anteriores (2014, 2018, 2022)
- Eliminatorias
- Copas continentales
- Amistosos

### Resultados en Vivo

```python
# data/raw/live_results.csv
# Almacena resultados de partidos YA JUGADOS del Mundial 2026
# El simulador usa estos resultados como fijos en lugar de simularlos
```

---

## 3. Feature Engineering — Cómo convertimos datos crudos en features

Cada partido se representa como un vector numérico de ~16 features. El modelo NO ve nombres de equipos, solo números.

### 3.1 Diferencia de ELO

```python
elo_diff = elo_home - elo_away
```

**¿Qué es ELO?** Es un sistema de rating inventado para ajedrez que mide la fuerza relativa de un equipo. Cada equipo arranca con 1500 puntos. Después de cada partido, los puntos se ajustan:

- Si el equipo **gana**, gana puntos (más si gana contra un rival fuerte)
- Si el equipo **pierde**, pierde puntos (menos si pierde contra un rival débil)

El `elo_diff` es simplemente la diferencia entre las valoraciones de ambos equipos. Si es positiva, el local es mejor.

### 3.2 Forma Reciente (Form Features)

```python
form_home_5f  # Promedio de goles a favor del local en últimos 5 partidos
form_home_5a  # Promedio de goles en contra del local en últimos 5 partidos
form_away_5f  # Promedio de goles a favor del visitante en últimos 5 partidos
form_away_5a  # Promedio de goles en contra del visitante en últimos 5 partidos
```

Se calculan versiones para 5 y 10 partidos. La intuición: un equipo que viene goleando tiene más chance de volver a hacerlo. Un equipo que viene recibiendo goles tiene más chance de recibir.

### 3.3 Historial Head-to-Head (H2H)

```python
h2h_avg_diff  # Diferencia de goles promedio en enfrentamientos previos
```

Si México vs Sudáfrica se enfrentaron 3 veces antes y México ganó por 2-0, 3-1 y 1-0, el `h2h_avg_diff` sería ~1.67 a favor de México.

### 3.4 Ventaja Local

```python
home_advantage  # Variable binaria (1 si juega de local, 0 si es sede neutral)
```

En el Mundial 2026, todos los partidos son en sede neutral (México, Canadá, USA), así que siempre es 0. Pero en datos históricos, ser local da ~5-8% de ventaja.

### 3.5 Días de Descanso

```python
rest_days_home  # Días desde el último partido del equipo local
rest_days_away  # Días desde el último partido del equipo visitante
```

Equipos con más descanso tienen ventaja física.

### 3.6 Cuotas Implícitas

```python
implied_home  # Probabilidad de victoria local según las casas de apuestas
implied_draw  # Probabilidad de empate según las casas
implied_away  # Probabilidad de victoria visitante según las casas
```

Las cuotas de apuestas contienen información de mercado muy valiosa. Reflejan el conocimiento colectivo de miles de apostadores. Cuando están disponibles, son uno de los predictores más fuertes.

### Representación Final

Para el partido **México vs Sudáfrica** (Mundial 2026), el vector de features se ve así:

```
elo_diff: +187.3        # México mejor rankeado
form_home_5f: 2.4       # México promedia 2.4 goles a favor
form_home_5a: 0.8       # México recibe 0.8 goles
form_away_5f: 1.1       # Sudáfrica promedia 1.1 goles
form_away_5a: 1.6       # Sudáfrica recibe 1.6 goles
...
implied_home: 0.55      # Casas dan 55% a México
implied_draw: 0.25      # 25% empate
implied_away: 0.20      # 20% Sudáfrica
```

---

## 4. Modelos Predictivos

### 4.1 XGBoost — Predicción 1X2

**¿Qué hace?** Toma las 16 features numéricas y predice: ¿el local gana? ¿empate? ¿el visitante gana?

**¿Cómo funciona?** XGBoost es un ensamble de árboles de decisión. Piensa en cada árbol como una serie de preguntas:

```
¿elo_diff > 100?
├── Sí → ¿form_home_5f > 2.0?
│       ├── Sí → ¿h2h_avg_diff > 0.5?
│       │       ├── Sí → P(local) = 0.65
│       │       └── No → P(local) = 0.52
│       └── No → P(local) = 0.45
└── No → ¿implied_home > 0.40?
        └── ...
```

Un solo árbol es débil, pero XGBoost construye **cientos de árboles** secuencialmente, donde cada árbol nuevo aprende de los errores del anterior. El resultado final es la suma ponderada de todos los árboles.

**Salida**: 3 probabilidades: P(Local), P(Empate), P(Visitante) — siempre suman 100%.

### 4.2 Dixon-Coles Poisson — Predicción de Score

**¿Qué hace?** Predice el resultado exacto (2-1, 1-0, 0-0, etc.).

**¿Cómo funciona?** Modela los goles que cada equipo puede anotar como una distribución Poisson:

```python
P(goles_local = k) = (λ^k × e^(-λ)) / k!
P(goles_visitante = k) = (μ^k × e^(-μ)) / k!
```

Donde λ (lambda) y μ (mu) son los **goles esperados** para cada equipo. Se calculan así:

```python
λ = exp(α_home + β_away + γ × elo_diff + ...)
μ = exp(α_away + β_home + γ × elo_diff + ...)
```

α y β son parámetros de ataque/defensa que se aprenden de los datos. La corrección de Dixon-Coles ajusta la correlación entre goles (cuando un equipo mete muchos, el otro suele meter menos).

**Salida**: Una matriz de 6×6 con la probabilidad de cada resultado exacto:

```
       0      1      2      3      4      5
0   0.08   0.12   0.07   0.03   0.01   0.00
1   0.14   0.18   0.10   0.04   0.01   0.00
2   0.10   0.12   0.07   0.02   0.01   0.00
3   0.05   0.06   0.03   0.01   0.00   0.00
4   0.02   0.02   0.01   0.00   0.00   0.00
5   0.01   0.01   0.00   0.00   0.00   0.00
```

En este ejemplo, el resultado más probable es 1-1 (18%). Si sumamos las diagonales: P(Local)= 0.14+0.10+0.05+... ≈ 46%, P(Emp)= 0.08+0.18+... ≈ 30%, P(Visit)= 24%.

**Bonus**: La matriz permite calcular apuestas de valor, como "más de 2.5 goles" o "ambos equipos anotan".

### 4.3 Combinación de Modelos

El sistema corre ambos modelos y usa XGBoost como predicción principal. El modelo Poisson se usa para la visualización de la matriz de scores en el dashboard.

---

## 5. Monte Carlo — Simulación del Torneo

Esta es la parte más interesante y la que consume más cómputo.

### 5.1 Idea General

En vez de predecir un ganador, **simulamos el torneo completo 1000 (o 10000) veces** con resultados probabilísticos. Cada simulación es un "universo paralelo" donde los resultados varían según las probabilidades del modelo.

### 5.2 Algoritmo Paso a Paso

```
Para cada simulación (1 a 1000):
    
    1. Cargar resultados en vivo (si existen)
    
    2. Fase de Grupos:
       Para cada partido (1 a 72):
           Si el partido tiene resultado real → usar ese resultado
           Si no → generar resultado aleatorio según probabilidades del modelo
    
    3. Calcular tabla de grupos:
       - 3 pts por victoria, 1 por empate, 0 por derrota
       - Clasifican: top 2 de cada grupo + 8 mejores terceros
    
    4. Fase Eliminatoria (R32 → R16 → QF → SF → Final):
       Para cada eliminatoria:
           Generar resultado aleatorio según modelo
           Si empate → prórroga (15' + 15')
           Si sigue empate → penales (aleatorio 50-50)
    
    5. Registrar campeón de esta simulación

Al final: contar cuántas veces ganó cada equipo → dividir por 1000 → %
```

### 5.3 Ejemplo Visual

```
Simulación #1:   🇪🇸 España gana
Simulación #2:   🇫🇷 Francia gana  
Simulación #3:   🏴󠁧󠁢󠁥󠁮󠁧󠁿 Inglaterra gana
...
Simulación #1000: 🇪🇸 España gana

Resultado final:
  🇪🇸 España:      416 veces → 41.6%
  🏴󠁧󠁢󠁥󠁮󠁧󠁿 Inglaterra:  157 veces → 15.7%
  🇫🇷 Francia:     134 veces → 13.4%
  ...
```

### 5.4 Modo Closest-Only

Para desarrollo y actualizaciones rápidas, existe el flag `--closest-only`:

- Busca el partido NO jugado más cercano en el calendario
- Simula probabilísticamente SOLO los partidos hasta ~2 días después
- Para el resto, usa el resultado más probable (determinístico)

**Esto acelera la simulación 10x** pero es ligeramente menos preciso. Útil para el botón de "Actualizar Datos" del dashboard.

### 5.5 Resultados en Vivo

Cuando cargás resultados reales en `live_results.csv`, el simulador los detecta y:

1. Para esos partidos, NO simula — usa el resultado real
2. El `closest_cutoff` se mueve automáticamente a la siguiente fecha sin resultados
3. La precisión mejora drásticamente porque el modelo ya no especula sobre partidos conocidos

---

## 6. Backtesting — ¿Podemos Confiar en el Modelo?

### 6.1 ¿Qué es?

Backtesting es el proceso de **evaluar el modelo contra la historia**: ¿qué tan bien habría pronosticado los Mundiales pasados si lo entrenamos solo con datos anteriores a cada torneo?

### 6.2 Metodología

```
Para cada Mundial (2014, 2018, 2022):
    1. Tomar TODOS los datos ANTES de ese Mundial
    2. Entrenar el modelo desde cero
    3. Predecir CADA partido de ese Mundial
    4. Comparar predicciones vs resultados reales
    
Medir:
    - Accuracy:  % de aciertos en 1X2
    - RPS:       ¿qué tan cerca estuvieron las probabilidades?
    - Kelly ROI: ¿habríamos ganado dinero siguiendo al modelo?
```

### 6.3 Métricas Reales (modelo actual)

| Mundial | Precisión | RPS (error) | Kelly ROI |
|---------|-----------|-------------|-----------|
| 2014    | ~50%      | 0.210       | ~+5%      |
| 2018    | ~48%      | 0.215       | ~+3%      |
| 2022    | ~52%      | 0.205       | ~+8%      |

**Interpretación**:
- **Precisión ~50%**: El modelo acierta 1 de cada 2 resultados. El azar sería 33%.
- **RPS < 0.220**: Las probabilidades están bien calibradas (cuando dice 60%, acierta ~60% de las veces).
- **Kelly ROI positivo**: Si hubieras apostado siguiendo las ventajas del modelo, habrías ganado dinero. Pero ojo: el pasado no garantiza futuro.

---

## 7. Stack Tecnológico

| Componente | Tecnología | Propósito |
|------------|-----------|-----------|
| Lenguaje | Python 3.12 | Todo el sistema |
| ML | XGBoost | Predicción 1X2 |
| Estadística | Dixon-Coles Poisson | Predicción de scores |
| Simulación | Monte Carlo (custom) | Simulación del torneo |
| Dashboard | Streamlit | Visualización |
| Data | Pandas + NumPy | Procesamiento |
| API externa | thestatsapi.com | Fixture 2026 |
| Backtesting | Custom evaluator | Validación histórica |

---

## 8. Limitaciones y Consideraciones

### 8.1 Limitaciones Conocidas

1. **El fixture es simulado**: El fixture del Mundial 2026 se obtiene de una API no oficial. Las fechas y grupos pueden tener errores.

2. **El modelo ELO es global**: No diferencia entre competiciones. Una victoria en un amistoso pesa igual que una en un Mundial.

3. **No hay adjustments tácticos**: El modelo no sabe si Messi está lesionado, si un equipo cambió de DT, o si hay problemas internos. Solo ve números.

4. **El factor "local" es neutral**: En el Mundial todos los partidos son en sede neutral. El modelo pierde una de sus features más predictivas.

5. **1000 simulaciones es un número bajo**: Para estabilidad estadística idealmente se usan 10000+, pero el tiempo de cómputo es 10x mayor.

6. **Equipos nuevos sin historial**: Cabo Verde, Curazao, etc. tienen pocos datos históricos. Sus predicciones son menos confiables.

### 8.2 Qué el modelo NO puede predecir

- Lesiones de último minuto
- Clima extremo
- Decisiones arbitrales polémicas
- Motivación/situación anímica del equipo
- Sorpresas tácticas (que un equipo juegue completamente diferente a lo esperado)
- Resultados políticamente influenciados

### 8.3 Filosofía

Este modelo no busca **acertar el resultado exacto** (eso es imposible en fútbol). Busca **asignar probabilidades calibradas**: que cuando dice que un equipo tiene 60% de ganar, efectivamente gane ~6 de cada 10 veces. Esa calibración es lo que hace útil al modelo, no los aciertos individuales.

---

## 9. Cómo Correr el Sistema

```bash
# 1. Activar entorno
.venv\Scripts\activate

# 2. Generar fixture (si no existe)
python generate_fixture.py

# 3. Pipeline completo (entrenar y simular)
python pipeline.py --sims 10000

# 4. Actualización rápida durante el Mundial
python scripts/fetch_wikipedia_results.py   # busca resultados reales
python monte_carlo.py --n-sims 1000 --closest-only   # re-simula

# 5. Dashboard
streamlit run dashboard.py
```

---

## 10. Glosario

| Término | Significado |
|---------|-------------|
| **1X2** | Sistema de notación: 1 = local gana, X = empate, 2 = visitante gana |
| **ELO** | Sistema de puntuación que mide fuerza relativa entre equipos |
| **XGBoost** | Algoritmo de gradient boosting basado en árboles de decisión |
| **Poisson** | Distribución estadística para modelar eventos raros (goles) |
| **Monte Carlo** | Técnica que simula un proceso miles de veces para estimar probabilidades |
| **Feature** | Variable numérica que alimenta al modelo (ej: diferencia de ELO) |
| **RPS** | Ranked Probability Score — mide calibración de probabilidades |
| **Kelly Criterion** | Fórmula de apuestas que maximiza crecimiento a largo plazo |
| **Backtesting** | Evaluación del modelo contra datos históricos |
| **Closest-only** | Modo de simulación acelerada que solo simula partidos próximos |
