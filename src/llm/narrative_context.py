"""
NarrativeContext — adaptador determinístico entre o pipeline ML e a LLM.

Transforma os 3 JSONs Pydantic (race_summary, timeline, driver_profiles) em uma
representação compacta e otimizada para geração de narrativa.

Objetivos:
  - Reduzir tokens de input ~60% (sem indent, sem campos ML-internos)
  - Adicionar nomes completos dos pilotos junto com os códigos
  - Suprimir external_incidents durante safety car (SC já explica as voltas lentas)
  - Agrupar external_incidents concorrentes (mesmo lap) em multi_car_incident APENAS
    quando NÃO há safety car na mesma volta (evita narrativas de acidentes fictícios)
  - Traduzir compostos de pneu para português (nunca HARD/MEDIUM/SOFT em inglês)
  - Tornar pace profile legível: "ataque 45% | base 50% | desgaste 5%"
  - Garantir que o LLM nunca receba campos ML-internos sem significado narrativo
"""

import json
import re

# ---------------------------------------------------------------------------
# Mapeamento de códigos de piloto → nome completo (temporada 2025)
# ---------------------------------------------------------------------------

# Glossário de termos de penalidade F1 (API FastF1 EN → PT jornalístico)
# Aplicado deterministicamente na entrada, antes da LLM ver o texto.
# Evita traduções literais como "Unsafe Release" → "lançamento não seguro".
_PENALTY_REASONS_PT: dict[str, str] = {
    "Unsafe Release": "saída insegura do boxe",
    "Track Limits": "ultrapassagem dos limites de pista",
    "Speeding In Pit Lane": "excesso de velocidade no boxe",
    "Speeding in the Pit Lane": "excesso de velocidade no boxe",
    "Causing A Collision": "causar colisão",
    "Causing a Collision": "causar colisão",
    "Collision": "colisão",
    "Ignoring Blue Flags": "ignorar bandeiras azuis",
    "Crossing The White Line": "cruzar a linha branca",
    "Crossing the White Line": "cruzar a linha branca",
    "Driving Unnecessarily Slowly": "conduzir desnecessariamente devagar",
    "Blocking": "bloqueio em pista",
}


def _translate_penalty_reason(reason: str) -> str:
    """Returns the Portuguese journalistic equivalent of an F1 penalty reason.
    Falls back to the original string if not found in the glossary.
    """
    return _PENALTY_REASONS_PT.get(reason, reason)

_DRIVER_NAMES: dict[str, str] = {
    "NOR": "Lando Norris",
    "VER": "Max Verstappen",
    "RUS": "George Russell",
    "HAM": "Lewis Hamilton",
    "LEC": "Charles Leclerc",
    "SAI": "Carlos Sainz",
    "PIA": "Oscar Piastri",
    "ALO": "Fernando Alonso",
    "STR": "Lance Stroll",
    "GAS": "Pierre Gasly",
    "ANT": "Andrea Kimi Antonelli",
    "TSU": "Yuki Tsunoda",
    "ALB": "Alexander Albon",
    "HUL": "Nico Hülkenberg",
    "OCO": "Esteban Ocon",
    "BEA": "Oliver Bearman",
    "LAW": "Liam Lawson",
    "BOR": "Gabriel Bortoleto",
    "DOO": "Jack Doohan",
    "HAD": "Isack Hadjar",
}


def _name(code: str) -> str:
    """Returns 'Full Name (CODE)' or just 'CODE' if not in mapping."""
    full = _DRIVER_NAMES.get(code.upper())
    return f"{full} ({code})" if full else code


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_narrative_context(
    race_summary: dict,
    timeline: list,
    driver_profiles: list,
    race_control: list | None = None,
    driver_quality_scores: dict | None = None,
) -> str:
    """
    Constrói o contexto narrativo compacto para envio à LLM.

    Args:
        race_summary:          Conteúdo de race_summary.json (dict).
        timeline:              Conteúdo de timeline.json (list de eventos).
        driver_profiles:       Conteúdo de driver_profiles.json (list de perfis).
        race_control:          Lista de mensagens do race_control (opcional).
                               ML-C: usado para validar incidente_multiplo vs INCIDENT/COLLISION.
        driver_quality_scores: Dict {driver_code: quality_score} (opcional).
                               ML-07: informa à LLM a confiabilidade dos dados por piloto.

    Returns:
        String JSON minificada pronta para ser inserida no prompt da LLM.
        Tipicamente 60-70% menor em tokens do que os 3 JSONs indentados originais.
    """
    context = {
        "race": _build_race_section(race_summary),
        "drivers": _build_drivers_section(driver_profiles, driver_quality_scores=driver_quality_scores),
        "must_cover": _build_must_cover(timeline, race_summary),
    }
    # Minificado: sem espaços extras, sem indentação
    return json.dumps(context, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Seção: informações gerais da corrida
# ---------------------------------------------------------------------------

def _build_race_section(rs: dict) -> dict:
    winner_code = rs.get("winner", "")
    fastest_driver = rs.get("fastest_lap_driver", "")
    fastest_time = rs.get("fastest_lap_time", 0)

    podium = [
        {
            "pos": e["position"],
            "driver": _name(e["driver"]),
            "team": e["team"],
            "gap": e["gap_to_leader"],
        }
        for e in rs.get("podium", [])
    ]

    w = rs.get("weather", {})
    weather_str = (
        f"{w.get('condition','unknown')}"
        f", {'chuva' if w.get('had_rainfall') else 'sem chuva'}"
        f", ar {w.get('air_temp_avg_c','?')}°C"
        f", pista {w.get('track_temp_avg_c','?')}°C"
    )

    return {
        "winner": _name(winner_code),
        "podium": podium,
        "total_laps": rs.get("total_laps"),
        "safety_cars": rs.get("safety_car_count"),
        "fastest_lap": f"{_name(fastest_driver)} — {fastest_time}s",
        "weather": weather_str,
    }


# ---------------------------------------------------------------------------
# Seção: cronologia de eventos
# ---------------------------------------------------------------------------

def _build_events_section(timeline: list, race_control: list | None = None) -> list:
    """
    Processa a timeline:
      - Adiciona nomes completos em todos os campos de driver
      - Suprime external_incidents durante safety car (evita narrativas de acidentes fictícios)
      - ML-C: agrupa external_incidents concorrentes em incidente_multiplo APENAS quando
        o race_control confirma INCIDENT/COLLISION/CONTACT no mesmo lap.
        Sem confirmação → incidentes individuais (incidente_externo).
      - Mantém outros eventos individuais
    """
    # Construir conjunto de laps com incidente confirmado no race_control (ML-C)
    confirmed_incident_laps: set[int] = set()
    if race_control:
        _INCIDENT_PATTERN = re.compile(
            r"INCIDENT|COLLISION|CONTACT|ACCIDENT|CRASH",
            re.IGNORECASE,
        )
        for msg_entry in race_control:
            if isinstance(msg_entry, dict):
                msg = str(msg_entry.get("Message", ""))
                lap = msg_entry.get("Lap")
                if lap is not None and _INCIDENT_PATTERN.search(msg):
                    try:
                        confirmed_incident_laps.add(int(lap))
                    except (ValueError, TypeError):
                        pass

    # Separar external_incidents dos demais
    ext_by_lap: dict[int, list[str]] = {}
    other_events: list[dict] = []

    for evt in timeline:
        t = evt.get("type", "")
        if t == "external_incident":
            lap = evt.get("lap", 0)
            ext_by_lap.setdefault(lap, []).append(evt.get("driver", ""))
        else:
            other_events.append(evt)

    # Voltas com safety car ativo — external_incidents nessas voltas são efeito do SC,
    # não causa. Suprimi-los evita narrativas de "grande acidente" fictícias.
    sc_laps: set[int] = set()
    for e in other_events:
        if e.get("type") == "safety_car":
            start = e.get("deployed_on_lap") or e.get("lap", -1)
            duration = e.get("duration_laps", 1)
            # Inclui a volta ANTES do SC (start-1) porque o incidente que causa o SC
            # normalmente gera bandeiras amarelas na volta anterior ao deploy oficial.
            sc_laps.update(range(max(1, start - 1), start + duration + 1))

    processed: list[dict] = []

    # Processar eventos não-agrupados
    for evt in other_events:
        processed.append(_format_event(evt))

    # Agrupar external_incidents por lap
    for lap, drivers in ext_by_lap.items():
        if lap in sc_laps:
            # Sob safety car → voltas lentas são esperadas, não narrar como incidente
            continue

        if len(drivers) == 1:
            processed.append({
                "lap": lap,
                "type": "incidente_externo",
                "piloto": _name(drivers[0]),
            })
        else:
            # ML-C: só gerar incidente_multiplo se race_control confirmar colisão/incidente
            # Sem confirmação → incidentes individuais (evita narrativas de acidentes fictícios)
            if race_control is not None and lap not in confirmed_incident_laps:
                # Sem confirmação → tratar como incidentes externos individuais
                for d in drivers:
                    processed.append({
                        "lap": lap,
                        "type": "incidente_externo",
                        "piloto": _name(d),
                    })
            else:
                # Confirmado no race_control (ou sem race_control disponível como fallback)
                processed.append({
                    "lap": lap,
                    "type": "incidente_multiplo",
                    "pilotos": [_name(d) for d in drivers],
                })

    # Ordenar cronologicamente
    processed.sort(key=lambda e: e.get("lap", 0))
    return processed


def _format_event(evt: dict) -> dict:
    """Formata um evento individual adicionando nomes completos."""
    t = evt.get("type", "")

    if t in ("driver_error", "retirement"):
        return {
            "lap": evt.get("lap"),
            "type": "erro_piloto" if t == "driver_error" else "abandono",
            "piloto": _name(evt.get("driver", "")),
            **( {"causa": evt["cause"]} if evt.get("cause") else {} ),
        }

    if t == "tire_dropoff":
        # Não expor drop_seconds (métrica ML interna) — mencionar apenas quando validado
        out = {
            "lap": evt.get("lap"),
            "type": "queda_de_pneu",
            "piloto": _name(evt.get("driver", "")),
            "confirmado": evt.get("cliff_validated", False),
        }
        if evt.get("positions_lost"):
            out["posicoes_perdidas"] = evt["positions_lost"]
        return out

    if t == "undercut":
        return {
            "lap": evt.get("lap"),
            "type": "undercut",
            "vencedor": _name(evt.get("winner", "")),
            "perdedor": _name(evt.get("loser", "")),
            "ganho_s": round(evt.get("time_gained_seconds", 0), 2),
        }

    if t == "overcut":
        return {
            "lap": evt.get("lap"),
            "type": "overcut",
            "vencedor": _name(evt.get("winner", "")),
            "perdedor": _name(evt.get("loser", "")),
            "vantagem_s": round(evt.get("time_saved_seconds", 0), 2),
        }

    if t == "safety_car":
        out = {
            "lap": evt.get("lap"),
            "type": "safety_car",
            "duracao_voltas": evt.get("duration_laps"),
        }
        if evt.get("cause"):
            out["causa"] = evt["cause"]
        return out

    if t == "penalty":
        return {
            "lap": evt.get("lap"),
            "type": "penalidade",
            "piloto": _name(evt.get("driver", "")),
            "motivo": _translate_penalty_reason(evt.get("reason", "")),
        }

    # Fallback: retorna o evento como está
    return evt


# ---------------------------------------------------------------------------
# Seção: perfis de pilotos (somente campos narrativos)
# ---------------------------------------------------------------------------

def _build_drivers_section(
    profiles: list,
    driver_quality_scores: dict | None = None,
) -> list:
    """
    Transforma driver_profiles em representação compacta para narrativa.

    Remove campos ML-internos (cliff_count, anomaly_count bruto quando zero,
    percentuais como floats crus) e formata de forma legível.

    ML-07: driver_quality_scores indica a confiabilidade dos dados por piloto.
    Score < 0.5 → dados fragmentados (DNF precoce, muitas anomalias) → LLM
    deve ser mais conservadora nas inferências sobre esse piloto.
    """
    result = []
    for p in profiles:
        driver_code = p.get("driver", "")
        compounds = _format_compounds(p.get("compounds_used", []))

        entry: dict = {
            "driver": _name(driver_code),
            "team": p.get("team", ""),
            "grid": p.get("grid_position"),
            "finish": p.get("final_position"),  # None se DNF
            "delta": p.get("positions_gained", 0),
            "compounds": compounds,
        }

        if p.get("had_tire_cliff"):
            entry["queda_abrupta_pneu"] = True

        # ML-07: qualidade dos dados do piloto
        if driver_quality_scores:
            score = driver_quality_scores.get(driver_code)
            if score is not None and score < 0.5:
                entry["dados_limitados"] = True  # DNF precoce ou poucos laps analisados

        result.append(entry)

    return result


# Mapeamento de compostos para português (nunca expor HARD/MEDIUM/SOFT à LLM)
_COMPOUND_PT: dict[str, str] = {
    "SOFT": "pneus macios",
    "MEDIUM": "pneus médios",
    "HARD": "pneus duros",
    "INTERMEDIATE": "pneus intermediários",
    "WET": "pneus de chuva",
}


def _format_compounds(compounds_used: list) -> str:
    """
    Formata o uso de compostos em português.

    Ex: [{"compound":"HARD","laps":10},{"compound":"INTERMEDIATE","laps":47}]
        → "pneus duros (10 voltas) → pneus intermediários (47 voltas)"
    """
    if not compounds_used:
        return "desconhecido"
    parts = []
    for c in compounds_used:
        raw = str(c.get("compound", "")).upper()
        name_pt = _COMPOUND_PT.get(raw, raw.lower())
        parts.append(f"{name_pt} ({c.get('laps', '?')} voltas)")
    return " → ".join(parts)


# ---------------------------------------------------------------------------
# NC-A: Agenda obrigatória de cobertura (determinística)
# ---------------------------------------------------------------------------

def _build_must_cover(timeline: list, race_summary: dict) -> dict:
    """
    NC-A: Gera a agenda obrigatória de cobertura a partir dos dados.

    100% determinístico — deriva diretamente da timeline e race_summary.
    A LLM não decide o que cobrir, apenas como escrever.

    Returns:
        Dict com chaves 'desenvolvimento', 'analise_tecnica', 'destaques_individuais',
        cada uma sendo uma lista de strings compactas descrevendo o que DEVE ser narrado.
    """
    # Step 1: Compute SC windows.
    # sc_deploy_laps: laps where SC was formally deployed.
    # sc_active_ranges: (deploy, deploy+duration) for each SC — all laps inside are "active SC".
    sc_deploy_laps: set[int] = set()
    sc_active_ranges: list[tuple[int, int]] = []
    for evt in timeline:
        if evt.get("type") == "safety_car":
            start = int(evt.get("deployed_on_lap") or evt.get("lap", 0))
            duration = int(evt.get("duration_laps", 1))
            sc_deploy_laps.add(start)
            sc_active_ranges.append((start, start + duration))

    # Step 2: Collect all retirements by lap up front (needed for cause detection).
    ret_by_lap: dict[int, list[str]] = {}
    for evt in timeline:
        if evt.get("type") == "retirement":
            lap = int(evt.get("lap", 0))
            ret_by_lap.setdefault(lap, []).append(evt.get("driver", ""))

    def _is_sc_cause(lap: int) -> bool:
        """True if retirements on this lap likely caused (or triggered) a Safety Car.

        Two cases:
        1. Exact deploy lap — retirement and SC recorded on the same lap.
        2. deploy_lap+1 — F1 timing sometimes records the incident one lap later
           (e.g. opening-lap crash: SC deployed lap 1, retirements show as lap 2).
           Only applies when deploy_lap itself has NO retirements (otherwise the
           genuine cause is already on the deploy lap and lap+1 is a separate incident).
        """
        if lap in sc_deploy_laps:
            return True
        return any(
            lap == dl + 1 and not ret_by_lap.get(dl)
            for dl in sc_deploy_laps
        )

    def _in_active_sc(lap: int) -> bool:
        """True if lap is inside an active SC window and is NOT a cause lap.

        Retirements that happen mid-SC (not the cause) have lower narrative relevance
        and tend to confuse the LLM into attributing them as SC triggers.
        """
        if _is_sc_cause(lap):
            return False
        return any(start < lap < end for start, end in sc_active_ranges)

    # --- desenvolvimento: Safety Cars + Retirements + Penalties em ordem cronológica ---
    items: list[tuple[int, str]] = []

    for evt in timeline:
        t = evt.get("type")
        lap = int(evt.get("lap", 0))
        if t == "safety_car":
            duration = evt.get("duration_laps", 1)
            items.append((lap, f"Safety Car volta {lap} (duração: {duration} voltas)"))
        elif t == "penalty":
            reason = _translate_penalty_reason(evt.get("reason", ""))
            items.append((lap, f"Penalidade: {_name(evt.get('driver', ''))} volta {lap} — {reason}"))

    for lap in sorted(ret_by_lap.keys()):
        if _in_active_sc(lap):
            # Retirement during active SC: skip from desenvolvimento to avoid the LLM
            # misattributing it as an SC cause or double-mentioning it.
            # It will still appear in destaques_individuais via race_summary.dnfs.
            continue
        nomes = ", ".join(_name(d) for d in ret_by_lap[lap])
        suffix = " — causou o Safety Car" if _is_sc_cause(lap) else ""
        items.append((lap, f"Abandono de {nomes} (volta {lap}){suffix}"))

    items.sort(key=lambda x: x[0])
    desenvolvimento = [text for _, text in items]

    # --- analise_tecnica: top 3 undercuts + top 2 overcuts por magnitude ---
    # Limiting to the most impactful moves prevents the LLM from listing every
    # single manoeuvre (data dump) instead of writing a focused narrative.
    # Overcuts with very large margins (>20s) are typically SC-window artefacts
    # (opponent pitted late during SC) and are filtered out as misleading.
    _OVERCUT_MAX_S = 20.0

    undercuts_sorted = sorted(
        [e for e in timeline if e.get("type") == "undercut"],
        key=lambda e: e.get("time_gained_seconds", 0),
        reverse=True,
    )[:3]

    overcuts_all = sorted(
        [e for e in timeline if e.get("type") == "overcut"],
        key=lambda e: e.get("time_saved_seconds", 0),
    )
    # Prefer small-margin overcuts (real strategy); fall back to any if none qualify
    overcuts_real = [o for o in overcuts_all if o.get("time_saved_seconds", 0) <= _OVERCUT_MAX_S]
    overcuts_sorted = (overcuts_real if overcuts_real else overcuts_all)[:2]

    analise_tecnica: list[str] = []
    for u in undercuts_sorted:
        winner = _name(u.get("winner", ""))
        loser = _name(u.get("loser", ""))
        lap = u.get("lap", 0)
        margin = round(u.get("time_gained_seconds", 0), 1)
        analise_tecnica.append(f"Undercut: {winner} sobre {loser} (volta {lap}, +{margin}s)")
    for o in overcuts_sorted:
        winner = _name(o.get("winner", ""))
        loser = _name(o.get("loser", ""))
        lap = o.get("lap", 0)
        margin = round(o.get("time_saved_seconds", 0), 1)
        analise_tecnica.append(f"Overcut: {winner} sobre {loser} (volta {lap}, +{margin}s)")

    # --- destaques_individuais: pódio completo + todos os DNFs ---
    destaques: list[str] = []
    for p in race_summary.get("podium", []):
        code = p.get("driver", "")
        pos = p.get("position")
        destaques.append(f"{_name(code)} — P{pos}")
    for dnf in race_summary.get("dnfs", []):
        code = dnf.get("driver", "")
        lap = dnf.get("on_lap")
        destaques.append(f"{_name(code)} — abandono volta {lap}")

    return {
        "desenvolvimento": desenvolvimento,
        "analise_tecnica": analise_tecnica,
        "destaques_individuais": destaques,
    }
