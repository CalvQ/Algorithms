#!/usr/bin/env python3
"""
Berghain Challenge – Python implementation

Strategy highlights:
- Feasibility guards to avoid impossible endgames
- Scarcity- and deficit-aware scoring per attribute
- Soft correlation adjustment to avoid double-counting & handle scarcity coupling
- Dynamic threshold that tightens as the venue fills
- Online re-estimation of attribute frequencies (blend priors + observations)

Run:
  python berghain_solver.py --scenario 1 --player-id 7a15b7c6-3419-47a0-8949-8d330a6bca7c
"""

from __future__ import annotations
import argparse
import dataclasses
import logging
import sys
import time
from typing import Dict, List, Set, Tuple, Optional
import requests

# ----------------------------
# Config / Tunables
# ----------------------------
BASE_URL = "https://berghain.challenges.listenlabs.ai"
N_CAPACITY = 1000
MAX_REJECTIONS = 20000

# Scoring / thresholds
EPS = 1e-9
LAMBDA_CORR = 0.25   # strength of correlation-based nudging
ETA_THRESH = 0.5     # baseline threshold intensity
ETA_SCARCITY = 1.0   # how much we boost rare/urgent attributes
ETA_OVERSHOOT = 0.5  # discourage accepts that don't reduce any deficit (late game)

# Prior blending: how fast we trust observations over provided priors
PRIOR_PSEUDOCNT = 1000.0  # ~ how many "virtual samples" we grant the prior

# HTTP
TIMEOUT = 20
RETRY = 3
SLEEP_BETWEEN = 0.0  # seconds; set small>0 if you want to be polite to the API


@dataclasses.dataclass
class GameSpec:
    game_id: str
    constraints_min: Dict[str, int]
    rel_freq: Dict[str, float]
    corr: Dict[str, Dict[str, float]]  # sparse map: corr[a][b] in [-1,1]


@dataclasses.dataclass
class State:
    admitted_count: int = 0
    rejected_count: int = 0
    count_has: Dict[str, int] = dataclasses.field(default_factory=dict)

    seen_total: int = 0
    seen_has: Dict[str, int] = dataclasses.field(default_factory=dict)

    def ensure_attrs(self, attrs: List[str]):
        for a in attrs:
            self.count_has.setdefault(a, 0)
            self.seen_has.setdefault(a, 0)


# ----------------------------
# HTTP Helpers
# ----------------------------
def _get(session: requests.Session, url: str, params: Dict) -> dict:
    for i in range(RETRY):
        try:
            resp = session.get(url, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if i == RETRY - 1:
                raise
            time.sleep(0.5 * (i + 1))
    # unreachable
    return {}


# ----------------------------
# API wrappers
# ----------------------------
def new_game(session: requests.Session, scenario: int, player_id: str) -> GameSpec:
    url = f"{BASE_URL}/new-game"
    payload = {"scenario": scenario, "playerId": player_id}
    data = _get(session, url, payload)

    game_id = data["gameId"]

    constraints_min = {}
    for c in data.get("constraints", []):
        attr = c["attribute"]
        constraints_min[attr] = int(c["minCount"])

    stats = data.get("attributeStatistics", {})
    rel = stats.get("relativeFrequencies", {}) or {}
    corr = stats.get("correlations", {}) or {}

    # Ensure symmetric corr entries exist (fallback 0)
    # Normalize structure to corr[a][b]
    attrs = set(rel.keys()) | set(constraints_min.keys())
    for a in list(attrs):
        corr.setdefault(a, {})
        for b in list(attrs):
            if a == b:
                corr[a][b] = corr[a].get(b, 1.0)  # self-corr = 1
            else:
                rab = corr.get(a, {}).get(b, None)
                rba = corr.get(b, {}).get(a, None)
                if rab is None and rba is None:
                    corr[a][b] = 0.0
                    corr.setdefault(b, {})[a] = 0.0
                elif rab is None:
                    corr[a][b] = rba
                elif rba is None:
                    corr.setdefault(b, {})[a] = rab

    return GameSpec(
        game_id=game_id,
        constraints_min=constraints_min,
        rel_freq=rel,
        corr=corr,
    )


def get_next(session: requests.Session, game_id: str, person_index: int, accept: Optional[bool]) -> dict:
    url = f"{BASE_URL}/decide-and-next"
    params = {"gameId": game_id, "personIndex": person_index}
    if accept is not None:
        params["accept"] = "true" if accept else "false"
    return _get(session, url, params)


# ----------------------------
# Core logic
# ----------------------------
def remaining_slots(state: State) -> int:
    return N_CAPACITY - state.admitted_count


def deficits(spec: GameSpec, state: State, attrs: List[str]) -> Dict[str, int]:
    d = {}
    for a in attrs:
        need = spec.constraints_min.get(a, 0)
        have = state.count_has.get(a, 0)
        d[a] = max(0, need - have)
    return d


def must_accept_attributes(d: Dict[str, int], R: int) -> Set[str]:
    if R <= 0:
        return set()
    return {a for a, val in d.items() if val == R}


def accept_would_be_infeasible(person_attrs: Dict[str, bool], d: Dict[str, int], R: int) -> bool:
    # If we accept, remaining becomes R-1. If person lacks attribute a but we still
    # need d[a] of them, and d[a] > R-1 -> impossible to meet min later.
    if R <= 0:
        return True
    next_R = R - 1
    for a, need in d.items():
        if not person_attrs.get(a, False) and need > next_R:
            return True
    return False


def reject_would_be_infeasible(person_attrs: Dict[str, bool], d: Dict[str, int], R: int) -> bool:
    # If there are attributes that must appear in **all** remaining admits (need == R),
    # and this person has all of them, rejecting risks impossibility.
    if R <= 0:
        return False
    M = must_accept_attributes(d, R)
    if not M:
        return False
    for a in M:
        if not person_attrs.get(a, False):
            return False
    return True


def current_freq_estimate(spec: GameSpec, state: State, attrs: List[str]) -> Dict[str, float]:
    w = state.seen_total / (state.seen_total + PRIOR_PSEUDOCNT)
    phat = {}
    for a in attrs:
        p_seen = (state.seen_has.get(a, 0) / state.seen_total) if state.seen_total > 0 else 0.0
        phat[a] = (1 - w) * spec.rel_freq.get(a, 0.0) + w * p_seen
        phat[a] = min(1.0, max(0.0, phat[a]))
    return phat


def attribute_weights(d: Dict[str, int], phat: Dict[str, float], R: int, attrs: List[str]) -> Dict[str, float]:
    w = {}
    denom_R = max(1, R)
    for a in attrs:
        scarcity = d[a] / max(1.0, denom_R * max(EPS, phat[a]))
        w[a] = ETA_SCARCITY * scarcity
    return w


def correlation_adjust(weights: Dict[str, float], d: Dict[str, int], R: int,
                       attrs: List[str], corr: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    if LAMBDA_CORR <= 0:
        return weights
    u = {a: d[a] / max(1, R) for a in attrs}  # normalized deficit
    adj = dict(weights)
    for a in attrs:
        s = 0.0
        row = corr.get(a, {})
        for b in attrs:
            s += row.get(b, 0.0) * u[b]
        adj[a] = max(0.0, weights[a] + LAMBDA_CORR * s)
    return adj


def person_score(person_attrs: Dict[str, bool], weights: Dict[str, float], attrs: List[str]) -> float:
    s = 0.0
    for a in attrs:
        if person_attrs.get(a, False):
            s += weights.get(a, 0.0)
    return s


def overshoot_penalty(person_attrs: Dict[str, bool], d: Dict[str, int], attrs: List[str]) -> float:
    if sum(d.values()) == 0:
        return 0.0  # compliant; don't discourage filling
    helps = any(person_attrs.get(a, False) and d[a] > 0 for a in attrs)
    return 0.0 if helps else ETA_OVERSHOOT


def dynamic_threshold(state: State, d: Dict[str, int], R: int) -> float:
    """
    Threshold is purely a function of *constraint pressure*.
    If pressure is zero, threshold ~ 0 so we accept freely.
    """
    if R <= 0 or not d:
        return 0.0
    # Pressure: weighted by both max and mean deficit intensity
    pressures = [val / max(1, R) for val in d.values()]
    cliff = max(pressures)                  # worst single-attr pressure
    mean_p = sum(pressures) / len(pressures)
    # Tunable blend; scale small so it's comparable to typical weights
    return 0.3 * cliff + 0.1 * mean_p



def update_seen(state: State, person_attrs: Dict[str, bool], attrs: List[str]) -> None:
    state.seen_total += 1
    for a in attrs:
        if person_attrs.get(a, False):
            state.seen_has[a] = state.seen_has.get(a, 0) + 1


def apply_accept(state: State, person_attrs: Dict[str, bool], attrs: List[str]) -> None:
    state.admitted_count += 1
    for a in attrs:
        if person_attrs.get(a, False):
            state.count_has[a] = state.count_has.get(a, 0) + 1


def apply_reject(state: State) -> None:
    state.rejected_count += 1


def decide(spec: GameSpec, state: State, person_attrs: Dict[str, bool], attrs: List[str]) -> bool:
    """
    Returns True to accept, False to reject.
    """
    # Include everyone seen in online estimates (admitted or not)
    update_seen(state, person_attrs, attrs)

    R = remaining_slots(state)
    d = deficits(spec, state, attrs)

    if sum(d.values()) == 0:
        return True

    # Hard guards
    if R <= 0:
        return False
    if accept_would_be_infeasible(person_attrs, d, R):
        return False
    if reject_would_be_infeasible(person_attrs, d, R):
        return True

    # Scoring
    phat = current_freq_estimate(spec, state, attrs)
    w0 = attribute_weights(d, phat, R, attrs)
    w = correlation_adjust(w0, d, R, attrs, spec.corr)

    score = person_score(person_attrs, w, attrs)
    score -= overshoot_penalty(person_attrs, d, attrs)

    tau = dynamic_threshold(state, d, R)

    return score >= tau


# ----------------------------
# Driver
# ----------------------------
def run_game(scenario: int, player_id: str, verbose: bool = True) -> Tuple[int, Optional[str]]:
    session = requests.Session()
    spec = new_game(session, scenario, player_id)

    # Build the working attribute universe
    attrs = sorted(set(spec.rel_freq.keys()) | set(spec.constraints_min.keys()))
    state = State()
    state.ensure_attrs(attrs)

    # First fetch: no decision yet
    person_index = 0
    resp = get_next(session, spec.game_id, person_index, accept=None)

    while True:
        status = resp.get("status")
        if status == "failed":
            reason = resp.get("reason", "unknown")
            logging.error("Game failed: %s", reason)
            return state.rejected_count, reason

        if status == "completed":
            # Server's truth of rejectedCount is final, but we also track locally.
            server_rc = resp.get("rejectedCount", state.rejected_count)
            if verbose:
                logging.info("Game completed. Rejections: %d", server_rc)
            return server_rc, None

        if status != "running":
            logging.error("Unexpected status: %s", status)
            return state.rejected_count, f"unexpected status {status}"

        # Extract person
        next_person = resp.get("nextPerson") or {}
        person_index = next_person.get("personIndex")
        person_attrs = next_person.get("attributes") or {}

        # Decide
        decision = decide(spec, state, person_attrs, attrs)

        # Apply local bookkeeping
        if decision:
            apply_accept(state, person_attrs, attrs)
        else:
            apply_reject(state)

        # Stop conditions (local guard; server should also handle)
        if state.admitted_count >= N_CAPACITY:
            # We don't have to send another call; but the API expects a final accept on last person to move forward.
            # We'll still send the decision for this person (we already did) and continue until server says completed.
            pass
        if state.rejected_count >= MAX_REJECTIONS:
            # Keep going—the server will return failed/completed accordingly.
            pass

        if SLEEP_BETWEEN > 0:
            time.sleep(SLEEP_BETWEEN)

        resp = get_next(session, spec.game_id, person_index, accept=decision)

        # Optional logging
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(
                "After idx=%s: admitted=%d, rejected=%d",
                str(person_index),
                state.admitted_count,
                state.rejected_count,
            )

    # unreachable
    # return state.rejected_count, None


# ----------------------------
# CLI
# ----------------------------
def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Berghain Challenge solver")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3], required=True,
                        help="Scenario number (1, 2, or 3)")
    parser.add_argument("--player-id", type=str, required=False,
                        default="7a15b7c6-3419-47a0-8949-8d330a6bca7c")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output")
    parser.add_argument("--debug", action="store_true", help="Verbose debug logging")
    args = parser.parse_args(argv)

    level = logging.INFO
    if args.quiet:
        level = logging.WARNING
    if args.debug:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    try:
        rejected, err = run_game(args.scenario, args.player_id, verbose=not args.quiet)
        if err:
            print(f"Game ended with error: {err} | rejected={rejected}")
            return 2
        print(f"Completed. Rejections: {rejected}")
        return 0
    except requests.HTTPError as e:
        logging.error("HTTP error: %s", e)
        return 3
    except Exception as e:
        logging.exception("Unhandled error: %s", e)
        return 4


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
