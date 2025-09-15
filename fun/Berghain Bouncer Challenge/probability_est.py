#!/usr/bin/env python3
"""
Berghain Challenge — probability-aware strategy

This script upgrades the baseline "threshold" rule by using the provided
attribute marginals and pairwise correlations to estimate the probability that
a *random future person* will contribute to at least one *still-needed*
attribute. For a non-contributor, we accept iff:

    p_contrib * seats_after_if_accept  >=  total_remaining

where p_contrib ≈ P(any needed attribute is true), estimated from marginals and
correlations (pairwise-adjusted union probability).

Usage:
  python berghain_toy_probaware.py \
    --base-url https://berghain.challenges.listenlabs.ai/ \
    --scenario 1 \
    --player-id 7a15b7c6-3419-47a0-8949-8d330a6bca7c
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
from typing import Any, Dict, Optional, Tuple, List

import requests


@dataclasses.dataclass
class GameConfig:
    base_url: str
    scenario: int
    player_id: str
    venue_capacity: int = 1000
    poll_delay_sec: float = 0.0
    timeout_sec: float = 30.0
    verbose: bool = True


class BerghainClient:
    def __init__(self, cfg: GameConfig):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.cfg.base_url.rstrip('/')}{path}"
        if self.cfg.verbose:
            print(f"GET {url} params={params}")
        r = self.session.get(url, params=params, timeout=self.cfg.timeout_sec)
        r.raise_for_status()
        return r.json()

    def new_game(self) -> Dict[str, Any]:
        return self._get(
            "/new-game",
            {"scenario": self.cfg.scenario, "playerId": self.cfg.player_id},
        )

    def decide_and_next(
        self, game_id: str, person_index: int, accept: Optional[bool] = None
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"gameId": game_id, "personIndex": person_index}
        if accept is not None:
            params["accept"] = str(accept).lower()  # true/false
        return self._get("/decide-and-next", params)


# ---------- Threshold (quota) bookkeeping ----------

@dataclasses.dataclass
class ThresholdState:
    remaining: Dict[str, int]
    total_remaining: int

    @classmethod
    def from_constraints(cls, constraints: list[dict]) -> "ThresholdState":
        remaining = {c["attribute"]: int(c["minCount"]) for c in constraints}
        total = sum(remaining.values())
        return cls(remaining=remaining, total_remaining=total)

    def needed_list(self) -> List[str]:
        return [a for a, r in self.remaining.items() if r > 0]

    def contribution_for(self, attrs: Dict[str, bool]) -> Tuple[int, Dict[str, int]]:
        """Return (contribution, decrements_per_attr) for this person."""
        decs: Dict[str, int] = {}
        contrib = 0
        if self.total_remaining <= 0:
            return 0, decs
        for a, needed in self.remaining.items():
            if needed > 0 and attrs.get(a):
                decs[a] = 1
                contrib += 1
        return contrib, decs

    def apply(self, decs: Dict[str, int]) -> None:
        for a, d in decs.items():
            if d <= 0:
                continue
            if self.remaining.get(a, 0) > 0:
                self.remaining[a] -= 1
                self.total_remaining -= 1


# ---------- Probability model from attributeStatistics ----------

@dataclasses.dataclass
class AttrStats:
    """Holds marginals and pairwise correlations; estimates P(union of needed)."""
    p: Dict[str, float]                       # P(A_i)
    rho: Dict[Tuple[str, str], float]         # rho_ij for binary attrs

    @staticmethod
    def from_api(stats_json: dict) -> "AttrStats":
        p = dict(stats_json.get("relativeFrequencies", {}) or {})
        corr = stats_json.get("correlations", {}) or {}
        rho: Dict[Tuple[str, str], float] = {}
        # Store correlations symmetrically in sorted key order
        for i, row in corr.items():
            if not isinstance(row, dict):
                continue
            for j, r in row.items():
                if i == j:
                    continue
                key = tuple(sorted((i, j)))
                rho[key] = float(r)
        return AttrStats(p=p, rho=rho)

    def _joint_pij(self, ai: str, aj: str) -> float:
        """Approximate P(A_i ∧ A_j) from marginals and correlation rho_ij."""
        i, j = sorted((ai, aj))
        rho_ij = self.rho.get((i, j), 0.0)
        p_i = float(self.p.get(i, 0.0))
        p_j = float(self.p.get(j, 0.0))
        base = p_i * p_j
        # For Bernoulli vars, cov = rho * sqrt(var_i var_j)
        term = rho_ij * math.sqrt(max(p_i * (1 - p_i), 0.0) * max(p_j * (1 - p_j), 0.0))
        pij = base + term
        # Clip to valid probability range
        return min(1.0, max(0.0, pij))

    def p_union_needed(self, needed: List[str]) -> float:
        """Pairwise-adjusted estimate of P(any needed attribute is true)."""
        if not needed:
            return 0.0

        # Conservative second-order Bonferroni lower bound:
        sum_p = 0.0
        sum_pij = 0.0
        rhos: List[float] = []
        n = len(needed)
        for a in needed:
            sum_p += float(self.p.get(a, 0.0))
        for x in range(n):
            for y in range(x + 1, n):
                ai, aj = needed[x], needed[y]
                pij = self._joint_pij(ai, aj)
                sum_pij += pij
                rhos.append(self.rho.get(tuple(sorted((ai, aj))), 0.0))
        p_lb = max(0.0, sum_p - sum_pij)

        # Optimistic independence-like estimate:
        prod = 1.0
        for a in needed:
            prod *= (1.0 - float(self.p.get(a, 0.0)))
        p_ind = 1.0 - prod

        # Blend based on average (positive) correlation; more positive → trust LB more.
        if rhos:
            avg_rho_pos = max(0.0, sum(rhos) / len(rhos))
        else:
            avg_rho_pos = 0.0
        alpha = min(1.0, max(0.0, avg_rho_pos))
        return alpha * p_lb + (1.0 - alpha) * p_ind


# ---------- Main gameplay loop with probability-aware reservation ----------

def play_probability_aware(cfg: GameConfig) -> int:
    """
    Risk-neutral (z=0) policy + online Beta calibration of p_contrib.
    - Uses API stats to initialize p_contrib for the CURRENT needed set.
    - Updates a Beta posterior with live observations of "would contribute?"
      for each arrival (regardless of accept/reject), and restarts the
      posterior whenever the needed set changes (an epoch).
    """
    client = BerghainClient(cfg)

    game = client.new_game()
    game_id = game.get("gameId")
    if not game_id:
        raise RuntimeError("/new-game response missing gameId")

    constraints = game.get("constraints", [])
    attr_stats = AttrStats.from_api(game.get("attributeStatistics", {}))
    tstate = ThresholdState.from_constraints(constraints)

    if cfg.verbose:
        print("Game created:")
        print(f"  gameId: {game_id}")
        print(f"  constraints: {constraints}")
        print(f"  initial remaining: {tstate.remaining}")
        print(f"  threshold_total: {tstate.total_remaining}")
        print(f"  venue_capacity: {cfg.venue_capacity}")
        print(f"  stats: marginals={attr_stats.p} (|rho|={len(attr_stats.rho)})")

    # --- Online Beta calibration state (per-needed-set epoch) ---
    epoch_key: Optional[tuple] = None
    p_contrib_api: float = 0.0
    alpha0 = beta0 = 0.0
    seen_trials = 0
    seen_hits = 0
    K_PRIOR = 50.0      # prior strength; 20–100 is reasonable
    EPS = 1e-6          # to avoid degenerate 0/1 probs

    def ensure_epoch(needed: list[str]) -> None:
        nonlocal epoch_key, p_contrib_api, alpha0, beta0, seen_trials, seen_hits
        key = tuple(sorted(needed))
        if key == epoch_key:
            return
        epoch_key = key
        # Recompute API-based prior for the NEW needed set
        p_contrib_api = attr_stats.p_union_needed(needed) if needed else 0.0
        p0 = max(EPS, min(1.0 - EPS, p_contrib_api))
        alpha0 = K_PRIOR * p0
        beta0  = K_PRIOR * (1.0 - p0)
        seen_trials = 0
        seen_hits = 0
        if cfg.verbose:
            print(f"[epoch] needed={list(key)} | p_api≈{p_contrib_api:.4f} | "
                  f"alpha0={alpha0:.2f} beta0={beta0:.2f}")

    def current_p_contrib() -> float:
        # Posterior mean
        denom = alpha0 + beta0 + seen_trials
        if denom <= 0:
            return max(0.0, min(1.0, p_contrib_api))
        return max(0.0, min(1.0, (alpha0 + seen_hits) / denom))

    # First fetch: omit `accept` to get the first person
    state = client.decide_and_next(game_id, person_index=0, accept=None)

    while state.get("status") == "running":
        next_person = state.get("nextPerson") or {}
        idx = int(next_person.get("personIndex", 0))
        attrs = next_person.get("attributes", {})

        admitted = int(state.get("admittedCount", 0))
        capacity_left = cfg.venue_capacity - admitted
        R = tstate.total_remaining

        # Establish / refresh epoch for current needed set
        needed = tstate.needed_list()
        ensure_epoch(needed)

        # Update Beta counts with what we observe about THIS arrival
        # (while quotas remain). "Would contribute?" is measured against the
        # *current* needed set prior to making a decision.
        if R > 0:
            would_contribute = any(attrs.get(a, False) for a in needed)
            seen_trials += 1
            if would_contribute:
                seen_hits += 1

        # Decide using risk-neutral reservation
        contrib, decs = tstate.contribution_for(attrs)
        if R <= 0:
            decision = True
        elif contrib > 0:
            decision = True
            tstate.apply(decs)
        else:
            p_contrib = current_p_contrib()
            seats_after_if_accept = max(0, capacity_left - 1)
            expected_hits = p_contrib * seats_after_if_accept
            decision = (expected_hits >= R)

            if cfg.verbose:
                print(
                    f"[reserve] needed={needed} p_api≈{p_contrib_api:.4f} "
                    f"p_hat≈{p_contrib:.4f} trials={seen_trials} hits={seen_hits} "
                    f"S'={seats_after_if_accept} E[hits]≈{expected_hits:.2f} R={R}"
                )

        if cfg.verbose:
            print(
                f"Person {idx}: attrs={attrs} | contrib={contrib} | admitted={admitted} "
                f"| capacity_left={capacity_left} | accept={decision} "
                f"| remaining={tstate.remaining} | total_remaining={tstate.total_remaining}"
            )

        state = client.decide_and_next(game_id, person_index=idx, accept=decision)

        if cfg.verbose:
            print(
                f"Progress: admitted={state.get('admittedCount')} rejected={state.get('rejectedCount')}"
            )

        if cfg.poll_delay_sec:
            time.sleep(cfg.poll_delay_sec)

    _print_final(state)
    return int(state.get("rejectedCount", 0))



# ------------------------
# Helpers
# ------------------------

def _print_final(state: Dict[str, Any]) -> None:
    status = state.get("status")
    rejected = int(state.get("rejectedCount", 0))
    if status == "completed":
        print("Game completed ✅")
    elif status == "failed":
        print(f"Game failed ❌ reason={state.get('reason')}")
    else:
        print(f"Unexpected status: {status}")
    print(f"Total rejected: {rejected}")


def parse_args(argv: list[str]) -> GameConfig:
    p = argparse.ArgumentParser(description="Berghain Challenge — probability-aware client")
    p.add_argument("--base-url", required=False, default="https://berghain.challenges.listenlabs.ai/")
    p.add_argument("--scenario", type=int, default=1, choices=[1, 2, 3])
    p.add_argument(
        "--player-id",
        default="7a15b7c6-3419-47a0-8949-8d330a6bca7c",
        help="Player UUID",
    )
    p.add_argument("--venue-capacity", type=int, default=1000, help="Venue capacity (default 1000)")
    p.add_argument("--timeout-sec", type=float, default=30.0)
    p.add_argument("--poll-delay-sec", type=float, default=0.0)
    p.add_argument("--quiet", action="store_true", help="Reduce logging output")
    args = p.parse_args(argv)

    return GameConfig(
        base_url=args.base_url,
        scenario=args.scenario,
        player_id=args.player_id,
        venue_capacity=args.venue_capacity,
        poll_delay_sec=args.poll_delay_sec,
        timeout_sec=args.timeout_sec,
        verbose=not args.quiet,
    )


def main():
    cfg = parse_args(sys.argv[1:])
    try:
        play_probability_aware(cfg)
    except requests.HTTPError as e:
        print(f"HTTP error: {e}")
        if e.response is not None:
            print(f"Response text: {e.response.text}")
        sys.exit(2)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
