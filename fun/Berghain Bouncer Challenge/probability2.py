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

@dataclasses.dataclass
class AttrBayesCalibrator:
    """Per-attribute Beta calibration for marginal P(A=1).
    We only need marginals to estimate E[tokens per person] = sum P(A_i)."""
    alpha: Dict[str, float]
    beta: Dict[str, float]

    @staticmethod
    def from_marginals(p: Dict[str, float], prior_strength: float = 50.0) -> "AttrBayesCalibrator":
        alpha = {}
        beta = {}
        for a, pi in p.items():
            pi = float(max(1e-6, min(1.0 - 1e-6, pi)))
            alpha[a] = prior_strength * pi
            beta[a]  = prior_strength * (1.0 - pi)
        return AttrBayesCalibrator(alpha=alpha, beta=beta)

    def update_with_person(self, attrs: Dict[str, bool]) -> None:
        # Update Beta for any attribute we have priors for
        for a, val in attrs.items():
            if a not in self.alpha:
                continue
            if val:
                self.alpha[a] += 1.0
            else:
                self.beta[a]  += 1.0

    def mean(self, a: str) -> float:
        if a not in self.alpha:
            return 0.0
        den = self.alpha[a] + self.beta[a]
        return float(self.alpha[a] / den) if den > 0 else 0.0

    def mu_tokens_per_person(self, needed: list[str]) -> float:
        # Expected token count per random future person for the current needed set
        return sum(self.mean(a) for a in needed)

def _poisson_binomial_pmf(ps: List[float]) -> List[float]:
    """
    Poisson–binomial PMF for sum of independent Bernoullis with probs ps.
    Returns probs for K=0..m where m=len(ps).
    """
    m = len(ps)
    pmf = [1.0] + [0.0] * m
    for p in ps:
        # convolve with Bernoulli(p)
        for k in range(m, 0, -1):
            pmf[k] = pmf[k] * (1.0 - p) + pmf[k - 1] * p
        pmf[0] *= (1.0 - p)
    return pmf

def _choose_k_threshold(needed: List[str], bayes: "AttrBayesCalibrator", R: int, seats_left: int) -> int:
    """
    Choose the minimal k such that E[K | K>=k] >= R / seats_left.
    If no k satisfies it, return m (max tokens).
    """
    m = len(needed)
    if m == 0 or seats_left <= 0:
        return 0
    # probs for each needed attribute (calibrated)
    ps = [max(1e-9, min(1.0 - 1e-9, bayes.mean(a))) for a in needed]
    pmf = _poisson_binomial_pmf(ps)
    req = R / max(1, seats_left)

    # Precompute tail mass and tail expected tokens for k..m
    tail_mass = [0.0] * (m + 2)
    tail_tokens = [0.0] * (m + 2)
    for k in range(m, -1, -1):
        tail_mass[k] = tail_mass[k + 1] + pmf[k]
        tail_tokens[k] = tail_tokens[k + 1] + k * pmf[k]

    for k in range(1, m + 1):
        mass = tail_mass[k]
        if mass <= 0.0:
            continue
        avg_tokens_if_acceptors = tail_tokens[k] / mass  # E[K | K>=k]
        if avg_tokens_if_acceptors >= req:
            return k
    return m  # most selective if even K=m doesn't reach req


def play_probability_aware(cfg: GameConfig) -> int:
    """
    Dynamic K-threshold policy (risk-neutral, multi-token aware):
      - Maintain per-attribute Beta-calibrated P_hat(A=1).
      - At each step, compute needed attrs and pick the smallest k* s.t.
        E[K | K>=k*] >= (total_remaining_tokens / seats_left).
      - ACCEPT iff person's contribution (over needed attrs) >= k*.
      - Apply decrements on accept. When total_remaining hits 0, accept all.
    """
    client = BerghainClient(cfg)

    game = client.new_game()
    game_id = game.get("gameId")
    if not game_id:
        raise RuntimeError("/new-game response missing gameId")

    constraints = game.get("constraints", [])
    attr_stats = AttrStats.from_api(game.get("attributeStatistics", {}))
    tstate = ThresholdState.from_constraints(constraints)

    # Per-attribute Bayesian calibrator on marginals
    bayes = AttrBayesCalibrator.from_marginals(attr_stats.p, prior_strength=50.0)

    if cfg.verbose:
        print("Game created:")
        print(f"  gameId: {game_id}")
        print(f"  constraints: {constraints}")
        print(f"  initial remaining: {tstate.remaining}")
        print(f"  threshold_total: {tstate.total_remaining}")
        print(f"  venue_capacity: {cfg.venue_capacity}")
        print(f"  stats: marginals={attr_stats.p} (|rho|={len(attr_stats.rho)})")

    # First fetch: omit `accept` to get the first person
    state = client.decide_and_next(game_id, person_index=0, accept=None)

    while state.get("status") == "running":
        next_person = state.get("nextPerson") or {}
        idx = int(next_person.get("personIndex", 0))
        attrs = next_person.get("attributes", {})

        admitted = int(state.get("admittedCount", 0))
        seats_left = cfg.venue_capacity - admitted
        R = tstate.total_remaining

        # Update per-attribute marginals with every arrival (policy-independent)
        bayes.update_with_person(attrs)

        # Short-circuit: once quotas met, accept everyone
        if R <= 0:
            decision = True
            contrib, decs = tstate.contribution_for(attrs)  # not used
        else:
            needed = tstate.needed_list()

            # Person's contribution wrt current needs
            contrib, decs = tstate.contribution_for(attrs)

            # Pick dynamic token threshold k*
            k_star = _choose_k_threshold(needed, bayes, R=R, seats_left=seats_left)

            # Accept iff contrib >= k*
            decision = (contrib >= k_star)

            if cfg.verbose:
                print(f"[k-threshold] needed={needed} R={R} seats_left={seats_left} k*={k_star} contrib={contrib}")

            if decision and contrib > 0:
                tstate.apply(decs)

        if cfg.verbose:
            print(
                f"Person {idx}: attrs={attrs} | contrib={contrib} | admitted={admitted} "
                f"| seats_left={seats_left} | accept={decision} "
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

def is_feasible_after_accept(
    remaining: Dict[str, int],
    decs: Dict[str, int],
    seats_left: int,
    bayes: "AttrBayesCalibrator",
    eps: float = 1e-9,
) -> bool:
    """
    Check if accepting this person (applying `decs` and consuming one seat)
    still leaves a feasible path *in expectation* for every attribute:
        p_hat[a] * seats_after >= r'_a  for all a
    """
    seats_after = seats_left - 1
    if seats_after < 0:
        return False
    for a, r in remaining.items():
        r_prime = r - int(decs.get(a, 0))
        if r_prime <= 0:
            continue
        p_hat = bayes.mean(a)
        # If we don't expect enough future hits for attribute a, infeasible
        if p_hat * seats_after + eps < r_prime:
            return False
    return True


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
