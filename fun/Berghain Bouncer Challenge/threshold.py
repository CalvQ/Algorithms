#!/usr/bin/env python3
"""
Berghain Challenge

This script plays the game using:
  threshold  — accept according to per-attribute minimums using the rule:

Threshold rule:
- Keep remaining[attributeId] = minCount and `total_remaining = sum(remaining.values())`.
- For each person, their **contribution** is how many still-needed attributes they have
  (e.g., well_dressed+young contributes 2 if both are still needed).
- If contribution > 0 → ACCEPT and decrement each corresponding attribute by 1,
  decreasing total_remaining by the contribution.
- If contribution == 0 and total_remaining > 0 →
    * ACCEPT **unless** we must reserve all remaining capacity for contributors.
    * Concretely, let `capacity_left = venue_capacity - admittedCount`.
      - If `capacity_left > total_remaining` → ACCEPT (we still have buffer).
      - Else (i.e., `capacity_left <= total_remaining`) → REJECT to preserve capacity.
- Once total_remaining == 0 → ACCEPT everyone to minimize rejections.

Usage:
  python berghain_toy_accept_all.py \
    --base-url http://localhost:8000 \
    --scenario 1 \
    --player-id 7a15b7c6-3419-47a0-8949-8d330a6bca7c \

Notes:
- API paths are taken from the prompt. Set --base-url to your server.
- The first call to /decide-and-next omits the `accept` flag (per API), which
  returns the first person. Subsequent calls include `accept` for the current
  personIndex.
- This is a baseline; it doesn't use attribute frequencies or correlations yet.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from typing import Any, Dict, Optional, Tuple

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

@dataclasses.dataclass
class ThresholdState:
    remaining: Dict[str, int]
    total_remaining: int

    @classmethod
    def from_constraints(cls, constraints: list[dict]) -> "ThresholdState":
        remaining = {c["attribute"]: int(c["minCount"]) for c in constraints}
        total = sum(remaining.values())
        return cls(remaining=remaining, total_remaining=total)

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


def play_threshold(cfg: GameConfig) -> int:
    client = BerghainClient(cfg)

    game = client.new_game()
    game_id = game.get("gameId")
    if not game_id:
        raise RuntimeError("/new-game response missing gameId")

    constraints = game.get("constraints", [])
    tstate = ThresholdState.from_constraints(constraints)

    if cfg.verbose:
        print("Game created:")
        print(f"  gameId: {game_id}")
        print(f"  constraints: {constraints}")
        print(f"  initial remaining: {tstate.remaining}")
        print(f"  threshold_total: {tstate.total_remaining}")
        print(f"  venue_capacity: {cfg.venue_capacity}")

    # First fetch: omit `accept` to get the first person
    state = client.decide_and_next(game_id, person_index=0, accept=None)

    while state.get("status") == "running":
        next_person = state.get("nextPerson") or {}
        idx = int(next_person.get("personIndex", 0))
        attrs = next_person.get("attributes", {})

        admitted = int(state.get("admittedCount", 0))
        capacity_left = cfg.venue_capacity - admitted

        # Decide using updated threshold logic
        contrib, decs = tstate.contribution_for(attrs)
        constraints_met = tstate.total_remaining <= 0

        if constraints_met:
            decision = True
        elif contrib > 0:
            decision = True
            tstate.apply(decs)
        else:
            # Non-contributor: accept unless we must reserve all remaining capacity
            # for contributors, i.e., capacity_left <= total_remaining
            decision = capacity_left > tstate.total_remaining

        if cfg.verbose:
            print(
                f"Person {idx}: attrs={attrs} | contrib={contrib} | admitted={admitted} | capacity_left={capacity_left} | accept={decision} | remaining={tstate.remaining} | total_remaining={tstate.total_remaining}"
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
    p = argparse.ArgumentParser(description="Berghain Challenge — toy client with strategies")
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
        play_threshold(cfg)
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
