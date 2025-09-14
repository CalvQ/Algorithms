#!/usr/bin/env python3
"""
Berghain Challenge — Toy Client (accepts everyone)

This script plays the game by always accepting each person until the venue is full
or the game ends. It demonstrates how to:
  1) start a new game
  2) fetch the first person
  3) decide+advance repeatedly (always accept=True)

Usage:
  python berghain_toy_accept_all.py \
    --base-url http://localhost:8000 \
    --scenario 1 \
    --player-id 7a15b7c6-3419-47a0-8949-8d330a6bca7c

Notes:
- The API paths are taken from the prompt. Set --base-url to wherever the server runs.
- The first call to /decide-and-next omits the `accept` flag (per API), which returns the first person.
- Subsequent calls include accept=True for the returned personIndex.
- This program logs minimal progress; tweak verbosity via --verbose.
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from typing import Any, Dict, Optional

import requests


@dataclasses.dataclass
class GameConfig:
    base_url: str
    scenario: int
    player_id: str
    poll_delay_sec: float = 0.0  # no delay by default
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


# ------------------------
# Strategy: accept everyone
# ------------------------

def play_accept_all(cfg: GameConfig) -> int:
    client = BerghainClient(cfg)

    game = client.new_game()
    game_id = game.get("gameId")
    if not game_id:
        raise RuntimeError("/new-game response missing gameId")

    if cfg.verbose:
        constraints = game.get("constraints", [])
        stats = game.get("attributeStatistics", {})
        print("Game created:")
        print(f"  gameId: {game_id}")
        print(f"  constraints: {constraints}")
        print(f"  attributeStatistics keys: {list(stats.keys())}")

    # First fetch: omit `accept` per API to get the first person
    state = client.decide_and_next(game_id, person_index=0, accept=None)

    steps = 0
    while state.get("status") == "running":
        next_person = state.get("nextPerson") or {}
        idx = int(next_person.get("personIndex", 0))
        attrs = next_person.get("attributes", {})

        if cfg.verbose:
            print(f"Person {idx}: attrs={attrs}")
            print(" -> decision: ACCEPT")

        # Always accept everyone
        state = client.decide_and_next(game_id, person_index=idx, accept=True)
        steps += 1

        if cfg.verbose:
            print(
                f"Progress: admitted={state.get('admittedCount')} rejected={state.get('rejectedCount')}"
            )

        if cfg.poll_delay_sec:
            time.sleep(cfg.poll_delay_sec)

    # Status is either completed or failed
    status = state.get("status")
    rejected = int(state.get("rejectedCount", 0))

    if status == "completed":
        print("Game completed ✅")
    elif status == "failed":
        print(f"Game failed ❌ reason={state.get('reason')}")
    else:
        print(f"Unexpected status: {status}")

    print(f"Total rejected: {rejected}")
    return rejected


def parse_args(argv: list[str]) -> GameConfig:
    p = argparse.ArgumentParser(description="Berghain Challenge — accept-all toy client")
    p.add_argument("--base-url", required=False, default="https://berghain.challenges.listenlabs.ai/")
    p.add_argument("--scenario", type=int, default=1, choices=[1, 2, 3])
    p.add_argument(
        "--player-id",
        default="7a15b7c6-3419-47a0-8949-8d330a6bca7c",
        help="Player UUID",
    )
    p.add_argument("--timeout-sec", type=float, default=30.0)
    p.add_argument("--poll-delay-sec", type=float, default=0.0)
    p.add_argument("--quiet", action="store_true", help="Reduce logging output")
    args = p.parse_args(argv)

    return GameConfig(
        base_url=args.base_url,
        scenario=args.scenario,
        player_id=args.player_id,
        poll_delay_sec=args.poll_delay_sec,
        timeout_sec=args.timeout_sec,
        verbose=not args.quiet,
    )


def main():
    cfg = parse_args(sys.argv[1:])
    try:
        play_accept_all(cfg)
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
