"""
US Open 2025 scraping via "graph closure" on classic player pages.

Why:
- Bulk scraping hundreds of profiles quickly tends to miss pages (timeouts / blocking)
- We only need the subset of players who actually appear in US Open 2025 matches

Approach:
- Start from a small seed set (players already seen in US Open 2025 rows)
- For each player, scrape their US Open 2025 rows from `player-classic.cgi`
- Add newly discovered opponents to the queue
- Repeat until queue exhausted (or max players cap)

Outputs:
- data/atp_matches_2025_us_open_only.csv
- data/atp_matches_2025_WITH_WIMBLEDON_USOPEN.csv
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import pandas as pd

from ta_tourney_scrape.parse_profile_classic import scrape_player_classic_matches
from ta_tourney_scrape.dedupe import deduplicate_profile_matches
from ta_tourney_scrape.http import get_session


def main() -> None:
    get_session()

    base_path_candidates = [
        Path("data/atp_matches_2025_WITH_WIMBLEDON.csv"),
        Path("data/atp_matches_2025_ULTIMATE_COMPLETE.csv"),
    ]
    base_path = next((p for p in base_path_candidates if p.exists()), None)
    if base_path is None:
        raise FileNotFoundError("Could not find a base dataset to append to.")
    base_df = pd.read_csv(base_path)

    seed_players = set()
    us_existing_path = Path("data/atp_matches_2025_us_open_only.csv")
    if us_existing_path.exists():
        try:
            us_df = pd.read_csv(us_existing_path)
            if "winner_name" in us_df.columns and "loser_name" in us_df.columns:
                seed_players |= set(us_df["winner_name"].dropna().astype(str).str.strip())
                seed_players |= set(us_df["loser_name"].dropna().astype(str).str.strip())
        except Exception:
            pass

    if not seed_players:
        seed_players = {"Jannik Sinner", "Carlos Alcaraz"}

    queue = deque(sorted(p for p in seed_players if p))
    seen = set()
    all_matches: list[dict] = []

    max_players = 250
    sleep_seconds = 2.0  # polite

    print(f"🎾 US Open 2025 closure scrape starting with {len(queue)} seed players")
    print(f"   Base append target: {base_path}")
    print(f"   Max players to scrape: {max_players}")

    while queue and len(seen) < max_players:
        player = queue.popleft()
        if player in seen:
            continue
        seen.add(player)

        try:
            matches = scrape_player_classic_matches(player_name=player, year=2025, tournament_filter="US Open")
        except Exception:
            matches = []

        if matches:
            all_matches.extend(matches)
            # discover new players
            for m in matches:
                w = str(m.get("winner_name", "")).strip()
                l = str(m.get("loser_name", "")).strip()
                if w and w not in seen:
                    queue.append(w)
                if l and l not in seen:
                    queue.append(l)

        if len(seen) % 10 == 0:
            print(f"… scraped {len(seen)} players; raw US Open rows: {len(all_matches)}; queue: {len(queue)}", flush=True)

        time.sleep(sleep_seconds)

    print("\n📊 Closure scrape finished:")
    print(f"  Players scraped: {len(seen)}")
    print(f"  Raw match rows: {len(all_matches)}")

    if not all_matches:
        print("❌ No US Open matches found. Exiting.")
        return

    print("\n🔄 Deduplicating matches...")
    deduped = deduplicate_profile_matches(all_matches, year=2025)
    print(f"  Deduplicated to {len(deduped)} unique matches")

    us_df = pd.DataFrame(deduped)
    us_df["tourney_name"] = "US Open"
    us_df["surface"] = "Hard"
    us_df["tourney_level"] = "G"
    us_df["draw_size"] = 128
    us_df["year"] = 2025

    out_us_only = Path("data/atp_matches_2025_us_open_only.csv")
    us_df.to_csv(out_us_only, index=False)
    print(f"\n💾 Saved US Open-only matches to: {out_us_only} ({len(us_df)} matches)")

    # Append to base without deleting
    key_cols = ["tourney_name", "winner_name", "loser_name", "round", "score"]
    base_keys = set()
    if all(c in base_df.columns for c in key_cols):
        base_keys = set(tuple(x) for x in base_df[key_cols].fillna("").astype(str).to_numpy())

    if all(c in us_df.columns for c in key_cols) and base_keys:
        us_df["_k"] = list(map(tuple, us_df[key_cols].fillna("").astype(str).to_numpy()))
        us_df = us_df[~us_df["_k"].isin(base_keys)].drop(columns=["_k"])

    for col in base_df.columns:
        if col not in us_df.columns:
            us_df[col] = ""
    us_df = us_df[base_df.columns]

    combined = pd.concat([base_df, us_df], ignore_index=True)
    out_combined = Path("data/atp_matches_2025_WITH_WIMBLEDON_USOPEN.csv")
    combined.to_csv(out_combined, index=False)

    print(f"\n✅ Saved combined dataset to: {out_combined}")
    print(f"   Total: {len(combined)} matches")
    print(f"   Added: {len(us_df)} new US Open matches")


if __name__ == "__main__":
    main()


