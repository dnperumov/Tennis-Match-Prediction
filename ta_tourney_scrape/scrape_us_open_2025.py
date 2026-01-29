"""
Scrape US Open 2025 matches from TennisAbstract classic player pages.

Strategy:
- Use the existing dataset's player universe as the seed list (winners/losers seen in 2025 so far)
- For each player, scrape their US Open 2025 rows from `player-classic.cgi`
- Deduplicate (same match appears on both players' pages) and merge winner/loser-side stats where possible
- Append to the existing dataset WITHOUT removing existing rows

Outputs:
- data/atp_matches_2025_us_open_only.csv
- data/atp_matches_2025_WITH_WIMBLEDON_USOPEN.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ta_tourney_scrape.parse_profile_classic import scrape_players_classic_matches
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
        raise FileNotFoundError("Could not find a base dataset to seed player list.")

    base_df = pd.read_csv(base_path)
    if "winner_name" not in base_df.columns or "loser_name" not in base_df.columns:
        raise ValueError(f"Base dataset {base_path} missing winner_name/loser_name columns.")

    players = (
        pd.concat([base_df["winner_name"], base_df["loser_name"]], ignore_index=True)
        .dropna()
        .astype(str)
        .str.strip()
    )
    players = sorted({p for p in players if p})

    print(f"🎾 Scraping US Open 2025 from {len(players)} seeded player profiles...")
    print("   Seed source:", base_path)

    # Bulk scrape with browser reuse for speed/politeness
    all_matches = scrape_players_classic_matches(
        player_names=players,
        year=2025,
        tournament_filter="US Open",
        sleep_seconds=0.25,
    )

    print("\n📊 Scraping complete:")
    print(f"  Total raw matches scraped: {len(all_matches)}")
    # Failures are logged inside the bulk scraper

    if not all_matches:
        print("\n❌ No US Open matches scraped. Exiting.")
        return

    print("\n🔄 Deduplicating matches...")
    deduped = deduplicate_profile_matches(all_matches, year=2025)
    print(f"  Deduplicated to {len(deduped)} unique matches")

    us_df = pd.DataFrame(deduped)

    # Standardize metadata
    us_df["tourney_name"] = "US Open"
    us_df["surface"] = "Hard"
    us_df["tourney_level"] = "G"
    us_df["draw_size"] = 128
    us_df["year"] = 2025

    out_us_only = Path("data/atp_matches_2025_us_open_only.csv")
    us_df.to_csv(out_us_only, index=False)
    print(f"\n💾 Saved US Open-only matches to: {out_us_only} ({len(us_df)} matches)")

    # Append to base without deleting base rows.
    key_cols = ["tourney_name", "winner_name", "loser_name", "round", "score"]
    base_keys = set()
    if all(c in base_df.columns for c in key_cols):
        base_keys = set(tuple(x) for x in base_df[key_cols].fillna("").astype(str).to_numpy())

    if all(c in us_df.columns for c in key_cols) and base_keys:
        us_df["_k"] = list(map(tuple, us_df[key_cols].fillna("").astype(str).to_numpy()))
        us_df = us_df[~us_df["_k"].isin(base_keys)].drop(columns=["_k"])

    # Align columns to base
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


