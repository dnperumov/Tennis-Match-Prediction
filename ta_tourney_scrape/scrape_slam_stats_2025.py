"""
Backfill missing match-level % stats for 2025 Grand Slams from TennisAbstract classic player pages.

What it does:
- Builds (or loads) the player list for a slam and saves it to disk for reproducibility
- Scrapes each player's classic page rows for the slam (e.g. "Roland Garros") and collects per-player % stats
- Deduplicates into unique matches and merges winner/loser-side stats
- Upserts stats into an existing dataset (fills winner_stats/loser_stats/dr/time where missing)

Outputs:
- data/intermediate/slams_2025/{slug}_players_2025.csv
- data/slams/{slug}_matches_2025_from_classic.csv
- data/atp_matches_2025_WITH_SLM_STATS.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import ast
from collections import deque

from ta_tourney_scrape.parse_profile_classic import scrape_players_classic_matches
from ta_tourney_scrape.dedupe import deduplicate_profile_matches
from ta_tourney_scrape.normalize import normalize_player_name, normalize_score, normalize_round, create_match_key
from ta_tourney_scrape.http import get_session
from ta_tourney_scrape.ta_names import resolve_ta_player_name


SLAM_CANON = {
    "roland_garros": "Roland Garros",
    "wimbledon": "Wimbledon",
    "us_open": "US Open",
}


def _slugify(s: str) -> str:
    return s.lower().replace(" ", "_")


def _ensure_dirs() -> None:
    Path("data/intermediate/slams_2025").mkdir(parents=True, exist_ok=True)
    Path("data/slams").mkdir(parents=True, exist_ok=True)


def _load_or_build_players(base_df: pd.DataFrame, slam_key: str, slam_name: str, player_csv: Optional[str]) -> pd.DataFrame:
    if player_csv:
        p = Path(player_csv)
        df = pd.read_csv(p)
        if "player_name" not in df.columns:
            raise ValueError(f"{p} must contain a 'player_name' column.")
        df["player_name"] = df["player_name"].astype(str).str.strip()
        df = df[df["player_name"] != ""].drop_duplicates(subset=["player_name"])
        return df

    # Derive from base by filtering tourney_name text and pulling unique players
    if slam_key == "roland_garros":
        mask = base_df["tourney_name"].astype(str).str.contains("Garros", case=False, na=False)
    elif slam_key == "us_open":
        mask = base_df["tourney_name"].astype(str).str.contains("US Open", case=False, na=False)
    elif slam_key == "wimbledon":
        mask = base_df["tourney_name"].astype(str).str.contains("Wimbledon", case=False, na=False)
    else:
        mask = base_df["tourney_name"].astype(str).str.contains(slam_name, case=False, na=False)
    sub = base_df[mask]
    players = pd.concat([sub["winner_name"], sub["loser_name"]], ignore_index=True).dropna().astype(str).str.strip()
    players = sorted({p for p in players if p})
    return pd.DataFrame({"player_name": players})


def _mk_key_for_row(row: pd.Series, year: int, slam_canon: str) -> str:
    # Use canonical slam label as "tournament_key" for stable join
    return create_match_key(
        year,
        slam_canon,
        normalize_round(str(row.get("round", ""))),
        str(row.get("winner_name", "")),
        str(row.get("loser_name", "")),
        normalize_score(str(row.get("score", ""))),
        match_date=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slam", required=True, choices=list(SLAM_CANON.keys()))
    parser.add_argument("--base", default="data/atp_matches_2025_WITH_WIMBLEDON_USOPEN.csv")
    parser.add_argument("--players-csv", default=None, help="Optional CSV with column player_name")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--closure", action="store_true", help="If set, do opponent-closure crawl to capture more slam matches/stats.")
    parser.add_argument("--max-players", type=int, default=300, help="Max players to scrape in closure mode.")
    parser.add_argument("--sleep", type=float, default=0.25, help="Sleep seconds between player fetches.")
    parser.add_argument("--batch-size", type=int, default=25, help="Batch size for bulk scraping in closure mode.")
    args = parser.parse_args()

    get_session()
    _ensure_dirs()

    slam_canon = SLAM_CANON[args.slam]
    slug = args.slam

    base_path = Path(args.base)
    if not base_path.exists():
        raise FileNotFoundError(f"Base dataset not found: {base_path}")
    base_df = pd.read_csv(base_path)

    # Build/save player list
    players_df = _load_or_build_players(base_df, slam_key=args.slam, slam_name=slam_canon, player_csv=args.players_csv)
    players_out = Path(f"data/intermediate/slams_2025/{slug}_players_2025.csv")
    players_df.to_csv(players_out, index=False)
    print(f"✅ Saved player list: {players_out} ({len(players_df)} players)")

    player_names = players_df["player_name"].tolist()
    resolved = [resolve_ta_player_name(p) for p in player_names]
    players_df["player_name_resolved"] = resolved
    players_df.to_csv(players_out, index=False)
    player_names = players_df["player_name_resolved"].dropna().astype(str).str.strip().tolist()

    if not args.closure:
        print(f"🎾 Scraping {slam_canon} {args.year} from {len(player_names)} classic profiles...")
        rows = scrape_players_classic_matches(
            player_names,
            year=args.year,
            tournament_filter=slam_canon,
            sleep_seconds=args.sleep,
        )
    else:
        print(f"🎾 Closure scraping {slam_canon} {args.year}")
        print(f"   seed={len(player_names)} max_players={args.max_players} batch_size={args.batch_size}")
        seen = set()
        q = deque([p for p in player_names if p])
        rows = []

        while q and len(seen) < args.max_players:
            batch = []
            while q and len(batch) < args.batch_size and len(seen) < args.max_players:
                p = q.popleft()
                if p in seen:
                    continue
                seen.add(p)
                batch.append(p)

            if not batch:
                continue

            batch_rows = scrape_players_classic_matches(
                batch,
                year=args.year,
                tournament_filter=slam_canon,
                sleep_seconds=args.sleep,
            )
            rows.extend(batch_rows)

            for r in batch_rows:
                w = str(r.get("winner_name", "")).strip()
                l = str(r.get("loser_name", "")).strip()
                if w and w not in seen:
                    q.append(w)
                if l and l not in seen:
                    q.append(l)

            print(f"… closure progress players={len(seen)} rows={len(rows)} queue={len(q)}", flush=True)
    print(f"\n📊 Raw rows scraped: {len(rows)}")

    if not rows:
        print("❌ No rows scraped. Exiting.")
        return

    merged_matches = deduplicate_profile_matches(rows, year=args.year)
    slam_df = pd.DataFrame(merged_matches)

    # Keep only the relevant rounds (main draw + Q if present)
    slam_df["round_norm"] = slam_df["round"].astype(str).apply(normalize_round)
    slam_df = slam_df.drop(columns=["round_norm"], errors="ignore")

    # Normalize tourney_name to canonical for output consistency
    slam_df["tourney_name"] = slam_canon

    slam_out = Path(f"data/slams/{slug}_matches_2025_from_classic.csv")
    slam_df.to_csv(slam_out, index=False)
    print(f"✅ Wrote slam match stats: {slam_out} ({len(slam_df)} matches)")

    # Build match keys for upsert
    slam_df["_mk"] = slam_df.apply(lambda r: _mk_key_for_row(r, args.year, slam_canon), axis=1)

    # Identify slam rows in base and build keys
    base_mask = base_df["tourney_name"].astype(str).str.contains(slam_canon, case=False, na=False) | (
        (args.slam == "roland_garros") & base_df["tourney_name"].astype(str).str.contains("Garros", case=False, na=False)
    )
    base_slam = base_df[base_mask].copy()
    base_slam["_mk"] = base_slam.apply(lambda r: _mk_key_for_row(r, args.year, slam_canon), axis=1)

    # Map stats into base
    slam_lookup = slam_df.set_index("_mk").to_dict(orient="index")

    def _fill_if_missing(dst, src, col):
        if col not in src:
            return dst
        v = src.get(col)
        if col in ("winner_stats", "loser_stats"):
            # Merge dict contents (base has stringified dicts)
            base_val = dst.get(col)
            base_dict = {}
            if isinstance(base_val, dict):
                base_dict = base_val
            elif isinstance(base_val, str) and base_val.strip():
                try:
                    base_dict = ast.literal_eval(base_val)
                    if not isinstance(base_dict, dict):
                        base_dict = {}
                except Exception:
                    base_dict = {}
            src_dict = v if isinstance(v, dict) else {}
            if not isinstance(src_dict, dict):
                src_dict = {}
            merged = dict(base_dict)
            for k, vv in src_dict.items():
                if k not in merged or merged.get(k) in ("", None) or (isinstance(merged.get(k), float) and pd.isna(merged.get(k))):
                    merged[k] = vv
            dst[col] = merged
            return dst

        if col not in dst or pd.isna(dst.get(col)) or dst.get(col) in ("", "nan", None):
            dst[col] = v
        return dst

    updated_rows = 0
    for idx, row in base_slam.iterrows():
        mk = row["_mk"]
        src = slam_lookup.get(mk)
        if not src:
            continue
        dst = base_df.loc[idx].to_dict()
        before = (dst.get("winner_stats"), dst.get("loser_stats"), dst.get("dr"), dst.get("time"))
        for col in ["winner_stats", "loser_stats", "dr", "time", "winner_rank", "loser_rank"]:
            dst = _fill_if_missing(dst, src, col)
        after = (dst.get("winner_stats"), dst.get("loser_stats"), dst.get("dr"), dst.get("time"))
        if after != before:
            updated_rows += 1
            for k, v in dst.items():
                base_df.at[idx, k] = v

    out_path = Path("data/atp_matches_2025_WITH_SLM_STATS.csv")
    base_df.to_csv(out_path, index=False)
    print(f"✅ Upserted stats into base: {out_path}")
    print(f"   Rows updated for {slam_canon}: {updated_rows}")


if __name__ == "__main__":
    main()


