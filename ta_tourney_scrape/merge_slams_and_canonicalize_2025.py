"""
Merge slam classic-profile extracts into the main 2025 dataset and canonicalize tournament names.

User goals:
- Add slam datasets to our other dataset (append only truly-missing matches from classic extracts)
- Ensure the same tournament/event does not appear under multiple tournament name variants
- Deduplicate duplicate match rows (same match scraped multiple ways)

Outputs:
- data/atp_matches_2025_FINAL_CLEAN.csv
- data/intermediate/tourney_canonical_map_2025.csv
- data/output/tourney_dedupe_report_2025.md
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


BASE_PATH = Path("data/atp_matches_2025_WITH_SLM_STATS.csv")
SLAM_DIR = Path("data/slams")


def _norm_space(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _canon_slam_name(name: str) -> str:
    n = _norm_space(name)
    if re.search(r"garros", n, flags=re.IGNORECASE):
        return "Roland Garros"
    if re.search(r"wimbledon", n, flags=re.IGNORECASE):
        return "Wimbledon"
    if re.search(r"\bUS Open\b", n, flags=re.IGNORECASE):
        return "US Open"
    if re.search(r"\bAustralian Open\b", n, flags=re.IGNORECASE):
        return "Australian Open"
    return n


def _match_sig(row: pd.Series) -> Tuple[str, str, str, str, str]:
    """Signature for identifying the same match across sources."""
    t = _canon_slam_name(str(row.get("tourney_name", "")))
    r = _norm_space(str(row.get("round", ""))).upper()
    w = _norm_space(str(row.get("winner_name", "")))
    l = _norm_space(str(row.get("loser_name", "")))
    s = _norm_space(str(row.get("score", "")))
    return (t, r, w, l, s)


def _nonempty(v: object) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and pd.isna(v):
        return False
    s = str(v).strip()
    return s not in ("", "nan", "None")


def _row_score(r: pd.Series) -> int:
    """Prefer rows with more stats filled."""
    score = 0
    for col in ("winner_stats", "loser_stats", "dr", "time"):
        if col in r.index and _nonempty(r.get(col)):
            score += 1
    return score


def _build_match_sets(df: pd.DataFrame) -> Dict[str, Set[Tuple[str, str, str, str]]]:
    """tourney_name -> set(round,winner,loser,score) to compare overlaps."""
    out: Dict[str, Set[Tuple[str, str, str, str]]] = {}
    for tn, g in df.groupby("tourney_name"):
        s: Set[Tuple[str, str, str, str]] = set()
        for _, r in g.iterrows():
            s.add(
                (
                    _norm_space(str(r.get("round", ""))).upper(),
                    _norm_space(str(r.get("winner_name", ""))),
                    _norm_space(str(r.get("loser_name", ""))),
                    _norm_space(str(r.get("score", ""))),
                )
            )
        out[str(tn)] = s
    return out


@dataclass
class UnionFind:
    parent: Dict[str, str]

    def find(self, x: str) -> str:
        p = self.parent.get(x, x)
        if p != x:
            self.parent[x] = self.find(p)
        else:
            self.parent[x] = x
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def canonicalize_tournaments(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Canonicalize tournament names so the same event doesn't appear under multiple labels.

    Heuristics:
    - First, slam names are canonicalized (Roland Garros/Wimbledon/US Open/Australian Open)
    - Then, tournament labels with high match overlap are clustered (Jaccard >= 0.50, intersection >= 30)
    - Canonical label per cluster is chosen by highest stats coverage, then largest match count, then shortest name
    """
    df = df.copy()
    df["tourney_name"] = df["tourney_name"].astype(str).apply(_canon_slam_name)

    tourney_names = sorted(df["tourney_name"].dropna().astype(str).unique())
    uf = UnionFind(parent={tn: tn for tn in tourney_names})

    sets = _build_match_sets(df)

    # only compare sufficiently large tournaments
    candidates = [tn for tn in tourney_names if len(sets.get(tn, set())) >= 30]

    for i in range(len(candidates)):
        a = candidates[i]
        sa = sets.get(a, set())
        if not sa:
            continue
        for j in range(i + 1, len(candidates)):
            b = candidates[j]
            sb = sets.get(b, set())
            if not sb:
                continue
            inter = len(sa & sb)
            if inter < 30:
                continue
            jacc = inter / max(1, len(sa | sb))
            if jacc >= 0.50:
                uf.union(a, b)

    clusters: Dict[str, List[str]] = {}
    for tn in tourney_names:
        clusters.setdefault(uf.find(tn), []).append(tn)

    def stats_cov(sub: pd.DataFrame) -> float:
        cov = 0.0
        for col in ("winner_stats", "loser_stats", "dr", "time"):
            if col in sub.columns:
                cov += float((sub[col].notna() & (sub[col].astype(str).str.strip() != "") & (sub[col].astype(str) != "nan")).mean())
        return cov

    canon_for: Dict[str, str] = {}
    mapping_rows = []
    for root, members in clusters.items():
        best = None
        best_tuple = None
        for m in members:
            sub = df[df["tourney_name"] == m]
            t = (stats_cov(sub), len(sub), -len(m))  # higher cov, more rows, shorter name
            if best is None or t > best_tuple:
                best = m
                best_tuple = t
        for m in members:
            canon_for[m] = best or m
            mapping_rows.append({"tourney_name_original": m, "tourney_name_canonical": best or m})

    map_df = pd.DataFrame(mapping_rows).drop_duplicates().sort_values(["tourney_name_canonical", "tourney_name_original"])
    df["tourney_name"] = df["tourney_name"].map(canon_for).fillna(df["tourney_name"])
    return df, map_df


def main() -> None:
    if not BASE_PATH.exists():
        raise FileNotFoundError(f"Missing base file: {BASE_PATH}")

    base = pd.read_csv(BASE_PATH)

    # Load slam extracts
    slam_files = sorted(SLAM_DIR.glob("*_matches_2025_from_classic.csv"))
    slam_dfs = [pd.read_csv(p) for p in slam_files]
    slam = pd.concat(slam_dfs, ignore_index=True) if slam_dfs else pd.DataFrame()

    # Align slam extracts to base schema (keep base columns only)
    if not slam.empty:
        for col in base.columns:
            if col not in slam.columns:
                slam[col] = ""
        slam = slam[base.columns]

    # Canonicalize slam naming in both
    base["tourney_name"] = base["tourney_name"].astype(str).apply(_canon_slam_name)
    if not slam.empty:
        slam["tourney_name"] = slam["tourney_name"].astype(str).apply(_canon_slam_name)

    # Append only matches not already in base
    base_keys = set(_match_sig(r) for _, r in base.iterrows())
    add_rows = []
    if not slam.empty:
        for _, r in slam.iterrows():
            if _match_sig(r) not in base_keys:
                add_rows.append(r)
    add_df = pd.DataFrame(add_rows) if add_rows else pd.DataFrame(columns=base.columns)

    merged = pd.concat([base, add_df], ignore_index=True)

    # Canonicalize tournaments (so one event isn't represented by multiple labels)
    merged, map_df = canonicalize_tournaments(merged)

    # Match-level dedupe: keep the row with most stats
    merged["_sig"] = merged.apply(lambda r: _match_sig(r), axis=1)
    merged["_score"] = merged.apply(_row_score, axis=1)
    merged = merged.sort_values("_score", ascending=False).drop_duplicates("_sig", keep="first")
    merged = merged.drop(columns=["_sig", "_score"], errors="ignore")

    out = Path("data/atp_matches_2025_FINAL_CLEAN.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)

    map_out = Path("data/intermediate/tourney_canonical_map_2025.csv")
    map_out.parent.mkdir(parents=True, exist_ok=True)
    map_df.to_csv(map_out, index=False)

    report = []
    report.append("# Tournament canonicalization report (2025)\n\n")
    report.append(f"- Base input: `{BASE_PATH}`\n")
    report.append(f"- Slam extracts dir: `{SLAM_DIR}` ({len(slam_files)} files)\n")
    report.append(f"- Final output: `{out}`\n")
    report.append(f"- Tournament name map: `{map_out}`\n\n")
    report.append("## Row counts\n\n")
    report.append(f"- Base rows: {len(base)}\n")
    report.append(f"- Slam rows added (missing matches only): {len(add_df)}\n")
    report.append(f"- Final rows after dedupe: {len(merged)}\n\n")
    report.append("## Tournament name counts\n\n")
    report.append(f"- Unique tourney_name (base): {base['tourney_name'].nunique()}\n")
    report.append(f"- Unique tourney_name (final): {merged['tourney_name'].nunique()}\n")

    rep_out = Path("data/output/tourney_dedupe_report_2025.md")
    rep_out.parent.mkdir(parents=True, exist_ok=True)
    rep_out.write_text("".join(report), encoding="utf-8")

    print(f"✅ Wrote {out}")
    print(f"✅ Wrote {map_out}")
    print(f"✅ Wrote {rep_out}")


if __name__ == "__main__":
    main()


