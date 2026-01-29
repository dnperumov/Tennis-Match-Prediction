"""
Export enriched dataset and reconciliation report.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def compute_diff_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute difference columns between TA and calculated metrics.
    
    Args:
        df: DataFrame with ta_* and calc_* columns
        
    Returns:
        DataFrame with diff_* and abs_diff_* columns added
    """
    df = df.copy()
    
    # Metrics to compare
    metrics = [
        ('ace_pct', 'w'), ('ace_pct', 'l'),
        ('first_in_pct', 'w'), ('first_in_pct', 'l'),
        ('first_won_pct', 'w'), ('first_won_pct', 'l'),
        ('second_won_pct', 'w'), ('second_won_pct', 'l'),
        ('bpsvd', 'w'), ('bpsvd', 'l'),
        ('dr', 'w'),
    ]
    
    for metric, side in metrics:
        ta_col = f'ta_{metric}_{side}'
        calc_col = f'calc_{metric}_{side}'
        diff_col = f'diff_{metric}_{side}'
        abs_diff_col = f'abs_diff_{metric}_{side}'
        
        if ta_col in df.columns and calc_col in df.columns:
            # Compute difference
            df[diff_col] = df[ta_col] - df[calc_col]
            df[abs_diff_col] = (df[diff_col]).abs()
    
    return df


def generate_reconciliation_report(
    df: pd.DataFrame,
    output_path: str,
    year: int
):
    """
    Generate reconciliation report markdown.
    
    Args:
        df: Enriched dataframe
        output_path: Path to write report
        year: Year being processed
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    matched_df = df[df['ta_match_found'] == True].copy()
    total_matches = len(df)
    matched_count = len(matched_df)
    coverage_pct = (matched_count / total_matches * 100) if total_matches > 0 else 0
    
    report_lines = [
        f"# TennisAbstract Reconciliation Report {year}",
        "",
        "## Coverage",
        "",
        f"- **Total matches in base dataset**: {total_matches:,}",
        f"- **Matches matched to TA**: {matched_count:,}",
        f"- **Coverage**: {coverage_pct:.1f}%",
        "",
    ]
    
    # Coverage by tournament
    if 'tourney_name' in df.columns:
        report_lines.extend([
            "### Coverage by Tournament",
            "",
            "| Tournament | Total | Matched | Coverage % |",
            "|------------|-------|---------|------------|",
        ])
        
        tourney_stats = df.groupby('tourney_name').agg({
            'ta_match_found': ['count', 'sum']
        }).reset_index()
        tourney_stats.columns = ['tournament', 'total', 'matched']
        tourney_stats['coverage'] = (tourney_stats['matched'] / tourney_stats['total'] * 100).round(1)
        tourney_stats = tourney_stats.sort_values('total', ascending=False)
        
        for _, row in tourney_stats.iterrows():
            report_lines.append(
                f"| {row['tournament']} | {int(row['total'])} | {int(row['matched'])} | {row['coverage']:.1f}% |"
            )
        
        report_lines.append("")
    
    # Coverage by round
    if 'round' in df.columns:
        report_lines.extend([
            "### Coverage by Round",
            "",
            "| Round | Total | Matched | Coverage % |",
            "|-------|-------|---------|------------|",
        ])
        
        round_stats = df.groupby('round').agg({
            'ta_match_found': ['count', 'sum']
        }).reset_index()
        round_stats.columns = ['round', 'total', 'matched']
        round_stats['coverage'] = (round_stats['matched'] / round_stats['total'] * 100).round(1)
        round_stats = round_stats.sort_values('round')
        
        for _, row in round_stats.iterrows():
            report_lines.append(
                f"| {row['round']} | {int(row['total'])} | {int(row['matched'])} | {row['coverage']:.1f}% |"
            )
        
        report_lines.append("")
    
    # Metric agreement
    report_lines.extend([
        "## Metric Agreement",
        "",
        "### Absolute Differences (TA - Calculated)",
        "",
        "| Metric | Mean | Median | 95th Percentile | Mismatches (>0.02) |",
        "|--------|------|--------|------------------|-------------------|",
    ])
    
    metrics_to_check = [
        ('ace_pct', 'w'), ('ace_pct', 'l'),
        ('first_in_pct', 'w'), ('first_in_pct', 'l'),
        ('first_won_pct', 'w'), ('first_won_pct', 'l'),
        ('second_won_pct', 'w'), ('second_won_pct', 'l'),
        ('bpsvd', 'w'), ('bpsvd', 'l'),
        ('dr', 'w'),
    ]
    
    for metric, side in metrics_to_check:
        abs_diff_col = f'abs_diff_{metric}_{side}'
        if abs_diff_col in matched_df.columns:
            abs_diffs = matched_df[abs_diff_col].dropna()
            if len(abs_diffs) > 0:
                mean_diff = abs_diffs.mean()
                median_diff = abs_diffs.median()
                p95_diff = abs_diffs.quantile(0.95)
                mismatches = (abs_diffs > 0.02).sum()
                
                report_lines.append(
                    f"| {metric}_{side} | {mean_diff:.4f} | {median_diff:.4f} | {p95_diff:.4f} | {mismatches} |"
                )
    
    report_lines.append("")
    
    # Top mismatched matches
    report_lines.extend([
        "## Top 20 Most Mismatched Matches",
        "",
        "| Tournament | Round | Winner | Loser | Score | Metric | TA Value | Calculated | Diff |",
        "|------------|-------|--------|-------|-------|--------|----------|------------|------|",
    ])
    
    # Find matches with largest absolute differences
    mismatch_records = []
    for metric, side in metrics_to_check:
        abs_diff_col = f'abs_diff_{metric}_{side}'
        if abs_diff_col in matched_df.columns:
            top_mismatches = matched_df.nlargest(20, abs_diff_col)
            for _, row in top_mismatches.iterrows():
                if pd.notna(row[abs_diff_col]) and row[abs_diff_col] > 0.02:
                    mismatch_records.append({
                        'tournament': row.get('tourney_name', ''),
                        'round': row.get('round', ''),
                        'winner': row.get('winner_name', ''),
                        'loser': row.get('loser_name', ''),
                        'score': row.get('score', ''),
                        'metric': f'{metric}_{side}',
                        'ta_value': row.get(f'ta_{metric}_{side}', ''),
                        'calc_value': row.get(f'calc_{metric}_{side}', ''),
                        'diff': row[abs_diff_col]
                    })
    
    # Sort by absolute difference and take top 20
    mismatch_df = pd.DataFrame(mismatch_records)
    if len(mismatch_df) > 0:
        mismatch_df = mismatch_df.sort_values('diff', ascending=False).head(20)
        
        for _, row in mismatch_df.iterrows():
            report_lines.append(
                f"| {row['tournament']} | {row['round']} | {row['winner']} | {row['loser']} | "
                f"{row['score']} | {row['metric']} | {row['ta_value']:.4f} | {row['calc_value']:.4f} | {row['diff']:.4f} |"
            )
    
    report_lines.append("")
    report_lines.extend([
        "## Notes",
        "",
        "- **Mismatch threshold**: Absolute difference > 0.02 (2 percentage points)",
        "- **Hold%**: Not computed from base dataset counts (requires actual breaks, not just break points)",
        "- **Join quality**: 'exact' = exact match on all fields; 'relaxed_score' = match on tournament/round/players with score variation",
    ])
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Reconciliation report written to {output_path}")


def export_enriched_dataset(df: pd.DataFrame, output_path: str):
    """
    Export enriched dataset to CSV.
    
    Args:
        df: Enriched dataframe
        output_path: Path to write CSV
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove intermediate columns used for matching
    cols_to_drop = [
        'winner_name_norm', 'loser_name_norm', 'score_norm', 'round_norm',
        'tournament_key', 'match_key', 'year'
    ]
    
    export_df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    export_df.to_csv(output_path, index=False)
    logger.info(f"Enriched dataset written to {output_path} ({len(export_df)} rows)")

