"""
Compute derived percentage metrics from count statistics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


def compute_winner_metrics(row: pd.Series) -> Dict[str, Optional[float]]:
    """
    Compute derived metrics for winner from count stats.
    
    Args:
        row: DataFrame row with w_ace, w_df, w_svpt, etc.
        
    Returns:
        Dictionary of computed metrics
    """
    metrics = {}
    
    # Ace percentage
    if pd.notna(row.get('w_svpt')) and row['w_svpt'] > 0:
        if pd.notna(row.get('w_ace')):
            metrics['calc_ace_pct_w'] = row['w_ace'] / row['w_svpt']
        else:
            metrics['calc_ace_pct_w'] = None
    else:
        metrics['calc_ace_pct_w'] = None
    
    # Double fault percentage
    if pd.notna(row.get('w_svpt')) and row['w_svpt'] > 0:
        if pd.notna(row.get('w_df')):
            metrics['calc_df_pct_w'] = row['w_df'] / row['w_svpt']
        else:
            metrics['calc_df_pct_w'] = None
    else:
        metrics['calc_df_pct_w'] = None
    
    # First serve in percentage
    if pd.notna(row.get('w_svpt')) and row['w_svpt'] > 0:
        if pd.notna(row.get('w_1stIn')):
            metrics['calc_first_in_pct_w'] = row['w_1stIn'] / row['w_svpt']
        else:
            metrics['calc_first_in_pct_w'] = None
    else:
        metrics['calc_first_in_pct_w'] = None
    
    # First serve won percentage
    if pd.notna(row.get('w_1stIn')) and row['w_1stIn'] > 0:
        if pd.notna(row.get('w_1stWon')):
            metrics['calc_first_won_pct_w'] = row['w_1stWon'] / row['w_1stIn']
        else:
            metrics['calc_first_won_pct_w'] = None
    else:
        metrics['calc_first_won_pct_w'] = None
    
    # Second serve won percentage
    if pd.notna(row.get('w_svpt')) and pd.notna(row.get('w_1stIn')):
        second_serves = row['w_svpt'] - row['w_1stIn']
        if second_serves > 0 and pd.notna(row.get('w_2ndWon')):
            metrics['calc_second_won_pct_w'] = row['w_2ndWon'] / second_serves
        else:
            metrics['calc_second_won_pct_w'] = None
    else:
        metrics['calc_second_won_pct_w'] = None
    
    # Service points won (SPW)
    if pd.notna(row.get('w_svpt')) and row['w_svpt'] > 0:
        total_won = 0
        if pd.notna(row.get('w_1stWon')):
            total_won += row['w_1stWon']
        if pd.notna(row.get('w_2ndWon')):
            total_won += row['w_2ndWon']
        metrics['calc_spw_w'] = total_won / row['w_svpt']
    else:
        metrics['calc_spw_w'] = None
    
    # Break points saved percentage
    if pd.notna(row.get('w_bpFaced')) and row['w_bpFaced'] > 0:
        if pd.notna(row.get('w_bpSaved')):
            metrics['calc_bpsvd_pct_w'] = row['w_bpSaved'] / row['w_bpFaced']
        else:
            metrics['calc_bpsvd_pct_w'] = None
    else:
        metrics['calc_bpsvd_pct_w'] = None
    
    # Return points won (RPW) - from loser's serve stats
    if pd.notna(row.get('l_svpt')) and row['l_svpt'] > 0:
        return_points_won = row['l_svpt']
        if pd.notna(row.get('l_1stWon')):
            return_points_won -= row['l_1stWon']
        if pd.notna(row.get('l_2ndWon')):
            return_points_won -= row['l_2ndWon']
        metrics['calc_rpw_w'] = return_points_won / row['l_svpt']
    else:
        metrics['calc_rpw_w'] = None
    
    # Total points won (TPW)
    if (pd.notna(row.get('w_svpt')) and pd.notna(row.get('l_svpt')) and 
        (row['w_svpt'] + row['l_svpt']) > 0):
        total_points = row['w_svpt'] + row['l_svpt']
        total_won = 0
        if pd.notna(row.get('w_1stWon')):
            total_won += row['w_1stWon']
        if pd.notna(row.get('w_2ndWon')):
            total_won += row['w_2ndWon']
        if pd.notna(row.get('l_svpt')):
            return_won = row['l_svpt']
            if pd.notna(row.get('l_1stWon')):
                return_won -= row['l_1stWon']
            if pd.notna(row.get('l_2ndWon')):
                return_won -= row['l_2ndWon']
            total_won += return_won
        metrics['calc_tpw_w'] = total_won / total_points
    else:
        metrics['calc_tpw_w'] = None
    
    # Break percentage (Brk%) - break points converted / break points faced by opponent
    if pd.notna(row.get('l_bpFaced')) and row['l_bpFaced'] > 0:
        bp_converted = row['l_bpFaced']
        if pd.notna(row.get('l_bpSaved')):
            bp_converted -= row['l_bpSaved']
        metrics['calc_brk_pct_w'] = bp_converted / row['l_bpFaced']
    else:
        metrics['calc_brk_pct_w'] = None
    
    # Dominance Ratio (DR) = RPW / (1 - SPW)
    if (metrics.get('calc_rpw_w') is not None and 
        metrics.get('calc_spw_w') is not None and
        metrics['calc_spw_w'] < 1.0):
        metrics['calc_dr_w'] = metrics['calc_rpw_w'] / (1.0 - metrics['calc_spw_w'])
    else:
        metrics['calc_dr_w'] = None
    
    return metrics


def compute_loser_metrics(row: pd.Series) -> Dict[str, Optional[float]]:
    """
    Compute derived metrics for loser from count stats.
    
    Args:
        row: DataFrame row with l_ace, l_df, l_svpt, etc.
        
    Returns:
        Dictionary of computed metrics
    """
    metrics = {}
    
    # Same calculations but for loser (swap w_* and l_*)
    # Ace percentage
    if pd.notna(row.get('l_svpt')) and row['l_svpt'] > 0:
        if pd.notna(row.get('l_ace')):
            metrics['calc_ace_pct_l'] = row['l_ace'] / row['l_svpt']
        else:
            metrics['calc_ace_pct_l'] = None
    else:
        metrics['calc_ace_pct_l'] = None
    
    # Double fault percentage
    if pd.notna(row.get('l_svpt')) and row['l_svpt'] > 0:
        if pd.notna(row.get('l_df')):
            metrics['calc_df_pct_l'] = row['l_df'] / row['l_svpt']
        else:
            metrics['calc_df_pct_l'] = None
    else:
        metrics['calc_df_pct_l'] = None
    
    # First serve in percentage
    if pd.notna(row.get('l_svpt')) and row['l_svpt'] > 0:
        if pd.notna(row.get('l_1stIn')):
            metrics['calc_first_in_pct_l'] = row['l_1stIn'] / row['l_svpt']
        else:
            metrics['calc_first_in_pct_l'] = None
    else:
        metrics['calc_first_in_pct_l'] = None
    
    # First serve won percentage
    if pd.notna(row.get('l_1stIn')) and row['l_1stIn'] > 0:
        if pd.notna(row.get('l_1stWon')):
            metrics['calc_first_won_pct_l'] = row['l_1stWon'] / row['l_1stIn']
        else:
            metrics['calc_first_won_pct_l'] = None
    else:
        metrics['calc_first_won_pct_l'] = None
    
    # Second serve won percentage
    if pd.notna(row.get('l_svpt')) and pd.notna(row.get('l_1stIn')):
        second_serves = row['l_svpt'] - row['l_1stIn']
        if second_serves > 0 and pd.notna(row.get('l_2ndWon')):
            metrics['calc_second_won_pct_l'] = row['l_2ndWon'] / second_serves
        else:
            metrics['calc_second_won_pct_l'] = None
    else:
        metrics['calc_second_won_pct_l'] = None
    
    # Service points won (SPW)
    if pd.notna(row.get('l_svpt')) and row['l_svpt'] > 0:
        total_won = 0
        if pd.notna(row.get('l_1stWon')):
            total_won += row['l_1stWon']
        if pd.notna(row.get('l_2ndWon')):
            total_won += row['l_2ndWon']
        metrics['calc_spw_l'] = total_won / row['l_svpt']
    else:
        metrics['calc_spw_l'] = None
    
    # Break points saved percentage
    if pd.notna(row.get('l_bpFaced')) and row['l_bpFaced'] > 0:
        if pd.notna(row.get('l_bpSaved')):
            metrics['calc_bpsvd_pct_l'] = row['l_bpSaved'] / row['l_bpFaced']
        else:
            metrics['calc_bpsvd_pct_l'] = None
    else:
        metrics['calc_bpsvd_pct_l'] = None
    
    # Return points won (RPW) - from winner's serve stats
    if pd.notna(row.get('w_svpt')) and row['w_svpt'] > 0:
        return_points_won = row['w_svpt']
        if pd.notna(row.get('w_1stWon')):
            return_points_won -= row['w_1stWon']
        if pd.notna(row.get('w_2ndWon')):
            return_points_won -= row['w_2ndWon']
        metrics['calc_rpw_l'] = return_points_won / row['w_svpt']
    else:
        metrics['calc_rpw_l'] = None
    
    # Total points won (TPW)
    if (pd.notna(row.get('w_svpt')) and pd.notna(row.get('l_svpt')) and 
        (row['w_svpt'] + row['l_svpt']) > 0):
        total_points = row['w_svpt'] + row['l_svpt']
        total_won = 0
        if pd.notna(row.get('l_1stWon')):
            total_won += row['l_1stWon']
        if pd.notna(row.get('l_2ndWon')):
            total_won += row['l_2ndWon']
        if pd.notna(row.get('w_svpt')):
            return_won = row['w_svpt']
            if pd.notna(row.get('w_1stWon')):
                return_won -= row['w_1stWon']
            if pd.notna(row.get('w_2ndWon')):
                return_won -= row['w_2ndWon']
            total_won += return_won
        metrics['calc_tpw_l'] = total_won / total_points
    else:
        metrics['calc_tpw_l'] = None
    
    # Break percentage (Brk%)
    if pd.notna(row.get('w_bpFaced')) and row['w_bpFaced'] > 0:
        bp_converted = row['w_bpFaced']
        if pd.notna(row.get('w_bpSaved')):
            bp_converted -= row['w_bpSaved']
        metrics['calc_brk_pct_l'] = bp_converted / row['w_bpFaced']
    else:
        metrics['calc_brk_pct_l'] = None
    
    # Dominance Ratio (DR)
    if (metrics.get('calc_rpw_l') is not None and 
        metrics.get('calc_spw_l') is not None and
        metrics['calc_spw_l'] < 1.0):
        metrics['calc_dr_l'] = metrics['calc_rpw_l'] / (1.0 - metrics['calc_spw_l'])
    else:
        metrics['calc_dr_l'] = None
    
    return metrics


def compute_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all derived metrics for a dataframe.
    
    Args:
        df: DataFrame with count statistics
        
    Returns:
        DataFrame with computed metrics added
    """
    df = df.copy()
    
    # Compute winner metrics
    winner_metrics_list = df.apply(compute_winner_metrics, axis=1).tolist()
    if winner_metrics_list and len(winner_metrics_list) > 0:
        for key in winner_metrics_list[0].keys():
            df[key] = [m.get(key) for m in winner_metrics_list]
    
    # Compute loser metrics
    loser_metrics_list = df.apply(compute_loser_metrics, axis=1).tolist()
    if loser_metrics_list and len(loser_metrics_list) > 0:
        for key in loser_metrics_list[0].keys():
            df[key] = [m.get(key) for m in loser_metrics_list]
    
    return df

