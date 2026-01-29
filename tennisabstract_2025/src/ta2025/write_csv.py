"""
Write normalized records to CSV in exact column order.
"""

import csv
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def write_csv(records: List[Dict[str, Any]], 
              output_path: str,
              column_list: List[str]):
    """
    Write records to CSV with exact column order.
    
    Args:
        records: List of normalized record dictionaries
        output_path: Path to output CSV file
        column_list: List of column names in exact order
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Writing {len(records)} records to {output_path}")
    logger.info(f"Columns ({len(column_list)}): {', '.join(column_list[:5])}...")
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=column_list, extrasaction='ignore')
        writer.writeheader()
        
        for record in records:
            # Ensure all columns are present, fill missing with ""
            row = {col: record.get(col, "") for col in column_list}
            writer.writerow(row)
    
    logger.info(f"Successfully wrote CSV to {output_path}")

