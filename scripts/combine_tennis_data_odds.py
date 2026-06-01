#!/usr/bin/env python3
"""
Combine Tennis-Data ATP season Excel files into one CSV odds file.
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Combine Tennis-Data ATP odds files')
    parser.add_argument('input_files', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    frames = []
    for input_file in args.input_files:
        frame = pd.read_excel(input_file)
        frame['source_file'] = Path(input_file).name
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f'Wrote {len(combined)} rows to {args.output}')


if __name__ == '__main__':
    main()
