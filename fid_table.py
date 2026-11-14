"""Render ``fid_results.jsonl`` as a paper-ready comparison table.

Usage
-----
    python fid_table.py --json fid_results.jsonl
    python fid_table.py --json fid_results.jsonl --baseline meanflow_imf30k
    python fid_table.py --json fid_results.jsonl --baseline meanflow_imf30k --out table.md

The **NFE** column is the number of network forward passes per sample:
1 for one-step, 2 for two-step, and 6 for AF-PSM adaptive over two intervals
(alpha prediction + two segments per interval). Compare models at equal NFE,
not at equal "steps": an adaptive 2-interval run costs three times a plain
2-step run.

The delta column matches the baseline by NFE, and reports
``no NFE-matched baseline`` when there is no baseline row at that NFE.
"""
import argparse
import json
import os
import sys


def parse_mode(mode):
    """'1' -> (1, False), '2' -> (2, False), '2a'/'adaptive' -> (2, True)"""
    s = str(mode).strip().lower()
    if s == 'adaptive':
        return 2, True
    if s.endswith('a'):
        return int(s[:-1]), True
    return int(s), False


def row_nfe(row):
    """NFE of a results row, recomputed for jsonl files written before --nfe."""
    if row.get('nfe') is not None:
        return int(row['nfe'])
    steps, adaptive = parse_mode(row.get('mode', '1'))
    if adaptive and 'AFPSM' in str(row.get('sampler', '')):
        return steps * 3
    return steps


def load_rows(path):
    rows = []
    with open(path) as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f'[warn] {path}:{line_no}: not valid JSON, skipped', file=sys.stderr)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', default='fid_results.jsonl')
    parser.add_argument('--baseline', default=None,
                        help='model name whose FID is subtracted (matched by NFE)')
    parser.add_argument('--out', default=None, help='also write the markdown here')
    parser.add_argument('--no_uniq', action='store_true',
                        help='keep every row (default: keep the last row per name/mode/NFE)')
    args = parser.parse_args()

    if not os.path.exists(args.json):
        raise SystemExit(f'no such file: {args.json}  (run fid_eval.py --json {args.json} first)')
    rows = load_rows(args.json)
    if not rows:
        raise SystemExit(f'{args.json} contains no results')

    for row in rows:
        row['nfe'] = row_nfe(row)

    if not args.no_uniq:
        # re-running fid_eval appends; keep only the latest row per (name, mode, NFE)
        uniq = {}
        for row in rows:
            uniq[(row.get('name', '?'), str(row.get('mode', '?')), row['nfe'])] = row
        rows = list(uniq.values())

    rows.sort(key=lambda r: (str(r.get('name', '?')), r['nfe'], str(r.get('mode'))))

    base_fid_by_nfe = {}
    if args.baseline:
        for row in rows:
            if row.get('name') == args.baseline:
                base_fid_by_nfe[row['nfe']] = float(row['fid'])
        if not base_fid_by_nfe:
            print(f'[warn] baseline "{args.baseline}" not found in {args.json}', file=sys.stderr)

    lines = ['| model | mode | NFE | n | FID | delta vs baseline |',
             '|---|---|---:|---:|---:|---:|']
    for row in rows:
        fid = float(row['fid'])
        delta = ''
        if args.baseline and row.get('name') != args.baseline:
            if row['nfe'] in base_fid_by_nfe:
                delta = f'{fid - base_fid_by_nfe[row["nfe"]]:+.2f}'
            else:
                delta = 'no NFE-matched baseline'
        lines.append(
            f"| {row.get('name', '?')} | {row.get('mode', '?')} | {row['nfe']} | "
            f"{row.get('n', '?')} | {fid:.2f} | {delta} |"
        )

    text = '\n'.join(lines)
    print(text)
    print()
    print('NFE = forward passes per sample. Compare at equal NFE; AF-PSM adaptive '
          '(2a, NFE=6) must be compared against a 6-NFE baseline, not a 2-step one.')
    if args.out:
        with open(args.out, 'w') as handle:
            handle.write(text + '\n')
        print(f'\nwrote {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
