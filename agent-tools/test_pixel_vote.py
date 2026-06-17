#!/usr/bin/env python3
"""Quick test for pixel_wise_majority_vote_grids logic."""
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Minimal copies to avoid full import
def _is_valid_arc_grid(parsed: Any) -> bool:
    if not isinstance(parsed, list) or not parsed:
        return False
    return all(isinstance(row, list) and all(isinstance(c, int) for c in row) for row in parsed)


def pixel_wise_majority_vote_grids(grid_hypotheses):
    parsed = []
    for rec in grid_hypotheses:
        grid = rec.get("parsed_grid")
        if grid is not None and _is_valid_arc_grid(grid):
            parsed.append(grid)
    shape_counts = Counter((len(g), len(g[0])) for g in parsed)
    target_h, target_w = shape_counts.most_common(1)[0][0]
    eligible = [g for g in parsed if len(g) == target_h and len(g[0]) == target_w]
    result = []
    for r in range(target_h):
        row = []
        for c in range(target_w):
            votes = [int(g[r][c]) for g in eligible]
            row.append(Counter(votes).most_common(1)[0][0])
        result.append(row)
    return result


hyps = [
    {"parsed_grid": [[1, 2], [3, 1]]},
    {"parsed_grid": [[1, 2], [3, 2]]},
    {"parsed_grid": [[1, 2], [3, 1]]},
]
out = pixel_wise_majority_vote_grids(hyps)
assert out == [[1, 2], [3, 1]], out
print("pixel_vote_ok", json.dumps(out))