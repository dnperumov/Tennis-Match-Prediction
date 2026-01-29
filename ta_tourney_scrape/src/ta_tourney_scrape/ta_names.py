"""
Resolve player display names to the canonical TennisAbstract name key used in JS files and player pages.

Why:
- Some names in our datasets are reversed (e.g., "Yunchaokete Bu" vs "Bu Yunchaokete")
- Some names have accents/hyphens that need normalization
- Using TA's own current ranking name list gives us a best-effort canonicalization
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Dict, Optional

from unidecode import unidecode

from .http import get_session


_CURR_RANK_JS_URL = "https://www.tennisabstract.com/jsplayers/curr_rank_atp.js"


def _norm(s: str) -> str:
    s = unidecode(s or "")
    s = re.sub(r"\s+", " ", s).strip().upper()
    return s


@lru_cache(maxsize=1)
def load_curr_rank_name_map() -> Dict[str, str]:
    """
    Returns mapping: NORMALIZED_NAME -> Canonical TA name (as used in currRank keys).
    """
    session = get_session()
    resp = session.session.get(_CURR_RANK_JS_URL)
    resp.raise_for_status()
    text = resp.text

    # Extract JS object: var currRank = {...};
    m = re.search(r"var\s+currRank\s*=\s*(\{.*?\});", text, flags=re.DOTALL)
    if not m:
        return {}
    obj_text = m.group(1)
    # json.loads requires double quotes; this file is already JSON-like
    curr = json.loads(obj_text)

    out: Dict[str, str] = {}
    for name in curr.keys():
        out[_norm(name)] = name
    return out


def resolve_ta_player_name(name: str) -> str:
    """
    Best-effort resolution to TA's canonical display name.
    If no match found, returns original name.
    """
    name = (name or "").strip()
    if not name:
        return ""

    mp = load_curr_rank_name_map()
    if not mp:
        return name

    n = _norm(name)
    if n in mp:
        return mp[n]

    # Try swapping first/last order (helps for some datasets)
    parts = re.split(r"\s+", name.strip())
    if len(parts) >= 2:
        swapped = " ".join([parts[-1]] + parts[:-1])
        ns = _norm(swapped)
        if ns in mp:
            return mp[ns]

    return name


def resolve_optional(name: str) -> Optional[str]:
    r = resolve_ta_player_name(name)
    return r if r else None


