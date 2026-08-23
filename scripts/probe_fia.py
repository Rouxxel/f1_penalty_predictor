#!/usr/bin/env python3
"""Probe whether FIA document pages are reachable from this machine."""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fia_ml.data.config import PipelineConfig
from fia_ml.data.fia_http import create_fia_session, fetch_fia_html, warmup_fia_session

EVENT_PATTERN = re.compile(
    r'value="(/documents/championships/fia-formula-one-world-championship-14/season/[^"]+/event/[^"]+)"'
)


def main() -> int:
    cfg = PipelineConfig.from_yaml()
    season_cfg = cfg.for_season(2020)
    url = season_cfg.season_url

    print("FIA document access probe")
    print(f"season url: {url}\n")

    session = create_fia_session(season_cfg)
    try:
        warmup_fia_session(session, season_cfg, url)
        html = fetch_fia_html(session, season_cfg, url, warmed=True)
    except Exception as exc:
        print(f"FAILED: {exc}")
        print(
            "\nIf your browser can open the docs via the FIA homepage but this script cannot,"
            "\nFIA is blocking automated clients (not just rate limits)."
            "\nTry waiting 24h, then one season at a time. If it still fails, use browser"
            "\nautomation (Playwright) or download PDFs manually into data/raw/fia/{season}/."
        )
        return 1

    events = len(EVENT_PATTERN.findall(html))
    print(f"OK: fetched {len(html)} bytes, found {events} event options")
    if events == 0:
        print("WARNING: page loaded but no events found — HTML layout may have changed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
