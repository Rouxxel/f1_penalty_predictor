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
from fia_ml.data.fia_http import create_fia_client

EVENT_PATTERN = re.compile(
    r'value="(/documents/championships/fia-formula-one-world-championship-14/season/[^"]+/event/[^"]+)"'
)


def main() -> int:
    cfg = PipelineConfig.from_yaml()
    season_cfg = cfg.for_season(2020)
    url = season_cfg.season_url
    backend = season_cfg.scraper.get("fetch_backend", "playwright")

    print("FIA document access probe")
    print(f"backend: {backend}")
    print(f"season url: {url}\n")

    client = create_fia_client(season_cfg)
    try:
        html = client.fetch_html(url)
    except Exception as exc:
        print(f"FAILED: {exc}")
        if backend == "requests":
            print(
                "\nTry Playwright instead:"
                "\n  pip install playwright"
                "\n  playwright install chromium"
                "\nThen set scraper.fetch_backend: playwright in configs/data.yaml"
            )
        return 1
    finally:
        client.close()

    events = len(EVENT_PATTERN.findall(html))
    print(f"OK: fetched {len(html)} bytes, found {events} event options")
    if events == 0:
        print("WARNING: page loaded but no events found — HTML layout may have changed.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
