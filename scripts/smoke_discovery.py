"""Smoke-test FIA discovery without downloading every PDF."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fia_ml.data.config import PipelineConfig
from fia_ml.data.download import discover_event_urls, discover_pdfs_for_event

year = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
cfg = PipelineConfig.from_yaml(ROOT / "configs" / "data.yaml").for_season(year)
events = discover_event_urls(cfg.season_url, cfg)
counts: list[tuple[str, int]] = []
for name, url in events:
    pdfs = discover_pdfs_for_event(url, name, cfg)
    counts.append((name, len(pdfs)))

total = sum(n for _, n in counts)
print(f"season={year} events={len(events)} stewarding_pdfs={total}")
for name, n in counts[:5]:
    print(f"  {name}: {n}")
if len(counts) > 5:
    print("  ...")
    for name, n in counts[-2:]:
        print(f"  {name}: {n}")
