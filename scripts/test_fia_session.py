#!/usr/bin/env python3
"""Test whether FIA document pages need cookies/referrer chain."""
from __future__ import annotations

import http.cookiejar
import re
import urllib.request

import requests

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
HOME = "https://www.fia.com/"
CHAMP = "https://www.fia.com/documents/championships/fia-formula-one-world-championship-14"
SEASON = (
    "https://www.fia.com/documents/championships/"
    "fia-formula-one-world-championship-14/season/season-2020-1059"
)
EVENT_PATTERN = re.compile(
    r'value="(/documents/championships/fia-formula-one-world-championship-14/season/[^"]+/event/[^"]+)"'
)


def count_events(html: str) -> int:
    return len(EVENT_PATTERN.findall(html))


def try_urllib(label: str, steps: list[tuple[str, str | None]]) -> None:
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    body = b""
    status = 0
    try:
        for url, referer in steps:
            headers = {
                "User-Agent": UA,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
            if referer:
                headers["Referer"] = referer
            req = urllib.request.Request(url, headers=headers)
            with opener.open(req, timeout=30) as response:
                status = response.status
                body = response.read()
        print(f"{label}: {status} bytes={len(body)} events={count_events(body.decode())} cookies={len(list(jar))}")
    except Exception as exc:
        print(f"{label}: FAIL {exc}")


def try_requests(label: str, steps: list[tuple[str, str | None]]) -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
    try:
        for url, referer in steps:
            headers = {"Referer": referer} if referer else {}
            response = session.get(url, headers=headers, timeout=30)
            response.raise_for_status()
        print(
            f"{label}: {response.status_code} bytes={len(response.text)} "
            f"events={count_events(response.text)} cookies={len(session.cookies)}"
        )
    except Exception as exc:
        print(f"{label}: FAIL {exc}")


if __name__ == "__main__":
    cold = [(SEASON, None)]
    warmed = [(HOME, None), (CHAMP, HOME), (SEASON, CHAMP)]
    print("urllib tests:")
    try_urllib("cold_direct", cold)
    try_urllib("home_champ_season", warmed)
    print("\nrequests tests:")
    try_requests("cold_direct", cold)
    try_requests("home_champ_season", warmed)
