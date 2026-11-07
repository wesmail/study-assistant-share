"""The SearXNG container behind web_search: its setup, and a live search
when it's running (scripts/searxng.sh start)."""
import re
from pathlib import Path

import httpx
import pytest
import yaml

from app.config import Settings

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (ROOT / "scripts" / "searxng.sh").read_text(encoding="utf-8")
SETTINGS = yaml.safe_load((ROOT / "searxng" / "settings.yml").read_text(encoding="utf-8"))


def test_it_answers_in_json_without_the_limiter_or_metrics():
    assert "json" in SETTINGS["search"]["formats"]
    assert SETTINGS["server"]["limiter"] is False and SETTINGS["server"]["public_instance"] is False
    assert SETTINGS["general"]["enable_metrics"] is False


def test_no_secret_is_kept_in_the_repository():
    assert "secret_key" not in SETTINGS.get("server", {})
    assert "SEARXNG_SECRET=" in SCRIPT and "/dev/urandom" in SCRIPT


def test_it_is_published_on_the_loopback_address_only():
    published = re.findall(r'-p "([^"]+)"', SCRIPT)
    assert published and all(port.startswith("127.0.0.1:") for port in published)


def test_the_image_is_pinned():
    image = re.search(r'IMAGE="([^"]+)"', SCRIPT).group(1)
    assert image.startswith("searxng/searxng:") and not image.endswith(":latest")
    assert re.fullmatch(r"searxng/searxng:\d{4}\.\d+\.\d+-[0-9a-f]+", image)


def test_start_sh_starts_it():
    assert "scripts/searxng.sh start" in (ROOT / "scripts" / "start.sh").read_text(encoding="utf-8")


def running(url: str) -> bool:
    try:
        return httpx.get(f"{url}/healthz", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.mark.skipif(not running(Settings().searxng_url), reason="SearXNG isn't running: scripts/searxng.sh start")
def test_a_live_search_returns_results_with_titles_and_links():
    response = httpx.get(f"{Settings().searxng_url}/search", params={"q": "fine-structure constant CODATA", "format": "json"},
                         headers={"X-Real-IP": "127.0.0.1"}, timeout=20)
    response.raise_for_status()
    results = response.json()["results"]
    assert len(results) >= 3
    assert all(result["url"].startswith(("http://", "https://")) and result["title"] for result in results[:3])
