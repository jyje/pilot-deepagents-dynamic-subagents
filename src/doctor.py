"""Diagnostic script: checks ENV and model reachability for the pilot setup."""

import os
import sys
import urllib.request
import urllib.error
from dotenv import load_dotenv

load_dotenv()

PASS = "  [PASS]"
FAIL = "  [FAIL]"
SKIP = "  [SKIP]"
SEP = "-" * 52


def section(title: str) -> None:
    print(f"\n{SEP}\n {title}\n{SEP}")


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = PASS if ok else FAIL
    line = f"{mark} {label}"
    if detail:
        line += f"\n         {detail}"
    print(line)
    return ok


section("1. Environment Variables")

api_key = os.getenv("ANTHROPIC_API_KEY", "")
base_url = os.getenv("ANTHROPIC_BASE_URL", "")
main_model = os.getenv("MAIN_MODEL", "claude-sonnet-4-6")
researcher_model = os.getenv("RESEARCHER_MODEL", "claude-haiku-4-5-20251001")
reviewer_model = os.getenv("REVIEWER_MODEL", "claude-haiku-4-5-20251001")

env_ok = True
env_ok &= check(
    "ANTHROPIC_API_KEY is set",
    bool(api_key) and not api_key.startswith("sk-ant-xxx"),
    "set" if api_key else "not set",
)
check(
    "ANTHROPIC_BASE_URL",
    True,
    f"value: {base_url}" if base_url else "not set — using official Anthropic API",
)
env_ok &= check("MAIN_MODEL", True, f"value: {main_model}")
env_ok &= check("RESEARCHER_MODEL", True, f"value: {researcher_model}")
env_ok &= check("REVIEWER_MODEL", True, f"value: {reviewer_model}")


section("2. API Connectivity")


def http_get(url: str, key: str) -> tuple[int, str]:
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return resp.status, resp.read(256).decode()
    except urllib.error.HTTPError as e:
        return e.code, str(e.reason)
    except Exception as e:
        return 0, str(e)


if not api_key or api_key.startswith("sk-ant-xxx"):
    print(f"{SKIP} API connectivity — ANTHROPIC_API_KEY not configured")
    conn_ok = False
else:
    models_url = f"{base_url.rstrip('/')}/v1/models" if base_url else "https://api.anthropic.com/v1/models"
    status, body = http_get(models_url, api_key)
    conn_ok = check(
        f"GET {models_url}",
        status == 200,
        f"HTTP {status}" + (f": {body[:80]}" if status != 200 else ""),
    )


section("Summary")
all_ok = env_ok and conn_ok
check("ENV", env_ok)
check("Connectivity", conn_ok)

print()
if all_ok:
    print("All checks passed. Ready to run main.py.")
else:
    print("Some checks failed. See details above.")

sys.exit(0 if all_ok else 1)
