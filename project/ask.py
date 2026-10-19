import sys, json, argparse
import requests

BASE = "http://127.0.0.1:8000"

p = argparse.ArgumentParser(description="Send a prompt to the FastAPI Gemini proxy")
p.add_argument("prompt", nargs="*", help="prompt text (or pipe via stdin)")
p.add_argument("-p", "--previous", help="previous_interaction_id to continue a conversation")
p.add_argument("--usage", action="store_true", help="show token totals instead")
args = p.parse_args()

try:
    if args.usage:
        r = requests.get(f"{BASE}/usage", timeout=10)
    else:
        prompt = " ".join(args.prompt) or sys.stdin.read().strip()
        if not prompt:
            sys.exit("No prompt given.")
        params = {"prompt": prompt}
        if args.previous:
            params["previous_interaction_id"] = args.previous
        r = requests.get(f"{BASE}/ask", params=params, timeout=120)

    r.raise_for_status()
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))
except requests.RequestException as e:
    sys.exit(f"Request failed: {e}")