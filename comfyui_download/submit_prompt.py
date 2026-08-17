import json, sys, time, urllib.request

BASE = "http://127.0.0.1:8188"


def post(path, data=None):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(BASE + path, data=body,
                                 headers={"Content-Type": "application/json"} if body else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def main():
    prompt_file = sys.argv[1]
    prompt = json.load(open(prompt_file, encoding="utf-8"))
    if len(sys.argv) > 2:  # filename_prefix override
        for node in prompt["prompt"].values():
            if node.get("class_type") == "SaveVideo":
                node["inputs"]["filename_prefix"] = sys.argv[2]
    if len(sys.argv) > 3:
        prompt["client_id"] = sys.argv[3]

    r = post("/prompt", prompt)
    pid = r["prompt_id"]
    print("prompt_id:", pid, flush=True)
    for _ in range(1500):  # up to ~25 min at 5s poll
        h = post(f"/history/{pid}")
        if pid in h:
            print("DONE", flush=True)
            print(json.dumps(h[pid], ensure_ascii=False)[:3000], flush=True)
            return
        time.sleep(5)
    print("TIMEOUT", flush=True)


if __name__ == "__main__":
    main()
