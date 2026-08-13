import json, sys, urllib.request

BASE = "http://127.0.0.1:8188"

def get_object_info():
    with urllib.request.urlopen(f"{BASE}/object_info", timeout=30) as r:
        return json.load(r)

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", "CLIP_TEXT_ENCODE",
                "COMFY_DYNAMICCOMBO_V2", "COMFY_DYNAMICCOMBO_V3", "SHORT_STRING"}

def is_widget(spec):
    if not isinstance(spec, (list, tuple)) or len(spec) < 1:
        return False
    t = spec[0]
    if isinstance(t, list):  # combo options list
        return True
    return t in WIDGET_TYPES

def convert(wf, obj):
    links = {l[0]: l for l in wf["links"]}  # id -> [id, src, src_slot, dst, dst_slot]
    prompt = {}
    problems = []
    for node in wf["nodes"]:
        if node.get("mode") == 4:
            continue
        nid = str(node["id"])
        ctype = node["type"]
        info = obj.get(ctype)
        if info is None:
            problems.append(f"[{nid}] {ctype}: 节点类型未注册")
            continue
        schema = info.get("input", {})
        # ordered widget-name list + full input schema
        widget_order = []
        full_schema = {}
        for section in ("required", "optional"):
            for name, spec in (schema.get(section) or {}).items():
                full_schema[name] = spec
                if is_widget(spec):
                    widget_order.append(name)
        inputs = {}
        linked = {}
        for inp in node.get("inputs") or []:
            if inp.get("link") is not None:
                l = links.get(inp["link"])
                if l:
                    linked[inp["name"]] = [str(l[1]), l[2]]
        wvals = node.get("widgets_values") or []
        wi = 0
        for name in widget_order:
            # 链接的 widget（被转换成输入的）仍占 widgets_values 一个位置，只是值用链接
            if name in linked:
                inputs[name] = linked[name]
            elif wi < len(wvals):
                inputs[name] = wvals[wi]
            wi += 1
        # 纯链接输入（MODEL/IMAGE 等非 widget 类型，如 model/images/audio）
        for name, val in linked.items():
            if name not in inputs:
                inputs[name] = val
        prompt[nid] = {"class_type": ctype, "inputs": inputs}
        # validation: required inputs must be filled
        for req in (schema.get("required") or {}):
            if req not in inputs:
                problems.append(f"[{nid}] {ctype}: 缺少 required 输入 {req}")
    return prompt, problems

def main():
    wf = json.load(open(sys.argv[1], encoding="utf-8"))
    obj = get_object_info()
    prompt, problems = convert(wf, obj)
    out = sys.argv[2] if len(sys.argv) > 2 else "prompt.json"
    json.dump(prompt, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"节点数: {len(prompt)}")
    print("问题:")
    for p in problems:
        print("  " + p)
    if not problems:
        print("  无")

if __name__ == "__main__":
    main()
