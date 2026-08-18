from __future__ import annotations

import json
from pathlib import Path

from src.config import get_settings
from src.knowledge import load_chunks
from src.retrieve import TfidfIndex


def main() -> None:
    settings = get_settings()
    chunks = load_chunks(settings.knowledge_path)
    index = TfidfIndex(chunks)
    cases = json.loads(Path(settings.eval_path).read_text(encoding="utf-8"))["cases"]

    hits_ok = 0
    print(f"Evaluating {len(cases)} questions against {len(chunks)} chunks\n")
    for case in cases:
        query = case["query"]
        allowed = set(case["must_include_any"])
        ranked = index.search(query, k=4)
        got = {c.chunk_id for c, _ in ranked}
        ok = bool(got & allowed)
        hits_ok += int(ok)
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {query}")
        print(f"      expected any of {sorted(allowed)}")
        print(f"      got {sorted(got)}\n")

    recall = hits_ok / max(len(cases), 1)
    print(f"Hit@4: {hits_ok}/{len(cases)} = {recall:.0%}")
    if recall < 0.8:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
