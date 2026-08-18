from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    title: str
    text: str

    @property
    def document(self) -> str:
        return f"{self.title}\n{self.text}"


_HEADING = re.compile(r"^##\s+(.+)$", re.MULTILINE)


def load_chunks(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    matches = list(_HEADING.finditer(raw))
    chunks: list[Chunk] = []
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[start:end].strip()
        if not body:
            continue
        chunks.append(Chunk(chunk_id=title, title=title.replace("-", " "), text=body))
    return chunks
