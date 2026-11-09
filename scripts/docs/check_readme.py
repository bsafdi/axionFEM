"""Check the links of README.md and the figure caption it copies.

    python scripts/docs/check_readme.py

README.md is rendered by GitHub (and as the PyPI description), not by MkDocs, so `mkdocs build --strict` does not see
it. This script checks what the site build would:

* every relative link and image (Markdown and HTML ``href``/``src``/``srcset``) names a file or folder of the
  repository, and a ``#fragment`` names a heading of the target page (GitHub's anchor rules);
* every link into the documentation site (``https://bsafdi.github.io/axionFEM/...``) names a page of ``docs/``;
* the hero caption in the README equals the ``hero`` section of ``docs/images/home/captions.md``, which
  docs/images/make_homepage_figures.py generates, so the README quotes no number by hand.

Exits 1 and lists the problems if there are any. External links other than the site are not fetched.
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = "https://bsafdi.github.io/axionFEM/"

_MD_LINK = re.compile(r"!?\[[^\]]*\]\(<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\)")
_HTML_ATTR = re.compile(r"\b(?:href|src|srcset)=\"([^\"]+)\"")
_FENCE = re.compile(r"^```.*?^```", re.S | re.M)


def _github_anchor(heading: str) -> str:
    """GitHub's heading anchor: lower case, punctuation removed, spaces to hyphens."""
    text = re.sub(r"<[^>]+>", "", heading)
    text = re.sub(r"[`*_]", "", text).strip().lower()
    text = "".join(c for c in text if c.isalnum() or c in " -" or unicodedata.category(c).startswith("L"))
    return text.replace(" ", "-")


def _anchors(md: Path) -> set[str]:
    text = _FENCE.sub("", md.read_text("utf-8"))
    return {_github_anchor(m.group(1)) for m in re.finditer(r"^#{1,6}\s+(.+?)\s*#*\s*$", text, re.M)}


def _targets(text: str) -> list[str]:
    text = _FENCE.sub("", text)
    out = [m.group(1) for m in _MD_LINK.finditer(text)]
    out += [m.group(1) for m in _HTML_ATTR.finditer(text)]
    return out


def check_links(readme: Path) -> list[str]:
    problems = []
    for target in _targets(readme.read_text("utf-8")):
        if target.startswith(SITE):
            rel = target[len(SITE):].split("#")[0].strip("/")
            candidates = [ROOT / "docs" / "index.md"] if not rel else [
                ROOT / "docs" / f"{rel}.md", ROOT / "docs" / rel / "index.md", ROOT / "docs" / rel]
            if not any(c.exists() for c in candidates):
                problems.append(f"site link to a page that docs/ does not have: {target}")
            continue
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
            continue                                   # other external links are not fetched
        path, _, frag = target.partition("#")
        dest = readme if not path else (readme.parent / path)
        if not dest.exists():
            problems.append(f"missing target: {target}")
            continue
        if frag and dest.suffix == ".md" and frag not in _anchors(dest):
            problems.append(f"no heading for #{frag} in {dest.relative_to(ROOT)}")
    return problems


def check_caption(readme: Path) -> list[str]:
    captions = (ROOT / "docs" / "images" / "home" / "captions.md").read_text("utf-8")
    m = re.search(r"\[start:hero\] -->\n(.*?)\n<!-- --8<-- \[end:hero\]", captions, re.S)
    if not m:
        return ["captions.md has no hero section"]
    if m.group(1).strip() not in readme.read_text("utf-8"):
        return ["the README's hero caption differs from docs/images/home/captions.md (hero): copy it again"]
    return []


def main() -> int:
    readme = ROOT / "README.md"
    problems = check_links(readme) + check_caption(readme)
    for p in problems:
        print("README.md:", p)
    print("README check:", "ok" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
