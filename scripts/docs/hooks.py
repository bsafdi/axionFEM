"""MkDocs hooks for the axionFEM site (see mkdocs.yml, ``hooks:``).

The Markdown in docs/ is written to be read on GitHub: it links to source
files (``../python/axionfem/srf.py``), benchmarks, notebooks and records
that are not part of the site.  This hook rewrites each relative link, for the
page's *real* location in the repository:

* a page or file that is part of the site  -> a relative link inside the site
  (docs/*, and the example notebooks that gen_pages.py copies in);
* anything else that exists in the repository -> its GitHub URL
  (``blob/main/...`` for files, ``tree/main/...`` for directories).

Links whose target does not exist in the repository are left alone, so
``mkdocs build --strict`` still reports genuinely broken links.  (Notebooks are
rendered by mkdocs-jupyter and are not processed here.)

It also separates consecutive one-line ``$$...$$`` equations by a blank line,
so that MathJax typesets them as separate displays.
"""
from __future__ import annotations

import posixpath
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BRANCH = "main"

# Virtual pages written by scripts/docs/gen_pages.py -> their real file.
REAL_SOURCE: dict[str, str] = {}
_SITE_PAGE = {v: k for k, v in REAL_SOURCE.items()}


def _site_path(resolved: str) -> str | None:
    """Path inside the site of a repository file that the site also carries, else None."""
    if resolved.startswith("docs/"):
        return resolved[len("docs/"):]
    if resolved in _SITE_PAGE:
        return _SITE_PAGE[resolved]
    if resolved.startswith("examples/") and resolved.endswith(".ipynb"):
        return resolved  # copied by gen_pages.py to the same path
    return None

# ](target) and ](target "title"); src="..." / href="..." in raw HTML.
_MD_LINK = re.compile(r"(\]\()(<?)([^)\s>]+)(>?)((?:\s+\"[^\"]*\")?\))")
_HTML_LINK = re.compile(r"(\b(?:src|href)=\")([^\"]+)(\")")
_FENCE = re.compile(r"^(\s*)(```|~~~)")
_DISPLAY_LINE = re.compile(r"^\s*\$\$.*\S.*\$\$\s*$")  # $$ ... $$ on one line


def _repo_url(config) -> str:
    return (config.get("repo_url") or "https://github.com/bsafdi/axionFEM").rstrip("/")


def _rewrite(target: str, page_real: str, page_src: str, files, config) -> str:
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith(("#", "/")):
        return target
    path, sep, anchor = target.partition("#")
    if not path:
        return target
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(page_real), path))
    if resolved.startswith(".."):
        return target
    on_disk = ROOT / resolved
    if not on_disk.exists():
        return target  # leave it: a broken link must stay visible to --strict
    docs_rel = _site_path(resolved) if on_disk.is_file() else None
    if docs_rel is not None:
        f = files.get_file_from_path(docs_rel)
        if f is not None and f.inclusion.is_included():
            new = posixpath.relpath(docs_rel, posixpath.dirname(page_src) or ".")
            return new + sep + anchor
    kind = "tree" if on_disk.is_dir() else "blob"
    return f"{_repo_url(config)}/{kind}/{BRANCH}/{resolved}" + sep + anchor


def on_page_markdown(markdown, page, config, files, **kwargs):
    src = page.file.src_uri
    if src.startswith("reference/"):
        return markdown
    page_real = REAL_SOURCE.get(src, f"docs/{src}")

    def md_sub(m):
        return m.group(1) + m.group(2) + _rewrite(m.group(3), page_real, src, files, config) + m.group(4) + m.group(5)

    def html_sub(m):
        return m.group(1) + _rewrite(m.group(2), page_real, src, files, config) + m.group(3)

    out, in_fence, fence = [], False, ""
    for line in markdown.split("\n"):
        fm = _FENCE.match(line)
        if fm:
            if not in_fence:
                in_fence, fence = True, fm.group(2)
            elif fm.group(2) == fence:
                in_fence = False
            out.append(line)
            continue
        if not in_fence:
            line = _MD_LINK.sub(md_sub, line)
            line = _HTML_LINK.sub(html_sub, line)
            # Consecutive one-line $$...$$ equations would merge into one MathJax
            # display (and clash on \tag): keep each its own block.
            if _DISPLAY_LINE.match(line) and out and _DISPLAY_LINE.match(out[-1]):
                out.append("")
        out.append(line)
    return "\n".join(out)


def on_files(files, config, **kwargs):
    # The literate-nav SUMMARY.md files written by gen_pages.py are nav sources,
    # not pages: keep them out of the built site and the search index.
    from mkdocs.structure.files import InclusionLevel

    for f in files:
        if f.src_uri.endswith("SUMMARY.md"):
            f.inclusion = InclusionLevel.EXCLUDED
    return files


def on_page_content(html, page, config, files, **kwargs):
    # Notebooks are rendered by mkdocs-jupyter straight to HTML, so their links
    # (written for the repository, e.g. ../docs/guides/comsol.md) are rewritten
    # here: to the page's URL when the target is part of the site, else to GitHub.
    src = page.file.src_uri
    if not src.endswith(".ipynb"):
        return html
    base = page.url if config.get("use_directory_urls") else posixpath.dirname(page.url)

    def sub(m):
        target = m.group(2)
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith(("#", "/")):
            return m.group(0)
        path, sep, anchor = target.partition("#")
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(src), path))
        if resolved.startswith("..") or not (ROOT / resolved).exists():
            return m.group(0)
        docs_rel = _site_path(resolved) if (ROOT / resolved).is_file() else None
        f = files.get_file_from_path(docs_rel) if docs_rel else None
        if f is not None and f.inclusion.is_included():
            new = posixpath.relpath(f.url or ".", base or ".")
            if f.url.endswith("/") and not new.endswith("/"):
                new += "/"
            return m.group(1) + new + sep + anchor + m.group(3)
        kind = "tree" if (ROOT / resolved).is_dir() else "blob"
        return m.group(1) + f"{_repo_url(config)}/{kind}/{BRANCH}/{resolved}" + sep + anchor + m.group(3)

    return re.sub(r'(\bhref=")([^"]+)(")', sub, html)
