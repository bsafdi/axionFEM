"""Griffe extension: render the Sphinx roles used in axionFEM docstrings.

The docstrings are NumPy style with reStructuredText cross-reference roles
(``:func:`signal_power```, ``:class:`axionfem.circuit.Readout```, ``:mod:`...```).
mkdocstrings renders Markdown, where those would show up verbatim.  This
extension rewrites each role to an *optional* autorefs link: a link to the
object's API entry when it is documented, plain inline code otherwise (no
warning either way).  It also turns the reST literal-block marker ``Text::``
into ``Text:`` (the indented block that follows is already a Markdown code
block) and reST "simple tables" (``====  ====`` borders) into Markdown tables,
and splits grouped NumPy parameter entries (``a, b, c`` on one line) into one
entry per name, because griffe would otherwise give every name of the group
the annotation and default of the first one.
Docstrings in the source are not changed.
"""
from __future__ import annotations

import re

import griffe

_ROLE = re.compile(r":(?:py:)?(func|class|meth|mod|attr|data|obj|exc|const):`([^`]+)`")
_LITERAL = re.compile(r"(\S)::[ \t]*$", re.M)
_BORDER = re.compile(r"^(\s*)=+(?:\s+=+)+\s*$")


def _simple_tables(text: str) -> str:
    """reST simple table (top border, header, border, rows, bottom border) -> Markdown."""
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines):
        m = _BORDER.match(lines[i])
        if not m or i + 2 >= len(lines) or not _BORDER.match(lines[i + 2]):
            out.append(lines[i])
            i += 1
            continue
        ncol = len(lines[i].split())
        j = i + 3
        while j < len(lines) and not _BORDER.match(lines[j]):
            j += 1
        if j >= len(lines):
            out.append(lines[i])
            i += 1
            continue

        def cells(row: str) -> list[str]:
            c = re.split(r"\s{2,}", row.strip(), maxsplit=ncol - 1)
            return [x.replace("|", "\\|") for x in c] + [""] * (ncol - len(c))

        ind = m.group(1)
        if out and out[-1].strip():
            out.append("")
        out.append(ind + "| " + " | ".join(cells(lines[i + 1])) + " |")
        out.append(ind + "|" + "---|" * ncol)
        for row in lines[i + 3:j]:
            if row.strip():
                out.append(ind + "| " + " | ".join(cells(row)) + " |")
        out.append("")
        i = j + 1
    return "\n".join(out)


_SECTION = re.compile(r"^(\S.*)\n(-{3,})\s*$", re.M)
_PARAM_SECTIONS = {"parameters", "other parameters", "keyword arguments", "other keyword arguments"}


def _split_grouped_params(text: str) -> str:
    lines = text.split("\n")
    out, in_params, i = [], False, 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if line and not line[0].isspace() and re.fullmatch(r"-{3,}\s*", nxt or "x"):
            in_params = line.strip().lower() in _PARAM_SECTIONS
            out += [line, nxt]
            i += 2
            continue
        m = re.fullmatch(r"([A-Za-z_*][\w*]*(?:\s*,\s*[A-Za-z_*][\w*]*)+)(\s*:.*)?", line) if in_params else None
        if m:
            names = [n.strip() for n in m.group(1).split(",")]
            typ = m.group(2) or ""
            j = i + 1
            while j < len(lines) and (not lines[j] or lines[j][0].isspace()):
                j += 1
            body = lines[i + 1:j]
            while body and not body[-1].strip():
                body.pop()
            indent = next((re.match(r"\s*", b).group(0) for b in body if b.strip()), "    ")
            out.append(names[0] + typ)
            out += body
            for n in names[1:]:
                out.append(n + typ)
                out.append(f"{indent}See `{names[0]}`.")
            if j < len(lines) and lines[j - 1:j] and not lines[j - 1].strip():
                out.append("")
            i = j
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _exists(pkg: griffe.Module, ident: str) -> bool:
    root = pkg.path + "."
    if ident == pkg.path:
        return True
    if not ident.startswith(root):
        return False
    try:
        pkg[ident[len(root):]]
    except (KeyError, griffe.AliasResolutionError, griffe.CyclicAliasError):
        return False
    return True


def _resolve(pkg: griffe.Module, obj: griffe.Object, target: str) -> str:
    target = target.lstrip(".")
    module = obj.module.path if not obj.is_module else obj.path
    cands = [target, f"{module}.{target}"]
    parent = obj if obj.is_class else obj.parent
    while parent is not None and not parent.is_module:
        cands.insert(1, f"{parent.path}.{target}")
        parent = parent.parent
    cands.append(f"{pkg.path}.{target}")
    for c in cands:
        if _exists(pkg, c):
            return c
    return cands[0] if target.startswith(pkg.path + ".") else cands[1]


class SphinxRoles(griffe.Extension):
    def on_package(self, *, pkg: griffe.Module, **kwargs) -> None:
        seen: set[int] = set()

        def visit(obj) -> None:
            if obj.is_alias or id(obj) in seen:
                return
            seen.add(id(obj))
            if obj.docstring is not None:
                obj.docstring.value = self._convert(pkg, obj, obj.docstring.value)
            for member in obj.members.values():
                visit(member)

        visit(pkg)

    @staticmethod
    def _convert(pkg, obj, text: str) -> str:
        def sub(m: re.Match) -> str:
            raw = m.group(2).strip()
            title = None
            tm = re.match(r"^(.*?)\s*<(.+)>$", raw)  # :func:`title <target>`
            if tm:
                title, raw = tm.group(1), tm.group(2)
            short = raw.startswith("~")
            target = raw.lstrip("~")
            shown = title or (target.rsplit(".", 1)[-1] if short else target)
            if m.group(1) in ("func", "meth") and not shown.endswith(")") and not title:
                shown += "()"
            ident = _resolve(pkg, obj, target)
            return f'<autoref identifier="{ident}" optional><code>{shown}</code></autoref>'

        text = _split_grouped_params(text)
        text = _simple_tables(text)
        text = _ROLE.sub(sub, text)
        return _LITERAL.sub(r"\1:", text)
