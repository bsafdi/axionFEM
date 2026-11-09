"""Single-file HTML export and a live viewer server for axionFEM scenes.

A *scene* (schema ``axionfem.scene/1``, see ``docs/scene_format.md``) is the
display description of a solved design. This module turns scenes into
self-contained HTML pages (three.js and the viewer are inlined; the page works
offline from ``file://``) and serves a live viewer that reloads whenever a
watched ``scene.json`` changes.

Standard library only; numpy is used, if installed, to encode numpy arrays
found in scene dictionaries. The module is importable on its own (no imports
of other ``axionfem`` modules), so it also runs as a script::

    python python/axionfem/viewer.py export runs/core/scene.json core.html
    python python/axionfem/viewer.py serve runs --port 8765

The viewer assets are looked up in ``$AXIONFEM_VIEWER_DIR``, then in the
repository (``python/axionfem/../../viewer``), then in the package data
directory ``axionfem/_viewer``.

SPDX-License-Identifier: BSD-3-Clause
"""
from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import html
import json
import math
import os
import re
import socket
import sys
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union
from urllib.parse import unquote, urlsplit

try:  # optional
    import numpy as _np
except Exception:  # pragma: no cover - numpy is optional
    _np = None

__all__ = [
    "export_html",
    "render_html",
    "serve",
    "load_scene",
    "viewer_dir",
    "VIEWER_VERSION",
    "THREE_VERSION",
]

VIEWER_VERSION = "0.1.0"
THREE_VERSION = "0.185.1"
SCENE_SCHEMA = "axionfem.scene/1"

PathLike = Union[str, "os.PathLike[str]"]

# template placeholder -> asset path inside the viewer directory
_ASSETS = {
    "AXF_THREE_CORE": "vendor/three.core.min.js",
    "AXF_THREE_MODULE": "vendor/three.module.min.js",
    "AXF_ORBIT_CONTROLS": "vendor/addons/controls/OrbitControls.js",
    "AXF_ROOM_ENVIRONMENT": "vendor/addons/environments/RoomEnvironment.js",
    "AXF_VIEWER": "src/viewer.js",
}
_PLACEHOLDERS = ("AXF_TITLE", "AXF_CONFIG_JSON", "AXF_SCENE_JSON", *_ASSETS)
_PLACEHOLDER_RE = re.compile(r"\{\{(AXF_[A-Z_]+)\}\}")
# three.module.min.js imports the core by a relative path, which cannot be
# resolved from a data: URL; it is rewritten to a bare specifier that the
# import map resolves.
_CORE_IMPORT = '"./three.core.min.js"'
_CORE_SPECIFIER = '"axf-three-core"'


# --------------------------------------------------------------------------- assets
def viewer_dir() -> Path:
    """Return the directory that holds ``template.html``, ``src/`` and ``vendor/``."""
    here = Path(__file__).resolve().parent
    candidates = []
    env = os.environ.get("AXIONFEM_VIEWER_DIR")
    if env:
        candidates.append(Path(env).expanduser())
    candidates += [here.parent.parent / "viewer", here / "_viewer"]
    for c in candidates:
        if (c / "template.html").is_file() and (c / "src" / "viewer.js").is_file():
            return c
    raise FileNotFoundError(
        "axionFEM viewer assets not found (looked in: "
        + ", ".join(str(c) for c in candidates)
        + "). Set AXIONFEM_VIEWER_DIR to the repository's viewer/ directory."
    )


_asset_cache: dict = {}
_asset_lock = threading.Lock()


def _read_cached(path: Path, transform=None) -> str:
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size, transform is not None)
    with _asset_lock:
        hit = _asset_cache.get(key)
    if hit is not None:
        return hit
    text = path.read_text(encoding="utf-8")
    if transform is not None:
        text = transform(text)
    with _asset_lock:
        _asset_cache[key] = text
    return text


def _rewrite_core_import(text: str) -> str:
    if _CORE_IMPORT not in text:
        raise RuntimeError(
            "vendor/three.module.min.js does not import ./three.core.min.js; "
            "the vendored three.js build is not the expected r185 layout"
        )
    return text.replace(_CORE_IMPORT, _CORE_SPECIFIER)


def _asset_text(vd: Path, placeholder: str, inline: bool) -> str:
    path = vd / _ASSETS[placeholder]
    if not path.is_file():
        raise FileNotFoundError(f"viewer asset missing: {path}")
    transform = _rewrite_core_import if (inline and placeholder == "AXF_THREE_MODULE") else None
    return _read_cached(path, transform)


def _data_url(js: str) -> str:
    return "data:text/javascript;base64," + base64.b64encode(js.encode("utf-8")).decode("ascii")


def _template(vd: Path) -> str:
    text = _read_cached(vd / "template.html")
    names = _PLACEHOLDER_RE.findall(text)
    missing = [p for p in _PLACEHOLDERS if p not in names]
    unknown = sorted(set(names) - set(_PLACEHOLDERS))
    if missing or unknown:
        raise RuntimeError(f"{vd / 'template.html'}: missing placeholders {missing}, unknown {unknown}")
    return text


def _fill(template: str, subs: Mapping[str, str]) -> str:
    # single pass: replacement text (scene JSON, code) is never re-scanned
    return _PLACEHOLDER_RE.sub(lambda m: subs[m.group(1)], template)


# --------------------------------------------------------------------------- scenes
def _encode_array(a) -> dict:
    a = _np.asarray(a)
    if a.dtype.kind in "biu" and (a.size == 0 or a.min() >= 0):
        dtype, arr = "uint32", _np.ascontiguousarray(a, dtype="<u4")
    else:
        dtype, arr = "float32", _np.ascontiguousarray(a, dtype="<f4")
    shape = [int(arr.shape[0]), int(arr.shape[1]) if arr.ndim > 1 else 1] if arr.ndim else [1, 1]
    if arr.ndim > 2:
        raise ValueError(f"scene arrays must be 1D or 2D, got shape {a.shape}")
    return {"dtype": dtype, "shape": shape, "data": base64.b64encode(arr.tobytes()).decode("ascii")}


def _jsonable(o: Any, where: str = "scene") -> Any:
    if o is None or isinstance(o, (str, bool, int)):
        return o
    if isinstance(o, float):
        return o if math.isfinite(o) else None  # NaN/inf are not JSON
    if isinstance(o, Mapping):
        return {str(k): _jsonable(v, f"{where}.{k}") for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v, f"{where}[{i}]") for i, v in enumerate(o)]
    if _np is not None:
        if isinstance(o, _np.ndarray):
            return _encode_array(o)
        if isinstance(o, _np.generic):
            return _jsonable(o.item(), where)
    if isinstance(o, os.PathLike):
        return os.fspath(o)
    raise TypeError(f"{where}: cannot write a value of type {type(o).__name__} to the scene JSON")


def load_scene(scene: Any) -> dict:
    """Return a scene dictionary from a dict, a path to ``scene.json`` (or its
    directory), a JSON string, or an object with a ``scene()`` method."""
    src = "scene"
    if isinstance(scene, (str, os.PathLike)) and not (isinstance(scene, str) and scene.lstrip().startswith("{")):
        p = Path(scene).expanduser()
        if p.is_dir():
            p = p / "scene.json"
        src = str(p)
        try:
            text = p.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise FileNotFoundError(f"scene file not found: {p}") from None
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"{p} is not valid JSON ({e}); is it still being written?") from None
    elif isinstance(scene, str):
        data = json.loads(scene)
    elif isinstance(scene, Mapping):
        data = scene
    elif hasattr(scene, "scene") and callable(scene.scene):
        data = scene.scene()
        if isinstance(data, str):
            data = json.loads(data)
    else:
        raise TypeError(
            f"expected a scene dict, a path to scene.json or an object with .scene(); got {type(scene).__name__}"
        )
    if not isinstance(data, Mapping):
        raise ValueError(f"{src}: a scene must be a JSON object, got {type(data).__name__}")
    data = _jsonable(data)
    schema = data.get("schema")
    if schema is not None and not str(schema).startswith("axionfem.scene/"):
        raise ValueError(f"{src}: not an axionFEM scene (schema {schema!r}; expected {SCENE_SCHEMA!r})")
    if not isinstance(data.get("regions"), list):
        raise ValueError(
            f"{src}: the scene has no 'regions' list; expected the output of Model.scene() or `axf solve --scene`"
        )
    return data


def _script_json(obj: Any) -> str:
    """JSON safe to place inside <script type="application/json">."""
    return _escape_script_json(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False))


# --------------------------------------------------------------------------- export
def render_html(scene: Any, title: Optional[str] = None, *, options: Optional[Mapping[str, Any]] = None) -> str:
    """Return a self-contained HTML page (str) displaying ``scene``.

    ``options`` are viewer options (see docs/viewer.md), e.g.
    ``{"mode": "section", "quantity": "Bz", "scale": "log"}``; URL query
    parameters override them when the page is opened.
    """
    data = load_scene(scene)
    vd = viewer_dir()
    name = str(data.get("name") or "design")
    subs = {ph: _data_url(_asset_text(vd, ph, inline=True)) for ph in _ASSETS}
    subs["AXF_TITLE"] = html.escape(title or f"{name} · axionFEM", quote=True)
    subs["AXF_CONFIG_JSON"] = _script_json(
        {"options": dict(options or {}), "viewer": VIEWER_VERSION, "three": THREE_VERSION}
    )
    subs["AXF_SCENE_JSON"] = _script_json(data)
    return _fill(_template(vd), subs)


def _umask() -> int:
    """The process umask, read without changing it (os.umask would briefly alter it for every thread)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("Umask:"):
                    return int(line.split()[1], 8)
    except (OSError, ValueError):
        pass
    return 0o022


def export_html(
    scene: Any,
    out_path: PathLike,
    title: Optional[str] = None,
    *,
    options: Optional[Mapping[str, Any]] = None,
) -> str:
    """Write a self-contained HTML viewer for ``scene`` and return its path.

    ``scene`` is a scene dict or a path to ``scene.json``. If ``out_path`` is an
    existing directory, the file is named after the design. The page needs no
    network access and works from ``file://``.
    """
    data = load_scene(scene)
    page = render_html(data, title, options=options)
    out = Path(out_path).expanduser()
    if out.is_dir():
        out = out / (re.sub(r"[^\w.-]+", "_", str(data.get("name") or "scene")) + ".html")
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".axf-", suffix=".html", dir=str(out.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(page)
        os.chmod(tmp, 0o666 & ~_umask())   # mkstemp creates the file private (0600): use the normal file mode
        os.replace(tmp, out)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return str(out)


# --------------------------------------------------------------------------- live server
_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "third_party", ".ipynb_checkpoints"}


def _walk_scenes(top: Path, max_depth: int = 4, limit: int = 500) -> list:
    found = []
    top_depth = len(top.parts)
    for dirpath, dirnames, filenames in os.walk(top):
        depth = len(Path(dirpath).parts) - top_depth
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".") and depth < max_depth)
        for fn in sorted(filenames):
            if fn == "scene.json" or fn.endswith(".scene.json"):
                found.append(Path(dirpath) / fn)
                if len(found) >= limit:
                    return found
    return found


class _SceneSet:
    """The scene files being watched (re-discovered on every request)."""

    def __init__(self, specs: Iterable[PathLike]):
        self.specs = [Path(s).expanduser().resolve() for s in specs]
        self._parsed: dict = {}
        self._lock = threading.Lock()

    def scenes(self) -> list:
        out, seen = [], set()
        for s in self.specs:
            if s.is_dir():
                paths = _walk_scenes(s)
            elif s.is_file() or s.suffix == ".json":  # a file that may appear later
                paths = [s]
            else:
                paths = []
            for p in paths:
                if p not in seen:
                    seen.add(p)
                    out.append((hashlib.sha1(str(p).encode()).hexdigest()[:10], p))
        return out

    def find(self, sid: str) -> Optional[Path]:
        for i, p in self.scenes():
            if i == sid:
                return p
        return None

    @staticmethod
    def version(p: Path) -> Optional[str]:
        try:
            st = p.stat()
        except OSError:
            return None
        return f"{st.st_mtime_ns}-{st.st_size}"

    def load(self, p: Path):
        """Return (bytes, name, version) of a complete scene file, or raise ValueError."""
        v = self.version(p)
        if v is None:
            raise FileNotFoundError(str(p))
        with self._lock:
            hit = self._parsed.get(p)
            if hit and hit[0] == v:
                return hit[1], hit[2], v
        raw = p.read_bytes()
        if self.version(p) != v:
            raise ValueError("scene file changed while reading")
        nonfinite = []
        try:
            data = json.loads(raw, parse_constant=lambda c: nonfinite.append(c))
        except (ValueError, UnicodeDecodeError) as e:
            raise ValueError(f"incomplete or invalid JSON: {e}") from None
        if not isinstance(data, dict) or not isinstance(data.get("regions"), list):
            raise ValueError("not an axionFEM scene (no 'regions' list)")
        if nonfinite:  # NaN/Infinity are not JSON and browsers reject them; they become null
            raw = json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        name = str(data.get("name") or p.parent.name)
        with self._lock:
            self._parsed[p] = (v, raw, name)
        return raw, name, v


_ALLOWED_ASSETS = {path: ph for ph, path in _ASSETS.items()}

_INDEX_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>axionFEM live viewer</title>
<style>
:root{color-scheme:light dark;--bg:#f6f8fa;--fg:#1f2328;--fg2:#57606a;--line:rgba(31,35,40,.12);--accent:#0969da}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--fg2:#9da7b3;--line:rgba(240,246,252,.12);--accent:#58a6ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:920px;margin:0 auto;padding:28px 16px}
h1{font-size:20px;margin:0 0 4px}p{color:var(--fg2);margin:0 0 18px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:8px 6px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;color:var(--fg2);font-weight:600}td.p{color:var(--fg2);font-size:12px;word-break:break-all}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
</style></head><body><main>
<h1>axionFEM live viewer</h1>
<p>Scenes under watch. Pages reload automatically when a scene file changes.</p>
<table><thead><tr><th>Design</th><th>File</th><th>Modified</th><th></th></tr></thead><tbody id="rows"></tbody></table>
</main>
<script>
async function refresh(){
  try{
    const r=await fetch('api/scenes',{cache:'no-store'}); const list=await r.json();
    const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    document.getElementById('rows').innerHTML=list.length?list.map(s=>`<tr><td><a href="view/${s.id}">${esc(s.name)}</a></td><td class="p">${esc(s.path)}</td><td>${s.mtime?new Date(s.mtime*1000).toLocaleString():'missing'}</td><td><a href="export/${s.id}.html" download>HTML</a></td></tr>`).join(''):'<tr><td colspan="4">No scene.json files found yet.</td></tr>';
  }catch(e){}
}
refresh(); setInterval(refresh, 2000);
</script></body></html>
"""


class _Handler(BaseHTTPRequestHandler):
    server_version = "axionFEM-viewer/" + VIEWER_VERSION
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet unless asked
        if not getattr(self.server, "quiet", True):
            sys.stderr.write("[axionFEM viewer] %s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, ctype: str, headers: Optional[dict] = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json; charset=utf-8")

    def _text(self, code: int, msg: str, headers: Optional[dict] = None) -> None:
        self._send(code, msg.encode(), "text/plain; charset=utf-8", headers)

    def do_HEAD(self):  # noqa: N802
        self.do_GET()

    def do_GET(self):  # noqa: N802
        try:
            self._route(unquote(urlsplit(self.path).path))
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # pragma: no cover - report, keep serving
            try:
                self._text(500, f"internal error: {e}")
            except Exception:
                pass

    def _route(self, path: str) -> None:
        srv = self.server
        scenes: _SceneSet = srv.scenes
        if path in ("/", "/index.html"):
            lst = scenes.scenes()
            if len(lst) == 1 and len(scenes.specs) == 1 and not scenes.specs[0].is_dir():
                self._send(302, b"", "text/plain", {"Location": f"view/{lst[0][0]}"})
                return
            self._send(200, _INDEX_HTML.encode(), "text/html; charset=utf-8")
        elif path == "/api/scenes":
            out = []
            for sid, p in scenes.scenes():
                try:
                    st = p.stat()
                    mtime = st.st_mtime
                except OSError:
                    mtime = None
                name = p.parent.name if p.name == "scene.json" else p.name[: -len(".scene.json")] if p.name.endswith(".scene.json") else p.stem
                out.append({"id": sid, "name": name, "path": str(p), "mtime": mtime, "version": scenes.version(p)})
            self._json(out)
        elif path.startswith("/api/version/"):
            p = scenes.find(path.rsplit("/", 1)[-1])
            if p is None:
                self._json({"exists": False, "version": None}, 404)
            else:
                v = scenes.version(p)
                self._json({"exists": v is not None, "version": v, "path": str(p)})
        elif path.startswith("/scene/") and path.endswith(".json"):
            p = scenes.find(path[len("/scene/"): -len(".json")])
            if p is None:
                self._text(404, "unknown scene id")
                return
            try:
                raw, _, _ = scenes.load(p)
            except FileNotFoundError:
                self._text(404, f"scene file not found: {p}")
                return
            except ValueError as e:
                self._text(503, f"scene not ready: {e}", {"Retry-After": "1"})
                return
            self._send(200, raw, "application/json; charset=utf-8")
        elif path.startswith("/view/"):
            sid = path[len("/view/"):].strip("/")
            p = scenes.find(sid)
            if p is None:
                self._text(404, "unknown scene id; see the index page")
                return
            try:
                raw, name, version = scenes.load(p)
            except (OSError, ValueError):
                raw, name, version = None, p.parent.name, None
            page = _live_page(sid, name, srv.poll_ms, srv.options, raw, version)
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
        elif path.startswith("/export/") and path.endswith(".html"):
            p = scenes.find(path[len("/export/"): -len(".html")])
            if p is None:
                self._text(404, "unknown scene id")
                return
            try:
                page = render_html(p, options=srv.options)
            except (OSError, ValueError) as e:
                self._text(503, f"cannot export: {e}", {"Retry-After": "1"})
                return
            fname = re.sub(r"[^\w.-]+", "_", p.parent.name or "scene") + ".html"
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8", {"Content-Disposition": f'attachment; filename="{fname}"'})
        elif path.startswith("/assets/"):
            rel = path[len("/assets/"):]
            if rel not in _ALLOWED_ASSETS:
                self._text(404, "not found")
                return
            body = _asset_text(viewer_dir(), _ALLOWED_ASSETS[rel], inline=False).encode("utf-8")
            self._send(200, body, "text/javascript; charset=utf-8")
        elif path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        else:
            self._text(404, "not found")


def _escape_script_json(text: str) -> str:
    # '<', '>', '&' and U+2028/9 can only occur inside JSON strings, where the
    # \uXXXX escapes are equivalent
    return (
        text.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _live_page(
    sid: str, name: str, poll_ms: int, options: Mapping[str, Any], raw: Optional[bytes] = None, version: Optional[str] = None
) -> str:
    vd = viewer_dir()
    subs = {ph: f"../assets/{_ASSETS[ph]}" for ph in _ASSETS}
    subs["AXF_TITLE"] = html.escape(f"{name} · axionFEM live", quote=True)
    subs["AXF_CONFIG_JSON"] = _script_json(
        {
            "options": dict(options or {}),
            "sceneUrl": f"../scene/{sid}.json",
            "versionUrl": f"../api/version/{sid}",
            "version": version if raw is not None else None,
            "poll": int(poll_ms),
            "viewer": VIEWER_VERSION,
            "three": THREE_VERSION,
        }
    )
    # the current scene is embedded so the first frame renders without a round trip
    subs["AXF_SCENE_JSON"] = _escape_script_json(raw.decode("utf-8")) if raw is not None else "null"
    return _fill(_template(vd), subs)


class ViewerServer(ThreadingHTTPServer):
    """HTTP server returned by :func:`serve` (``.url``, ``.stop()``)."""

    daemon_threads = True
    allow_reuse_address = True
    url: str = ""

    def stop(self) -> None:
        """Stop serving and close the socket."""
        self.shutdown()
        self.server_close()


def serve(
    paths_or_dir: Union[PathLike, Iterable[PathLike]],
    port: int = 8765,
    host: str = "127.0.0.1",
    open_browser: bool = False,
    *,
    poll: float = 1.0,
    block: bool = True,
    quiet: bool = True,
    options: Optional[Mapping[str, Any]] = None,
) -> ViewerServer:
    """Serve live viewers for scene files and reload them when they change.

    ``paths_or_dir`` is a ``scene.json`` path, a directory (searched for
    ``scene.json`` / ``*.scene.json`` up to four levels deep, re-scanned on every
    request, so new runs appear), or a list of those. The browser polls every
    ``poll`` seconds and swaps in the new scene when the file's mtime/size
    changes; incomplete files (still being written) are skipped until valid.

    By default the server listens on 127.0.0.1 only. From a laptop, tunnel with
    ``ssh -L 8765:localhost:8765 <cluster-login-host>`` and open
    http://localhost:8765/. With ``block=False`` the server runs in a daemon
    thread and the :class:`ViewerServer` is returned (call ``.stop()``);
    otherwise this blocks until Ctrl-C.
    """
    specs = [paths_or_dir] if isinstance(paths_or_dir, (str, os.PathLike)) else list(paths_or_dir)
    if not specs:
        raise ValueError("serve: give at least one scene.json path or directory")
    viewer_dir()  # fail early if the assets are missing
    try:
        httpd = ViewerServer((host, int(port)), _Handler)
    except OSError as e:
        raise OSError(
            f"cannot listen on {host}:{port} ({e.strerror or e}); choose another port, e.g. serve(..., port={int(port) + 1})"
        ) from e
    httpd.scenes = _SceneSet(specs)
    httpd.poll_ms = max(250, int(poll * 1000))
    httpd.quiet = quiet
    httpd.options = dict(options or {})
    real_port = httpd.server_address[1]
    shown = "localhost" if host in ("", "0.0.0.0", "::", "127.0.0.1") else host
    httpd.url = f"http://{shown}:{real_port}/"
    n = len(httpd.scenes.scenes())
    try:
        user, node = getpass.getuser(), socket.gethostname()
    except Exception:  # pragma: no cover
        user, node = "you", "this-host"
    if block or not quiet:
        sys.stderr.write(
            f"axionFEM viewer: watching {n} scene(s) at {httpd.url}\n"
            f"  from your laptop: ssh -L {real_port}:localhost:{real_port} {user}@{node}  then open http://localhost:{real_port}/\n"
        )
    if open_browser:
        try:
            webbrowser.open(httpd.url)
        except Exception:
            pass
    if not block:
        threading.Thread(target=httpd.serve_forever, name="axionfem-viewer", daemon=True).start()
        return httpd
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("axionFEM viewer: stopped\n")
    finally:
        httpd.server_close()
    return httpd


# --------------------------------------------------------------------------- CLI
def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="axionfem.viewer", description="axionFEM viewer: HTML export and live server")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="write a self-contained HTML viewer for a scene")
    ex.add_argument("scene", help="scene.json (or a run directory containing it)")
    ex.add_argument("out", help="output .html path (or a directory)")
    ex.add_argument("--title")
    ex.add_argument("--mode", choices=["3d", "section"])
    ex.add_argument("--quantity")
    ex.add_argument("--scale", choices=["linear", "log"])
    sv = sub.add_parser("serve", help="serve live viewers that reload when scene files change")
    sv.add_argument("paths", nargs="+", help="scene.json files and/or directories to watch")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--open", action="store_true", help="open a browser")
    sv.add_argument("--poll", type=float, default=1.0, help="browser poll interval in seconds")
    sv.add_argument("--verbose", action="store_true", help="log requests")
    a = ap.parse_args(argv)
    if a.cmd == "export":
        opts = {k: v for k, v in (("mode", a.mode), ("quantity", a.quantity), ("scale", a.scale)) if v}
        out = export_html(a.scene, a.out, a.title, options=opts)
        print(f"{out} ({os.path.getsize(out) / 1e6:.2f} MB)")
        return 0
    serve(a.paths, port=a.port, host=a.host, open_browser=a.open, poll=a.poll, quiet=not a.verbose)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
