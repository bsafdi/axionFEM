"""Parsing rank 0's summary from an MPI `axf` run among launcher/transport warnings.

MPI launchers and transports (UCX, Open MPI, PMIx) can print warnings into the stdout that Design.solve(mpi_ranks=N)
parses, which must not lose the result. design._parse_axf_output must find axf's summary ({"ok": ...}) with arbitrary
non-JSON lines before, after and between its lines, braces and small JSON objects inside the warnings, and must not take
any other object for it. Pure Python (milliseconds); the end-to-end case replaces the subprocess, so no MPI is needed.
"""
import json
from pathlib import Path
import subprocess
import types

import pytest

import axionfem as af
from axionfem import design as D

# warning text as UCX 1.14, OpenMPI 4.1 and PMIx print it on a cluster node (host names replaced)
UCX = [
    "[1730312345.123456] [node9:2345678:0]        ucp_context.c:1176 UCX  WARNING  UCP version is incompatible, "
    "required: 1.14, actual: 1.12 (release 1 /lib64/libucp.so.0)",
    "[1730312345.234567] [node9:2345679:0]          parser.c:2033 UCX  WARNING  unused environment variable: "
    "UCX_TLS_EXTRA (maybe: UCX_TLS?)",
    "[1730312345.345678] [node9:2345680:0]         ib_iface.c:1032 UCX  WARNING  failed to create a CQ {dev=mlx5_0, "
    "cqe=4096}: Cannot allocate memory; set max locked memory (ulimit -l) to 'unlimited'",
    '[1730312345.456789] [node9:2345681:0]    ucs_config.c:519  UCX  WARNING  invalid value {"UCX_NET_DEVICES": '
    '"mlx5_0:1"} ignored',
]
OMPI = [
    "--------------------------------------------------------------------------",
    "WARNING: There was an error initializing an OpenFabrics device.",
    "",
    "  Local host:   node9",
    "  Local device: mlx5_0",
    "--------------------------------------------------------------------------",
    "[node9.hpc:2345678] mca_base_component_repository_open: unable to open mca_btl_openib: librdmacm.so.1: "
    "cannot open shared object file: No such file or directory (ignored)",
    '[node9.hpc:2345678] pmix: {"status": 0, "ranks": [0, 1]}',
    "[node9.hpc:2345679] 1 more process has sent help message help-mpi-btl-openib.txt / error in device init",
    "{}",
]
NOISE = UCX + OMPI

RESULT = {"ok": True, "results": "/scratch/run/results.json",
          "warnings": ["3D linear solve did not converge (2 AMS-PCG iterations, relative residual 0.31); "
                       "{braces} in a message", "second"],
          "coupling": {"Psi_SI_H_T_m2": 1.25e-3, "L_p_H": 2.5e-7, "nested": {"a": [1, 2.5e-10, -3], "b": None,
                                                                             "c": {}, "d": []}}}
ERROR = {"ok": False, "error": "3D linear solve failed", "where": "/mesh", "hint": "raise AXF_GAUGE"}


def _pretty(obj):
    return json.dumps(obj, indent=2)  # the layout of nlohmann's dump(2): one key or value per line


def test_clean_output():
    assert D._parse_axf_output(_pretty(RESULT) + "\n") == RESULT


@pytest.mark.parametrize("where", ["before", "after", "both"])
def test_warnings_around_the_result(where):
    before = "\n".join(NOISE) + "\n" if where in ("before", "both") else ""
    after = "\n".join(NOISE) + "\n" if where in ("after", "both") else ""
    assert D._parse_axf_output(before + _pretty(RESULT) + "\n" + after) == RESULT


# warning lines that cannot be taken for a line of the result (a bare "{}" could: see test_json_form_line_inside)
INTERLEAVED = [n for n in NOISE if n != "{}"]


@pytest.mark.parametrize("every", [1, 3])
def test_warnings_between_the_result_lines(every):
    """Other ranks' UCX/OpenMPI lines interleaved with rank 0's JSON (after every line, or every third), including
    lines that start like JSON ("[...] [...]" after an opening "[")."""
    lines = _pretty(RESULT).splitlines()
    mixed = []
    for k, line in enumerate(lines):
        mixed.append(line)
        if k % every == 0 and k < len(lines) - 1:
            mixed.append(INTERLEAVED[k % len(INTERLEAVED)])
    text = "\n".join(NOISE[:3] + mixed + NOISE[3:]) + "\n"
    with pytest.raises(ValueError):
        json.loads(text)
    assert D._parse_axf_output(text) == RESULT


def test_json_form_line_inside():
    """A foreign line of JSON form ("{}") inside the result could be one of its own lines displaced: the result is
    not accepted from stdout (Design.solve reads the status file, or decides by the exit code)."""
    lines = _pretty(RESULT).splitlines()
    assert D._parse_axf_output("\n".join(lines[:4] + ["{}"] + lines[4:])) is None


def test_error_printed_by_every_rank_among_warnings():
    text = "\n".join([UCX[0], _pretty(ERROR), UCX[1], OMPI[7], _pretty(ERROR), UCX[3]])
    assert D._parse_axf_output(text) == ERROR


def test_no_wrong_object_is_accepted():
    """JSON-like objects in warnings are not axf's summary; without the summary there is no result."""
    assert D._parse_axf_output("\n".join(NOISE)) is None
    for obj in ({"ok": "yes"}, {"ok": False}, {"status": 0}, [{"ok": True}]):
        assert D._parse_axf_output("\n".join(UCX + [json.dumps(obj)] + OMPI)) is None, obj
    # a summary cut short (rank 0 killed while printing) is not completed from the warnings that follow
    cut = "\n".join(_pretty(RESULT).splitlines()[:6])
    assert D._parse_axf_output(cut + "\n" + "\n".join(NOISE)) is None


def test_design_solve_reports_the_error_among_warnings(tmp_path, monkeypatch):
    """End to end through Design.solve(mpi_ranks=2) with the launcher's output replaced: axf's error and hint
    reach the AxionFEMError, not a parse failure."""
    import axionfem._paths as paths

    monkeypatch.setattr(paths, "axf_path", lambda: "/bin/true")
    stdout = "\n".join(UCX[:2] + _pretty(ERROR).splitlines()[:3] + [UCX[2]] + _pretty(ERROR).splitlines()[3:] +
                       OMPI)

    def fake_run(cmd, **kw):
        return types.SimpleNamespace(returncode=1, stdout=stdout, stderr="srun: error: task 0: Exited with exit code 1")

    monkeypatch.setattr(subprocess, "run", fake_run)
    d = af.template("solenoid")
    with pytest.raises(af.AxionFEMError, match=r"3D linear solve failed \(hint: raise AXF_GAUGE\)"):
        d._solve_mpi(tmp_path, 0, 0, 2, "mpirun", None)


# ---- long summaries split by the launcher, status files, killed and failed launches ----
def _nl(obj):
    """nlohmann dump(2) layout: sorted keys, 2-space indent, raw UTF-8."""
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)


def _splice(text, line_no, at, foreign):
    """A foreign line (with its newline) inserted `at` characters into line `line_no` (srun's 1 KiB split)."""
    lines = text.split("\n")
    s = lines[line_no]
    lines[line_no] = s[:at] + foreign + "\n" + s[at:]
    return "\n".join(lines)


LONG_OK = {"ok": True, "results": "/scratch/run/results.json",
           "coupling": {"L_p_H": 2.5e-07, "Psi_SI_H_T_m2": 1.25e-3}, "benchmarks": [{"name": "abra", "Q": 1e6}],
           "warnings": ["w1 " + "x" * 900, "w2 " + "y" * 900, "w3 " + "z" * 900]}
LONG_ERR = {"ok": False, "error": "direct validation FEM solve failed: direct_source", "where": "/fullwave",
            "hint": "Mesh check (run because this solve failed): " + "h" * 900}


@pytest.mark.parametrize("obj, key, at", [(LONG_OK, '"w2', 400), (LONG_OK, '"w3', 400),
                                          (LONG_ERR, '"hint', 300), (LONG_ERR, '"error', 20)])
def test_a_damaged_member_is_never_accepted(obj, key, at):
    """A foreign line spliced into the middle of one of rank 0's lines damages that member: the parser returns None
    (Design.solve then decides by the status file or the exit code), never the object without that member."""
    text = _nl(obj)
    k = next(i for i, s in enumerate(text.split("\n")) if s.strip().startswith(key))
    assert D._parse_axf_output(_splice(text, k, at, UCX[3])) is None


def test_every_mid_line_splice_gives_the_object_or_none():
    text = _nl(LONG_OK)
    lines = text.split("\n")
    for k in range(1, len(lines)):
        if len(lines[k]) >= 4:
            got = D._parse_axf_output(_splice(text, k, len(lines[k]) // 2, UCX[0]))
            assert got is None or got == LONG_OK, k


def test_nested_and_bare_objects_are_not_the_summary():
    summ = {"ok": True, "results": "/r.json", "benchmarks": [{"name": "abra", "ok": False, "error": "nested"}],
            "warnings": ["a", "b"]}
    text = _nl(summ)
    k = next(i for i, s in enumerate(text.split("\n")) if s.strip().startswith('"a"'))
    assert D._parse_axf_output(_splice(text, k, 3, UCX[0])) is None   # not the nested array element
    assert D._parse_axf_output("\n".join([UCX[0], '{"ok": true}', UCX[1]])) is None   # no "results"


def test_line_separators_and_crlf():
    u = dict(LONG_OK, warnings=["w1", "region 'a b' is thin\u0085", "w3"])
    lines = _nl(u).split("\n")
    assert D._parse_axf_output("\n".join(lines[:2] + [UCX[0]] + lines[2:])) == u
    lines = _nl(LONG_OK).split("\n")
    assert D._parse_axf_output("\r\n".join(lines[:3] + [UCX[0]] + lines[3:])) == LONG_OK


def _fake_mpi(tmp_path, monkeypatch, returncode, stdout, status=None, fields=False, stderr=""):
    """Design._solve_mpi with the launch replaced: `status` is what axf would write to its --status-file."""
    import axionfem._paths as paths

    monkeypatch.setattr(paths, "axf_path", lambda: "/bin/true")

    def fake_run(cmd, **kw):
        path = Path(cmd[cmd.index("--status-file") + 1])
        assert path.parent == tmp_path and path.name.startswith("axf_status.") and path.name != "axf_status.json"
        if status is not None:
            path.write_text(_nl(status))
        if fields:
            (tmp_path / "fields").mkdir(exist_ok=True)
            (tmp_path / "fields" / "state.json").write_text("{}")
        return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(subprocess, "run", fake_run)
    loaded = []
    fake_core = types.SimpleNamespace(AxionFEMError=af.AxionFEMError,
                                      Model3D=types.SimpleNamespace(load=lambda path: loaded.append(path) or "model"))
    monkeypatch.setattr(D, "_core", fake_core)
    import axionfem.results as results

    monkeypatch.setattr(results, "Results3D", lambda model, design, wd: ("Results3D", model, wd))
    return loaded


def test_status_file_is_read_first(tmp_path, monkeypatch):
    """A damaged stdout does not matter when rank 0's status file is there."""
    text = _nl(LONG_ERR)
    k = next(i for i, s in enumerate(text.split("\n")) if s.strip().startswith('"hint'))
    d = af.template("solenoid")
    _fake_mpi(tmp_path, monkeypatch, 1, _splice(text, k, 300, UCX[3]), status=LONG_ERR)
    with pytest.raises(af.AxionFEMError, match=r"direct_source \(hint: Mesh check"):
        d._solve_mpi(tmp_path, 0, 0, 2, "mpirun", None)


def test_exit_zero_with_fields_is_a_success_whatever_stdout(tmp_path, monkeypatch):
    """No status file and a stdout damaged beyond use: exit 0 with the fields written is still a success."""
    text = _nl(LONG_OK)
    k = next(i for i, s in enumerate(text.split("\n")) if s.strip().startswith('"w3'))
    d = af.template("solenoid")
    loaded = _fake_mpi(tmp_path, monkeypatch, 0, _splice(text, k, 400, UCX[0]), fields=True)
    res = d._solve_mpi(tmp_path, 0, 0, 2, "mpirun", None)
    assert res[0] == "Results3D" and loaded == [str(tmp_path / "fields")]


def test_another_runs_status_file_is_not_read(tmp_path, monkeypatch):
    """Each launch has a status file of its own (--status-file): a file left by an earlier or concurrent run in the
    same workdir under the default name is not read."""
    (tmp_path / "axf_status.json").write_text(_nl(LONG_OK))
    d = af.template("solenoid")
    _fake_mpi(tmp_path, monkeypatch, 1, "\n".join(UCX + [_pretty(ERROR)]))
    with pytest.raises(af.AxionFEMError, match="3D linear solve failed"):
        d._solve_mpi(tmp_path, 0, 0, 2, "mpirun", None)


def test_exit_zero_does_not_accept_stale_fields(tmp_path, monkeypatch):
    """Every rank is killed before writing anything and the launcher still exits 0: fields/ of an earlier run must
    not be returned as this run's result."""
    (tmp_path / "fields").mkdir()
    (tmp_path / "fields" / "state.json").write_text('{"design": "an earlier run"}')
    d = af.template("solenoid")
    loaded = _fake_mpi(tmp_path, monkeypatch, 0, "\n".join(UCX))
    with pytest.raises(af.AxionFEMError, match=r"MPI solve failed \(exit 0\)"):
        d._solve_mpi(tmp_path, 0, 0, 2, "mpirun", None)
    assert loaded == [] and not (tmp_path / "fields").exists()


def test_success_then_failed_launch_names_exit_code_log_and_fields(tmp_path, monkeypatch):
    """Rank 0 saved the fields and reported success, then another rank failed: the error names the exit code, the log,
    the saved solution and the launcher's stderr."""
    d = af.template("solenoid")
    loaded = _fake_mpi(tmp_path, monkeypatch, 3, _nl(LONG_OK), status=LONG_OK, fields=True,
                       stderr="srun: error: node4: task 1: Exited with exit code 3")
    with pytest.raises(af.AxionFEMError) as info:
        d._solve_mpi(tmp_path, 0, 0, 2, "mpirun", None)
    text = str(info.value)
    assert "exit 3" in text and "after rank 0 reported success" in text and "mpi_solve.log" in text
    assert str(tmp_path / "fields") in text and "task 1: Exited with exit code 3" in text
    assert loaded == []
    assert (tmp_path / "axf_status.json").exists()   # this launch's status file, kept under the default name
