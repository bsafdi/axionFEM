"""Multi-rank launches of the verification cases (single node).

A case that needs several ranks calls `launch(argv, ranks, workdir)`.  The launcher is the one Design.solve and the
tests use (axionfem._paths.default_mpi_launcher: $AXF_MPI_LAUNCHER, else srun inside a SLURM job, else mpirun, else
mpiexec).  Open MPI's mpirun/mpiexec is told to start every rank on this host without binding (`--oversubscribe
--bind-to none`); MPICH's and Intel MPI's Hydra launcher gets `-launcher fork`; srun gets the rank count only.  Each
launch is appended to `workdir/.fluids_mpi.json` (launcher, rank count, exit status) for diagnosis.
"""
import json
import os
import shlex
import shutil
import socket
import subprocess
from pathlib import Path


def _record(workdir: Path, entry: dict) -> None:
    p = Path(workdir) / ".fluids_mpi.json"
    try:
        prev = json.loads(p.read_text()) if p.is_file() else []
    except ValueError:
        prev = []
    prev.append(entry)
    p.write_text(json.dumps(prev, indent=1))


def _launcher() -> list:
    try:
        from axionfem._paths import default_mpi_launcher
        argv = shlex.split(default_mpi_launcher())
    except Exception:  # noqa: BLE001  (an older package: fall back to the PATH)
        argv = []
    if argv and shutil.which(argv[0]):
        return argv
    for name in ("mpirun", "mpiexec"):
        if shutil.which(name):
            return [shutil.which(name)]
    raise RuntimeError("no MPI launcher: set AXF_MPI_LAUNCHER, or put srun (inside a SLURM job), mpirun or mpiexec "
                       "on PATH")


def launch(argv, ranks: int, workdir, timeout: float | None = None, extra_mpirun=()) -> subprocess.CompletedProcess:
    """Run `argv` on `ranks` ranks of this node; raises RuntimeError on a non-zero exit."""
    ranks = int(ranks)
    if ranks < 1:
        raise ValueError("ranks must be >= 1")
    base = _launcher()
    env = dict(os.environ)
    if os.path.basename(base[0]) == "srun":
        cmd = [*base, "-N", "1", "-n", str(ranks), *extra_mpirun, *[str(a) for a in argv]]
        env["SLURM_EXPORT_ENV"] = "ALL"
        for k in ("SLURM_MEM_PER_CPU", "SLURM_MEM_PER_GPU", "SLURM_TRES_PER_TASK"):
            env.pop(k, None)
    else:
        ver = subprocess.run([base[0], "--version"], capture_output=True, text=True, timeout=60)
        text = ver.stdout + ver.stderr
        if "Open MPI" in text or "OpenRTE" in text:
            local = ["--host", f"{socket.gethostname()}:{ranks}", "--oversubscribe", "--bind-to", "none"]
        elif "HYDRA" in text or "Intel(R) MPI" in text:
            local = ["-launcher", "fork"]
        else:
            local = []
        cmd = [*base, "-np", str(ranks), *local, *extra_mpirun, *[str(a) for a in argv]]
        env = {k: v for k, v in env.items() if not k.startswith("SLURM_")}
    r = subprocess.run(cmd, cwd=str(workdir), capture_output=True, text=True, timeout=timeout, env=env)
    _record(Path(workdir), {"launcher": cmd[:len(base)], "ranks": ranks, "argv": [str(a) for a in argv],
                            "returncode": r.returncode})
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(base)} with {ranks} ranks exited {r.returncode}:\n{(r.stdout + r.stderr)[-3000:]}")
    return r
