"""How the tests start a multi-rank run: srun, mpirun or mpiexec.

The launcher is the one Design.solve uses (axionfem._paths.default_mpi_launcher: $AXF_MPI_LAUNCHER, else srun inside a
SLURM job when srun is on PATH, else mpirun, else mpiexec), or None when it is not installed:

    argv = command(2) + [sys.executable, "script.py"]          # e.g. ["srun", "-n", "2", ...]
    subprocess.run(argv, env=environment(argv), ...)

- Open MPI's mpirun/mpiexec also gets --oversubscribe and --bind-to none (a test may ask for more ranks than the
  allocation's slots, and the ranks must not all be bound to one core); MPICH's and Intel MPI's do not know them.
- environment(argv): for srun, the settings Design.solve makes (srun refuses mutually exclusive SLURM memory variables
  inherited from the allocation; SLURM_EXPORT_ENV=ALL so the ranks get this environment); for mpirun/mpiexec, no
  SLURM_* variables, so that the ranks start on this node instead of through the allocation.

Inside a SLURM job srun is chosen even where mpirun is installed too, as Design.solve does. Where that does not work,
set AXF_MPI_LAUNCHER (it may carry options):
- tests started from inside an `srun` step (e.g. an interactive `srun --pty bash`): AXF_MPI_LAUNCHER="srun --overlap",
  or the inner srun waits for the resources the outer step holds;
- an MPI that srun cannot start without a plugin option (Open MPI through PMIx, for example):
  AXF_MPI_LAUNCHER="srun --mpi=pmix", or AXF_MPI_LAUNCHER=mpirun. Without it each rank may start as a separate
  one-rank job, which the tests report as a wrong rank count.
"""
import os
import shlex
import shutil
import subprocess


def launcher() -> list | None:
    """The launcher's argv (without the rank count), or None when none is installed."""
    from axionfem._paths import default_mpi_launcher
    argv = shlex.split(default_mpi_launcher())
    return argv if argv and shutil.which(argv[0]) else None


def available() -> bool:
    return launcher() is not None


def skip_reason() -> str:
    return "no MPI launcher (srun inside a SLURM job, mpirun or mpiexec on PATH, or AXF_MPI_LAUNCHER)"


def flavour(argv0: str) -> str:
    """'srun', 'openmpi', 'hydra' (MPICH, Intel MPI) or 'unknown', from the name and `--version`."""
    if os.path.basename(argv0) == "srun":
        return "srun"
    try:
        v = subprocess.run([argv0, "--version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    text = v.stdout + v.stderr
    if "Open MPI" in text or "OpenRTE" in text:
        return "openmpi"
    if "HYDRA" in text or "Intel(R) MPI" in text:
        return "hydra"
    return "unknown"


def command(nranks: int) -> list:
    """argv prefix that starts `nranks` ranks; RuntimeError when there is no launcher."""
    argv = launcher()
    if argv is None:
        raise RuntimeError(skip_reason())
    extra = ["--oversubscribe", "--bind-to", "none"] if flavour(argv[0]) == "openmpi" else []
    return argv + ["-n", str(int(nranks))] + extra


def environment(argv: list, base: dict | None = None) -> dict:
    """The environment for a launch with `argv` (see the module docstring), from `base` (default: os.environ)."""
    env = dict(os.environ if base is None else base)
    if os.path.basename(argv[0]) == "srun":
        if "SLURM_MEM_PER_NODE" in env:
            env.pop("SLURM_MEM_PER_CPU", None)
            env.pop("SLURM_MEM_PER_GPU", None)
        env.pop("SLURM_TRES_PER_TASK", None)
        env["SLURM_EXPORT_ENV"] = "ALL"
        return env
    return {k: v for k, v in env.items() if not k.startswith("SLURM_")}
