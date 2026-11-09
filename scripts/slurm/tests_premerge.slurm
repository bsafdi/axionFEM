#!/bin/bash
#SBATCH --job-name=axf_premerge
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=8
#SBATCH --time=01:15:00
#SBATCH --output=%x_%j.log
# Site-specific options: remove one '#' from these lines and fill them in, or give them to sbatch (-A, -p, -q), or
# set SBATCH_ACCOUNT, SBATCH_PARTITION and SBATCH_QOS in the environment.
##SBATCH --account=<account>
##SBATCH --partition=<partition>
##SBATCH --qos=<qos>
#
# Example SLURM job: the quick pre-merge subset (every C++ unit test, then the pytest node IDs in tests/premerge.txt
# with AXF_SLOW=1 and AXF_MPI=1), about 30 min on one node with 16 tasks x 8 CPUs. Submit it from the root of a source
# checkout with a build:
#
#   export SBATCH_ACCOUNT=<account> SBATCH_PARTITION=<partition>
#   sbatch scripts/slurm/tests_premerge.slurm
#   sbatch --export=ALL,AXF_BUILD_DIR=$PWD/build-debug scripts/slurm/tests_premerge.slurm   # another build directory
#   python scripts/premerge_tests.py check       (without a job, seconds: the list against the suite)
#
# Optional variables:
#   AXF_BUILD_DIR    the CMake build directory with axf, axf-mesher and the C++ tests (default ./build)
#   AXF_PYTHONPATH   the package directory with its compiled _core (default ./python, the in-place package)
#   AXF_PYTHON       the Python interpreter (default ./.venv/bin/python)
#   AXF_ENV_HOOK     a file of site setup (module loads, ...) that scripts/env.sh sources
#   AXF_MPI_LAUNCHER the MPI launcher for the multi-rank tests (default: srun inside a job)
#   AXF_TMPDIR       where the job's temporary directory is created (default $TMPDIR, else /tmp)
# Extra arguments pass through to pytest (e.g. -x). 16 tasks: one test needs 16 MPI slots. The job ends with
# PREMERGE OK or PREMERGE FAILED.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
echo "PREMERGE COMMIT $(git rev-parse HEAD)"
echo "PREMERGE DIRTY [$(git status --porcelain --untracked-files=no -- cpp cmake CMakeLists.txt python benchmarks tests | wc -l)]"
source scripts/env.sh
BUILD="${AXF_BUILD_DIR:-${SLURM_SUBMIT_DIR}/build}"
PY="${AXF_PYTHON:-${SLURM_SUBMIT_DIR}/.venv/bin/python}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} PYTHONDONTWRITEBYTECODE=1 AXF_SLOW=1 AXF_MPI=1
export PYTHONPATH="${AXF_PYTHONPATH:-${SLURM_SUBMIT_DIR}/python}"
export AXF_EXE="${BUILD}/axf" AXF_MESHER="${BUILD}/axf-mesher"
# a temporary directory outside any git repository (some tests must not create theirs inside one)
export TMPDIR="${AXF_TMPDIR:-${TMPDIR:-/tmp}}/axf_premerge_${SLURM_JOB_ID:-$$}"
mkdir -p "${TMPDIR}"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK  # srun: conflicting inherited settings
export SLURM_EXPORT_ENV=ALL  # srun passes this environment to the ranks (some sites default to NONE)
echo "build: ${BUILD}   package: ${PYTHONPATH}   python: ${PY}"
"$PY" -c "import axionfem._core as c; print('core:', c.__file__)" || { echo "PREMERGE FAILED: no compiled core"; exit 1; }
T0=$(date +%s)
ctest --test-dir "${BUILD}" --output-on-failure
CTEST=$?
T1=$(date +%s)
"$PY" scripts/premerge_tests.py run "$@"
PYTEST=$?
T2=$(date +%s)
echo "ctest exit ${CTEST} ($(( T1 - T0 )) s); pytest subset exit ${PYTEST} ($(( T2 - T1 )) s); total $(( T2 - T0 )) s"
if [ "${CTEST}" -ne 0 ] || [ "${PYTEST}" -ne 0 ]; then
  echo "PREMERGE FAILED"
  exit 1
fi
echo "PREMERGE OK"
