#!/bin/bash
#SBATCH --job-name=axf_tests
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log
# Site-specific options: remove one '#' from these lines and fill them in, or give them to sbatch (-A, -p, -q), or
# set SBATCH_ACCOUNT, SBATCH_PARTITION and SBATCH_QOS in the environment.
##SBATCH --account=<account>
##SBATCH --partition=<partition>
##SBATCH --qos=<qos>
#
# Example SLURM job: the full test suite (C++ ctest, then pytest on tests/) on one node. Submit it from the root of
# a source checkout with a build. Account, partition and QoS are site-specific: set them on the command line or
# through the environment variables sbatch reads. Extra arguments pass through to pytest:
#
#   export SBATCH_ACCOUNT=<account> SBATCH_PARTITION=<partition>
#   sbatch --export=ALL,AXF_SLOW=1 scripts/slurm/tests.slurm                          # everything (about 11 h)
#   sbatch scripts/slurm/tests.slurm                                                   # without the slow tests
#   sbatch --ntasks=4 scripts/slurm/tests.slurm tests/test_fullwave3d_strumpack_mpi.py # one file
#
# AXF_MPI=1 is set, so the multi-rank tests run (16 tasks, because one test needs 16 MPI slots); AXF_SLOW=1 adds the
# slow 3D tests; where the time limit is shorter than the full suite needs, pass groups of test files to several jobs
# (docs/developer/testing.md). Optional variables:
#   AXF_BUILD_DIR    the CMake build directory with axf, axf-mesher and the C++ tests (default ./build)
#   AXF_PYTHONPATH   the package directory with its compiled _core (default ./python, the in-place package)
#   AXF_PYTHON       the Python interpreter (default ./.venv/bin/python)
#   AXF_ENV_HOOK     a file of site setup (module loads, ...) that scripts/env.sh sources
#   AXF_MPI_LAUNCHER the MPI launcher for the multi-rank tests (default: srun inside a job)
#   AXF_TMPDIR       where the job's temporary directory is created (default $TMPDIR, else /tmp)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/env.sh
BUILD="${AXF_BUILD_DIR:-${SLURM_SUBMIT_DIR}/build}"
PY="${AXF_PYTHON:-${SLURM_SUBMIT_DIR}/.venv/bin/python}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} PYTHONDONTWRITEBYTECODE=1 AXF_MPI=1
export PYTHONPATH="${AXF_PYTHONPATH:-${SLURM_SUBMIT_DIR}/python}"
export AXF_EXE="${BUILD}/axf" AXF_MESHER="${BUILD}/axf-mesher"
export TMPDIR="${AXF_TMPDIR:-${TMPDIR:-/tmp}}/axf_tests_${SLURM_JOB_ID:-$$}"
mkdir -p "${TMPDIR}"
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_TRES_PER_TASK  # srun: conflicting inherited settings
export SLURM_EXPORT_ENV=ALL  # srun passes this environment to the ranks (some sites default to NONE)
echo "build: ${BUILD}   package: ${PYTHONPATH}   python: ${PY}"
ctest --test-dir "${BUILD}" --output-on-failure
CTEST=$?
"$PY" -m pytest -q -rfEs tests "$@"
PYTEST=$?
echo "ctest exit ${CTEST}; pytest exit ${PYTEST}"
[ "${CTEST}" -eq 0 ] && [ "${PYTEST}" -eq 0 ]
