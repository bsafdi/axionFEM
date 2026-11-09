# AI assistants (MCP)

Lumped-element detectors, cavity haloscopes and SRF heterodyne detectors search for axion dark matter. In this guide we connect an AI assistant to axionFEM through the Model Context Protocol (MCP), an open standard through which assistants call external tools. axionFEM ships an MCP server that runs on the local machine. Through it, an assistant can build, solve, plot and optimize lumped-element detector designs, work with the beta modules for cavity haloscopes, SRF heterodyne detectors and dc SQUID readout chains, and read and write COMSOL files. The numbers the tools return are computed by axionFEM and come with its checks and warnings; the assistant should quote them rather than compute its own.

Any MCP client that can start a local (stdio) server works. Below we give the configuration for a generic client and for two clients from Anthropic, Claude Code and Claude Desktop.

## Installation

Install the MCP package into the Python environment where axionFEM is installed (from the
[Installation page](install.md)):

```bash
pip install "mcp>=2.0"
```

Alternatively, reinstall with the `mcp` extra: after the sequential build
`AXIONFEM_DEPS=$PWD/third_party/install-seq pip install ".[mesh,mcp]"`, after the MPI build
`pip install --no-build-isolation -e ".[mcp]"`. Once wheels are on PyPI, the extra is `pip install "axionfem[mcp]"`. Either way the server is the command `axionfem-mcp` in the Python environment's `bin` directory; `python -m axionfem.mcp_server` starts the same server.

## Workspace and environment

The server keeps designs (`designs/<name>.json`) and results (`runs/<name>/`) in the directory `AXF_WORKSPACE` (default `./axf_workspace`, relative to the directory the client starts the server in). It inherits the environment of the client that starts it, so we set its variables in the client configuration:

| variable | meaning |
|---|---|
| `AXF_WORKSPACE` | the workspace directory |
| `AXF_THREADS` | OpenMP threads per solve (default 4) |
| `AXF_MESHER` | path of the `axf-mesher` program, when it is not found next to the package or in `build/` |
| `LD_LIBRARY_PATH` | for a source build whose dependency libraries are not on the loader path (what `source scripts/env.sh` sets) |

## Client configuration

### Any MCP client

The server speaks MCP over standard input and output. A client needs the command, optional arguments and the environment; most clients take them as JSON of this form:

```json
{
  "mcpServers": {
    "axionfem": {
      "command": "/path/to/venv/bin/axionfem-mcp",
      "args": [],
      "env": {"AXF_WORKSPACE": "/path/to/axf_workspace", "AXF_THREADS": "4"}
    }
  }
}
```

Use absolute paths: a client does not necessarily start the server in a directory of your choice.

### Claude Code

We register the server with Claude Code:

```bash
claude mcp add axionfem -e AXF_WORKSPACE=/path/to/axf_workspace -- /path/to/venv/bin/axionfem-mcp
```

With `--scope project` the entry is written to the project's `.mcp.json` instead of your user settings. The repository already contains such a file: started in the root of a source checkout whose virtual environment is `.venv`, Claude Code offers the `axionfem` server with the workspace `axf_workspace` in the checkout. Edit the `command` in `.mcp.json` if your environment lives elsewhere. The repository also contains a Claude Code skill, `.claude/skills/axionfem-design`, that summarizes the design workflow.

### Claude Desktop

We add the server to Claude Desktop's configuration file, `claude_desktop_config.json` (Settings, Developer, Edit Config):

```json
{
  "mcpServers": {
    "axionfem": {
      "command": "/path/to/venv/bin/axionfem-mcp",
      "env": {"AXF_WORKSPACE": "/path/to/axf_workspace"}
    }
  }
}
```

Restart Claude Desktop after editing the file.

The assistant should start with the `axionfem_guide` tool, which explains the workflow, units and conventions; the server also sends the same text to the client as its instructions when it starts.

## Long jobs

`solve_design`, `sweep`, `sensitivities`, `optimize_design`, `optimize_global` and `optimize` run in a worker subprocess. While they run they send MCP progress notifications; `list_jobs` shows what is running, and `cancel_job(job_id)` (or the client's own cancellation of the request) stops the worker. A crash in the solver ends the worker, not the server. Some clients stop waiting for a tool after a fixed time; for long sweeps and optimizations, raise the client's tool timeout if it has one.

## Example

> **User:** “Build an ABRACADABRA-style toroid with R = H = 2 m and inner radius ratio y = 0.6, solve it, show me a section plot and tell me the pickup inductance and the signal coupling.”
>
> The assistant calls `create_design(template="toroid", name="toroid", parameters={"R": 2, "H": 2, "y": 0.6})`, `solve_design` and `render_design` (section, `Bmag`), then `solve_design` with `order = 3` to check convergence.
>
> **Assistant:** “The pickup inductance is L_p ≈ 1.458 µH and the signal coupling is Ψ ≈ 3.09 × 10⁻⁷ H T m² (0.246 T m³ in natural units). At order 3, Ψ changes by 4 × 10⁻⁶ and L_p by 1 × 10⁻⁶ (relative), so these digits are converged. The static field stays inside the winding box and falls off as 1/r, from 0.167 T at the inner wall to 0.10 T at the outer wall, matching μ₀NI/(2πr) with NI = 10⁶ A. L_p computed from flux and from energy agrees to 5 × 10⁻¹³. To reach the optimistic benchmark the peak field would have to rise to about 7.4 T.”

Before quoting a number, the assistant should read the `warnings` and `checks` of the result and compare two orders or meshes, as here.

## Tools

| Tool | Purpose |
|---|---|
| `axionfem_guide` | Workflow, units and physics conventions |
| `list_templates`, `create_design`, `put_design`, `get_design`, `list_designs`, `design_schema` | Starting from a template or writing a design document |
| `set_parameters` | Changing parameters of a saved design, or saving a variant under a new name |
| `solve_design` | Meshing and solving: Ψ, L_p, F, fields, forces, benchmarks, cost metric |
| `render_design`, `start_viewer` | PNG section or 3D cutaway; live browser viewer |
| `requirement` | F_req for a noise benchmark |
| `sweep` | Solving for several values of one parameter |
| `sensitivities`, `list_objectives` | Shape derivatives of the figures of merit |
| `optimize_design`, `optimize_global`, `optimize` | Gradient-based (adjoint), global (differential evolution) or derivative-free (bounded Powell) optimization |
| `set_fullwave`, `set_hts`, `make_3d`, `list_conductors` | Full-wave sweep, HTS screening currents, 3D designs, conductor critical surfaces |
| `list_jobs`, `cancel_job` | Following or stopping long jobs |

### Other detector types

These tools wrap the same Python functions as the walkthroughs for [cavity haloscopes](../../examples/cavity_haloscope.ipynb), [SRF heterodyne detectors](../../examples/srf_heterodyne.ipynb), [dc SQUID readout chains](../../examples/readout_squid.ipynb) and [COMSOL export and import](comsol.md). The cavity, SRF and SQUID modules, and the COMSOL importer, are beta.

| Tool | Purpose |
|---|---|
| `cavity_presets`, `cavity_preset_limit` | Published cavity haloscope runs and the exclusion recomputed from their numbers |
| `cavity_scan` | Scan rate, optimal coupling, excluded coupling and time to cover a band for one cavity mode |
| `create_cavity` | Pillbox or coaxial cavity design; `solve_design` then returns its modes, form factors and Q |
| `srf_design`, `srf_mode_pair` | Volume, frequency, RF power and cost metric of an SRF heterodyne detector (`geometry`: `projectx`, the computed default cell, or `equator_diagnostic`); figures of merit of a mode pair |
| `squid_chain`, `squid_reach` | Added noise of a dc SQUID readout chain; reach of a solved design read out by it |
| `export_comsol`, `inspect_comsol`, `import_comsol` | Writing a design for COMSOL; summarising or importing a `.mph` file |
| `export_touchstone` | A solved design's full-wave S-, Z- or Y-parameters as a Touchstone v1 file, with passivity and reciprocity checks |

Asked for the scan rate of a cavity at 5.75 GHz with B = 9 T, V = 1.5 L, C = 0.5, Q0 = 30 000, a quantum-limited amplifier and the cavity at 50 mK, the assistant calls `cavity_scan` and reports 4.6 Hz/s at the KSVZ coupling, with β = 2, and 125 days to cover 5.75–5.80 GHz.

## Limits

- The tools write only inside the workspace (`designs/`, `runs/`, `jobs/`, `imports/`). `inspect_comsol` and `import_comsol` read the `.mph` file named in the call.
- `start_viewer` serves the workspace's results on `127.0.0.1` (default port 8765); on a remote machine, forward the port (`ssh -L 8765:localhost:8765 host`).
- The fluids, scalar-transport and chemistry modules and the cavity probe coupling have no MCP tools; use them from Python.
- 3D designs need an MPI build of axionFEM (`af.build_info()["engine_3d"]`). The server solves in its own worker process and does not submit jobs to a cluster scheduler.
- The server reports what the solver computed. Results are only as good as the mesh: the assistant should compare orders or meshes and read the warnings before quoting numbers.
