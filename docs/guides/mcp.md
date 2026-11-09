# Chat assistants (MCP)

Lumped-element detectors, cavity haloscopes and SRF heterodyne detectors search for axion dark matter. In this guide we configure chat assistants to use axionFEM's server for the Model Context Protocol (MCP), an open standard through which chat assistants such as Claude Code and Claude Desktop call external tools. With it, an assistant can build, solve, plot and optimize lumped-element detector designs and work with the beta modules for cavity haloscopes, SRF heterodyne detectors and dc SQUID readout chains and with COMSOL files on the local machine through axionFEM; the results it reports are the solver's.

## Installation

In the source checkout from the [Installation page](install.md), install the MCP extra (after the MPI build; after the
sequential build, `AXIONFEM_DEPS=$PWD/third_party/install-seq pip install ".[mesh,mcp]"`):

```bash
pip install --no-build-isolation -e ".[mcp]"
```

Wheels are not yet on PyPI; once they are, the same extra is `pip install "axionfem[mcp]"`. The server is the command
`axionfem-mcp` in the environment's `bin` directory.

The server keeps designs (`designs/<name>.json`) and results (`runs/<name>/`) in the directory `AXF_WORKSPACE` (default `./axf_workspace`).
It inherits the environment of the client that starts it, so we set `AXF_WORKSPACE` (and, if needed, `AXF_THREADS`, `AXF_MESHER` or `LD_LIBRARY_PATH`) in the client configuration.

## Client configuration

### Claude Code

We register the server with Claude Code to enable axionFEM tool calls.

```bash
claude mcp add axionfem -e AXF_WORKSPACE=$HOME/axf_workspace -- /path/to/axionFEM/.venv/bin/axionfem-mcp
```

### Claude Desktop

We add the server to `claude_desktop_config.json` to enable axionFEM tool calls.

```json
{
  "mcpServers": {
    "axionfem": {
      "command": "/path/to/axionFEM/.venv/bin/axionfem-mcp",
      "env": {"AXF_WORKSPACE": "/path/to/axf_workspace"}
    }
  }
}
```

Other MCP clients that start a local (stdio) server take the same command and environment. Some clients stop waiting
for a tool after a fixed time; for long sweeps and optimizations, we raise the client's tool timeout if it has one.

The assistant should start with the `axionfem_guide` tool, which explains the workflow, units and conventions; most clients read it automatically when the server starts.

## Example

> **User:** “Build an ABRACADABRA-style toroid with R = H = 2 m and inner radius ratio y = 0.6, solve it, show me a section plot and tell me the pickup inductance and the signal coupling.”
>
> The assistant calls `create_design` (template `"toroid"`, `R = 2`, `H = 2`, `y = 0.6`), `solve_design` and `render_design` (section, `Bmag`).
>
> **Assistant:** “The pickup inductance is L_p ≈ 1.458 µH and the signal coupling is Ψ ≈ 3.09 × 10⁻⁷ H T m² (0.246 T m³ in natural units). The static field stays inside the winding box and falls off as 1/r, from 0.167 T at the inner wall to 0.10 T at the outer wall, matching μ₀NI/(2π · 1.2 m). L_p computed from flux and from energy agrees to 5 × 10⁻¹³. To reach the optimistic benchmark the peak field would have to rise to about 7.4 T.”

A convergence check (order 3) should be requested and the `warnings` read before quoting a number.

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

These tools wrap the same Python functions as the walkthroughs for [cavity haloscopes](../../examples/cavity_demo.ipynb), [SRF heterodyne detectors](../../examples/srf_demo.ipynb), [dc SQUID readout chains](../../examples/squid_chain_demo.ipynb) and [COMSOL export and import](comsol.md). The cavity, SRF and SQUID modules, and the COMSOL importer, are beta.

| Tool | Purpose |
|---|---|
| `cavity_presets`, `cavity_preset_limit` | Published cavity haloscope runs and the exclusion recomputed from their numbers |
| `cavity_scan` | Scan rate, optimal coupling, excluded coupling and time to cover a band for one cavity mode |
| `create_cavity` | Pillbox or coaxial cavity design; `solve_design` then returns its modes, form factors and Q |
| `srf_design`, `srf_mode_pair` | Volume, frequency, RF power and cost metric of an SRF heterodyne detector (`geometry`: `projectx`, the computed default cell, or `equator_diagnostic`); figures of merit of a mode pair |
| `squid_chain`, `squid_reach` | Added noise of a dc SQUID readout chain; reach of a solved design read out by it |
| `export_comsol`, `inspect_comsol`, `import_comsol` | Writing a design for COMSOL; summarising or importing a `.mph` file |
| `export_touchstone` | A solved design's full-wave S-, Z- or Y-parameters as a Touchstone v1 file, with passivity and reciprocity checks |

Asked for the scan rate of a cavity at 5.75 GHz with B = 9 T, V = 1.5 L, C = 0.5, Q0 = 30 000, a quantum-limited amplifier and the cavity at 50 mK, the assistant called `cavity_scan` and reported 4.6 Hz/s at the KSVZ coupling, with β = 2, and 125 days to cover 5.75–5.80 GHz.
