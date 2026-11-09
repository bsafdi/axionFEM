"""Test meshes for tests/test_cavity3d.py (imported with mesh.file / mesh.info_file).

Exactly D4-symmetric (in x-y) tet mesh of a square box [0,a]^2 x [0,d]: uniform n x n x m cells, each split
into 24 tets (cell centre + face centres), written as gmsh MSH 2.2 ASCII (volume attr 2 "cavity", boundary
attr 901 "outer") plus a mesh info JSON in the format of axf-mesher. The mirror symmetry makes continuum-
degenerate pairs (TE101/TE011) degenerate to round-off on the discrete level. ``sheet=True`` adds a PEC
boundary group 902 "sheet" on interior faces (half of the plane x = a/2: a zero-thickness iris)."""
import json
import sys


def build(a, d, n, m, path_msh, path_info, sheet=False):
    hx, hz = a / n, d / m
    nodes = {}

    def node(x, y, z):
        key = (round(x / hx * 2), round(y / hx * 2), round(z / hz * 2))  # half-integer lattice
        if key not in nodes:
            nodes[key] = (len(nodes) + 1, key[0] * hx / 2, key[1] * hx / 2, key[2] * hz / 2)
        return nodes[key][0]

    tets, tris, sheet_tris = [], [], []
    for i in range(n):
        for j in range(n):
            for k in range(m):
                x0, y0, z0 = i * hx, j * hx, k * hz
                C = node(x0 + hx / 2, y0 + hx / 2, z0 + hz / 2)
                corners = lambda ax, s: None  # noqa
                # faces: axis ax, side s; the 4 corners in cyclic order
                for ax in range(3):
                    for s in (0, 1):
                        pts = []
                        for (u, v) in ((0, 0), (1, 0), (1, 1), (0, 1)):
                            c = [None, None, None]
                            c[ax] = s
                            o = [q for q in range(3) if q != ax]
                            c[o[0]], c[o[1]] = u, v
                            pts.append((x0 + c[0] * hx, y0 + c[1] * hx, z0 + c[2] * hz))
                        fc = [0, 0, 0]
                        fc[ax] = s
                        o = [q for q in range(3) if q != ax]
                        fc[o[0]] = fc[o[1]] = 0.5
                        F = node(x0 + fc[0] * hx, y0 + fc[1] * hx, z0 + fc[2] * hz)
                        P = [node(*p) for p in pts]
                        on_bdr = (ax == 0 and ((s == 0 and i == 0) or (s == 1 and i == n - 1))) or \
                                 (ax == 1 and ((s == 0 and j == 0) or (s == 1 and j == n - 1))) or \
                                 (ax == 2 and ((s == 0 and k == 0) or (s == 1 and k == m - 1)))
                        on_sheet = sheet and ax == 0 and s == 1 and i == n // 2 - 1 and j < n // 2
                        for q in range(4):
                            tets.append((C, F, P[q], P[(q + 1) % 4]))
                            if on_bdr:
                                tris.append((F, P[q], P[(q + 1) % 4]))
                            if on_sheet:
                                sheet_tris.append((F, P[q], P[(q + 1) % 4]))
    with open(path_msh, "w") as f:
        names = '2 901 "outer"\n' + ('2 902 "sheet"\n' if sheet else "") + '3 2 "cavity"\n'
        f.write(f'$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$PhysicalNames\n{3 if sheet else 2}\n{names}$EndPhysicalNames\n')
        f.write(f"$Nodes\n{len(nodes)}\n")
        for idx, x, y, z in sorted(nodes.values()):
            f.write(f"{idx} {x:.17g} {y:.17g} {z:.17g}\n")
        f.write("$EndNodes\n$Elements\n%d\n" % (len(tris) + len(sheet_tris) + len(tets)))
        e = 1
        for t in tris:
            f.write(f"{e} 2 2 901 1 {t[0]} {t[1]} {t[2]}\n"); e += 1
        for t in sheet_tris:
            f.write(f"{e} 2 2 902 2 {t[0]} {t[1]} {t[2]}\n"); e += 1
        X = {v[0]: v[1:] for v in nodes.values()}
        for t in tets:
            p = [X[q] for q in t]
            u = [[p[r][s] - p[0][s] for s in range(3)] for r in (1, 2, 3)]
            det = (u[0][0] * (u[1][1] * u[2][2] - u[1][2] * u[2][1]) - u[0][1] * (u[1][0] * u[2][2] - u[1][2] * u[2][0])
                   + u[0][2] * (u[1][0] * u[2][1] - u[1][1] * u[2][0]))
            if det < 0:
                t = (t[0], t[2], t[1], t[3])
            f.write(f"{e} 4 2 2 1 {t[0]} {t[1]} {t[2]} {t[3]}\n"); e += 1
        f.write("$EndElements\n")
    groups = [{"attr": 901, "name": "outer", "type": "pec"}] + ([{"attr": 902, "name": "sheet", "type": "pec"}] if sheet else [])
    info = {"boundary_groups": groups, "device_radius": a,
            "domain_center": [0, 0, 0], "domain_radius": 2 * a, "elements_3d": len(tets), "gmsh_version": "manual",
            "groups": [{"attr": 2, "dim": 3, "entities": 1, "name": "cavity"},
                       {"attr": 901, "dim": 2, "entities": 6, "name": "outer"}]
                      + ([{"attr": 902, "dim": 2, "entities": 1, "name": "sheet"}] if sheet else []),
            "ok": True, "pec_conductors": [], "removed_pec_volumes": 0, "shell_attr": -1,
            "shell_inner_radius": 2 * a, "shell_outer_radius": 2 * a, "size": hx, "surface_size": hx, "symmetry": []}
    json.dump(info, open(path_info, "w"), indent=1)
    return len(tets)

