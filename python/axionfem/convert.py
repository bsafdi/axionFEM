"""Convert axisymmetric designs to full 3D designs (revolve every profile about z).

- Regions: shape -> {"revolve": {"profile": shape}} over 360 degrees.
- An azimuthal (slit-sheath) pickup:
  - pickup="ideal" (default): the sheath stays closed and carries the readout current as a prescribed net
    current around the z axis (3D pickup type "azimuthal"). This is the axisymmetric ideal slit, so 3D and 2D
    results are directly comparable. Other `azimuthal_slit` bodies keep their flag (no net RF current).
  - pickup="slit": every `azimuthal_slit` body gets a real meridional slit of `slit_deg`; the pickup's slit is
    filled with a vacuum port region that feeds the readout current. Results then depend on how the port
    feeds the current (uniform conductivity), an O(1 %) effect for the toroid sheath.
- A poloidal pickup keeps its port region, revolved into an annular port.
- Zero-thickness curves are not supported in 3D (use thin regions).
"""
from __future__ import annotations

import copy

from .design import Design


def to_3d(design: Design | dict, slit_deg: float = 4.0, mesh: dict | None = None, pickup: str = "ideal") -> Design:
    if pickup not in ("ideal", "slit"):
        raise ValueError("pickup must be 'ideal' or 'slit'")
    doc = copy.deepcopy(design.doc if isinstance(design, Design) else design)
    if doc.get("symmetry", "axisymmetric") != "axisymmetric":
        raise ValueError("design is already 3D")
    if doc.get("curves"):
        raise ValueError("zero-thickness curves cannot be revolved; model sheets as thin regions")
    pick = doc.get("pickup")
    pick_body = pick["conductor"] if pick and pick.get("type") == "azimuthal" else None
    slit_bodies = {s["body"] for s in doc.get("superconductors", []) if s.get("azimuthal_slit")}
    if pickup == "slit" and pick_body:
        slit_bodies.add(pick_body)
    regions = []
    for r in doc["regions"]:
        rr = {k: v for k, v in r.items() if k != "shape"}
        if pickup == "slit" and r["name"] in slit_bodies:
            rr["solid"] = {"revolve": {"profile": r["shape"], "angles": [slit_deg / 2, 360 - slit_deg / 2]}}
            regions.append(rr)
            if r["name"] == pick_body:
                regions.append({"name": f"{r['name']}_slit_port", "material": "vacuum",
                                "solid": {"revolve": {"profile": r["shape"], "angles": [-slit_deg / 2, slit_deg / 2]}}})
        else:
            rr["solid"] = {"revolve": {"profile": r["shape"]}}
            regions.append(rr)
    doc["regions"] = regions
    doc["symmetry"] = "3d"
    if pick:
        if pick["type"] == "azimuthal":
            if pickup == "slit":
                doc["pickup"] = {"type": "port", "port": f"{pick_body}_slit_port"}
            else:
                doc["pickup"] = {"type": "azimuthal", "conductor": pick_body}
        elif pick["type"] == "poloidal":
            doc["pickup"] = {"type": "port", "port": pick["port"]}
    if pickup == "slit":
        for s in doc.get("superconductors", []):
            s.pop("azimuthal_slit", None)  # the slits are real now
    if mesh:
        doc.setdefault("mesh", {}).update(mesh)
    doc["name"] = doc.get("name", "design") + "_3d"
    return Design(doc)
