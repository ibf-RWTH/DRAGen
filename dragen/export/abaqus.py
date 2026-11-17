"""Abaqus input of an RVEResult: mesh, sets, sections and periodic boundary conditions, plus the
grain data file of the crystal plasticity UMAT. Materials and steps come from the user (author's
decision 04.11.2026): the written file is model data to be included into a job file, e.g.

    *Include, input=rve.inp
    *Include, input=materials.inp      ** one *Material per name used in the sections
    *Step ...                          ** loads on RP_X, RP_Y, RP_Z (see below)
"""
from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np

from dragen.export.mesh import MeshKind, solver_mesh

_ELEMENT = {'hexahedron': 'C3D8', 'tetra': 'C3D4', 'tetra10': 'C3D10',
            'quad': 'CPE4', 'triangle': 'CPE3', 'triangle6': 'CPE6'}
_NODES = {'C3D8': 8, 'C3D8R': 8, 'C3D8H': 8, 'C3D8RH': 8, 'C3D4': 4, 'C3D4H': 4,
          'C3D10': 10, 'C3D10M': 10, 'C3D10H': 10, 'C3D10HS': 10,
          'CPE4': 4, 'CPE4R': 4, 'CPS4': 4, 'CPS4R': 4, 'CPE3': 3, 'CPS3': 3, 'CPE6': 6, 'CPS6': 6,
          'CPE6M': 6, 'CPS6M': 6}
UNITS = {'m': 1.0, 'mm': 1e3, 'um': 1e6}
INSTANCE = 'RVE-1'


def material_name(result, grain: int, material_per: str = 'grain') -> str:
    """Material of a grain in the sections: '<Phase>_<grain>' (legacy naming, e.g. Ferrite_12)
    or, with material_per='phase', '<Phase>'."""
    phase = result.phase_names[int(result.phase_ids[grain - 1])].capitalize()
    return f'{phase}_{grain}' if material_per == 'grain' else phase


def export_abaqus(result, path: Path, mesh: MeshKind = 'auto', length_unit: str = 'm',
                  element_type: Optional[str] = None,
                  material_per: Literal['grain', 'phase'] = 'grain',
                  thickness: Optional[float] = None) -> Dict[str, Path]:
    """Write <path>.inp and graindata.inp next to it; returns both paths.

    <path>.inp
        Part RVE with the nodes, the elements (C3D8 for voxels, C3D4/C3D10 for tets, or
        `element_type`), element sets GRAIN_<g>, <PHASE> (e.g. FERRITE) and BAND_<k> (after a
        substructure GRAIN_<g> are the regions, plus PARENT_<g> and PACKET_<p>), node sets
        of the box faces (LEFT/RIGHT x, BOTTOM/TOP y, BACK/FRONT z), one solid section per grain
        with material material_name(...). Assembly with instance RVE-1 and, for every periodic
        axis a, a reference node RP_X/RP_Y/RP_Z whose displacement is the jump of the
        displacement across the box along a: u(x + L_a e_a) - u(x) = u(RP_a). A macroscopic
        deformation gradient F is applied as u(RP_a) = (F - I) L_a e_a. Periodic nodes are
        grouped into classes (a box corner: 8 nodes); each slave gets one equation per DOF to the
        class master, u_s - u_m - sum_a k_a u(RP_a) = 0, so no constraint is redundant. Node set
        ORIGIN is the master at the box origin (fix it against rigid body motion).
        Coordinates in `length_unit` ('m', 'mm' or 'um').
        2D: plane strain elements CPE4 (pixels), CPE3/CPE6 (triangles), or CPS* via
        element_type; sections with thickness `thickness` (default: one pixel edge); node sets
        LEFT/RIGHT/BOTTOM/TOP; equations for DOF 1 and 2.
    graindata.inp
        Legacy format of the UMAT input: '!MMM Crystal Plasticity Input File', then per grain
        'Grain: <g>: <phi1>: <PHI>: <phi2>: <size>' with Bunge angles in degrees and size = radius
        of the volume-equivalent sphere (voxel volume) in `length_unit`, as the legacy file wrote
        it in micrometres. Inclusions are left out, as in the legacy file.
    """
    if length_unit not in UNITS:
        raise ValueError(f'length_unit must be one of {list(UNITS)}, got {length_unit!r}')
    if material_per not in ('grain', 'phase'):
        raise ValueError(f"material_per must be 'grain' or 'phase', got {material_per!r}")
    m = solver_mesh(result, mesh)
    etype = element_type or _ELEMENT[m.cell_type]
    if _NODES.get(etype.upper(), m.cells.shape[1]) != m.cells.shape[1]:
        raise ValueError(f'element type {etype} does not fit {m.cells.shape[1]}-node cells')
    scale = UNITS[length_unit]
    section_line = '' if m.ndim == 3 else \
        f'{(thickness if thickness is not None else result.voxel_size_m[0]) * scale:.10g}'
    path = Path(path).with_suffix('.inp')
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='ascii', newline='\n') as f:
        meta = result.metadata
        f.write('*Heading\n')
        f.write(f'DRAGen RVE {meta.get("n_grains", result.n_grains)} grains, seed {meta.get("seed")}, '
                f'spec {meta.get("spec_sha256")}\n')
        f.write(f'** DRAGen {meta.get("dragen_version")}, commit {meta.get("git_commit")}; '
                f'lengths in {length_unit}\n')
        f.write('*Preprint, echo=NO, model=NO, history=NO, contact=NO\n')
        f.write('*Part, name=RVE\n*Node\n')
        _write_rows(f, np.arange(1, len(m.points) + 1), m.points * scale, '%.10g')
        f.write(f'*Element, type={etype}, elset=ALL\n')
        _write_rows(f, np.arange(1, len(m.cells) + 1), m.cells + 1, '%d')

        element = np.arange(1, len(m.cells) + 1)
        order = np.argsort(m.grain_ids, kind='stable')
        grains, starts = np.unique(m.grain_ids[order], return_index=True)
        for g, members in zip(grains, np.split(element[order], starts[1:])):
            _write_set(f, f'*Elset, elset=GRAIN_{g}', members)
        elem_phase = result.phase_ids[m.grain_ids - 1]
        for p in np.unique(elem_phase):
            _write_set(f, f'*Elset, elset={result.phase_names[int(p)].upper()}', element[elem_phase == p])
        elem_band = result.band_ids[m.grain_ids - 1]
        for k in np.unique(elem_band[elem_band >= 0]):
            _write_set(f, f'*Elset, elset=BAND_{k}', element[elem_band == k])
        if result.parent_grain_ids is not None:          # substructure: regions are blocks
            elem_parent = result.parent_grain_ids[m.grain_ids - 1]
            elem_packet = result.packet_ids[m.grain_ids - 1]
            for name, values in (('PARENT', elem_parent), ('PACKET', elem_packet)):
                order = np.argsort(values, kind='stable')
                ids, starts = np.unique(values[order], return_index=True)
                for v, members in zip(ids, np.split(element[order], starts[1:])):
                    if v > 0:
                        _write_set(f, f'*Elset, elset={name}_{v}', members)
        for name, nodes in m.face_nodes().items():
            _write_set(f, f'*Nset, nset={name.upper()}', nodes + 1)
        for g in grains:
            f.write(f'*Solid Section, elset=GRAIN_{g}, material={material_name(result, int(g), material_per)}\n'
                    f'{section_line},\n')
        f.write('*End Part\n**\n*Assembly, name=Assembly\n')
        f.write(f'*Instance, name={INSTANCE}, part=RVE\n*End Instance\n')
        _write_periodic(f, m, scale)
        f.write('*End Assembly\n')

    graindata = path.parent / 'graindata.inp'
    _write_graindata(result, graindata, scale)
    return {'inp': path, 'graindata': graindata}


def _write_rows(f, labels: np.ndarray, values: np.ndarray, fmt: str) -> None:
    """Lines '<label>, v1, v2, ...'."""
    values = np.asarray(values)
    np.savetxt(f, np.column_stack([labels, values]), fmt=['%d'] + [fmt] * values.shape[1], delimiter=', ')


def _write_set(f, header: str, ids: np.ndarray, per_line: int = 16) -> None:
    """Set header and its IDs, at most 16 per line (Abaqus limit)."""
    ids = np.asarray(ids, dtype=np.int64)
    f.write(header + '\n')
    full = len(ids) // per_line * per_line
    if full:
        np.savetxt(f, ids[:full].reshape(-1, per_line), fmt='%d', delimiter=', ')
    if full < len(ids):
        f.write(', '.join(str(int(i)) for i in ids[full:]) + '\n')


def _write_periodic(f, m, scale: float) -> None:
    axes = sorted(m.periodic_pairs)
    slave, master, offset = m.periodic_classes()
    origin = np.flatnonzero(np.all(np.abs(m.points) <= 1e-9 * max(m.box_size_m), axis=1))
    if len(origin):
        f.write(f'*Nset, nset=ORIGIN, instance={INSTANCE}\n{int(origin[0]) + 1}\n')
    if not axes:
        return
    names = {0: 'RP_X', 1: 'RP_Y', 2: 'RP_Z'}
    f.write('** reference nodes: u(RP_a) = jump of u across the box along axis a\n*Node\n')
    for i, a in enumerate(axes, start=1):
        pos = np.zeros(m.ndim)
        pos[a] = 1.5 * m.box_size_m[a] * scale
        f.write(f'{i}, ' + ', '.join(f'{c:.10g}' for c in pos) + '\n')
    for i, a in enumerate(axes, start=1):
        f.write(f'*Nset, nset={names[a]}\n{i}\n')
    f.write('** periodic boundary conditions: u_slave - u_master - sum_a k_a u(RP_a) = 0\n')
    lines = []
    for s, mst, k in zip(slave + 1, master + 1, offset):
        rps = [names[a] for a in range(m.ndim) if k[a]]
        for dof in range(1, m.ndim + 1):
            terms = [f'{INSTANCE}.{s}, {dof}, 1.', f'{INSTANCE}.{mst}, {dof}, -1.'] + \
                    [f'{rp}, {dof}, -1.' for rp in rps]
            lines.append('*Equation')
            lines.append(str(len(terms)))
            for j in range(0, len(terms), 4):                 # at most 4 terms per data line
                lines.append(', '.join(terms[j:j + 4]))
    f.write('\n'.join(lines) + ('\n' if lines else ''))


def _write_graindata(result, path: Path, scale: float) -> None:
    counts = np.bincount(np.asarray(result.grain_ids).ravel(), minlength=result.n_grains + 1)[1:]
    volume = counts * np.prod(result.voxel_size_m)
    radius = np.cbrt(volume * 3 / (4 * np.pi)) * scale
    inclusion = [g for g, name in result.phase_names.items() if name == 'inclusions']
    with open(path, 'w', encoding='ascii', newline='\n') as f:
        f.write('!MMM Crystal Plasticity Input File\n')
        for g in range(1, result.n_grains + 1):
            if result.phase_ids[g - 1] in inclusion:
                continue
            phi1, PHI, phi2 = result.euler_bunge_deg[g - 1]
            f.write(f'Grain: {g}: {phi1:.6f}: {PHI:.6f}: {phi2:.6f}: {radius[g - 1]:.10g}\n')
