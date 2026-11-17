"""Hand-built label fields for the special cases the mesher has to survive.

Each case carries the answers that follow from its construction (not from running any of the
mesh code), so the mesh metrics and later the topology (M1), the surface mesh (M2), the smoothing
(M3) and the volume mesh (M4) can be tested against them. All grains are unions of boxes, so the
expected values can be checked by hand.

Expected keys (grain IDs as keys; only the grains listed are checked):
  euler        Euler characteristic of every connected piece of the grain's surface in the box,
               sorted; box faces belong to the surface. Only given where the surface is a manifold.
  components   number of connected pieces of the grain's surface (pieces sharing at least an edge
               count as one)
  nonmanifold_edges     voxel edges at which the grain touches itself diagonally
  nonmanifold_vertices  voxel corners at which the grain touches itself only at the point
  patches      {(g1, g2): n} connected pieces of the boundary between two grains; pairs not
               listed must not touch
  bodies       face-connected voxel bodies per grain, as `metrics.components_per_grain` counts
               them with the case's periodicity
  topology     counts of dragen.mesh.topology with the box faces as virtual grains (M1): anchors,
               lines, patches, rings, imprinted lines, lines of higher order (four or more grains).
               Not given for cases the clean-up changes (edge_contact, vertex_contact).
  model        counts of dragen.mesh.model (M2): boundary loops over all surfaces, surfaces with
               holes, closed surfaces, surfaces of genus > 0, and the periodic slaves
               (points, curves, surfaces).

Generated RVEs complete the corpus; they are built in the tests and benchmarks from the
generator itself (tests/unit/test_mesh_corpus.py, benchmarks/mesh_baseline.py).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class CorpusCase:
    name: str
    labels: np.ndarray
    description: str
    periodic: Tuple[bool, bool, bool] = (False, False, False)
    expected: Dict[str, dict] = field(default_factory=dict)
    tags: Tuple[str, ...] = ()


def _box(shape, fill=1):
    return np.full(shape, fill, dtype=np.int32)


def _topology(anchors, lines, patches, rings=0, imprinted=0, higher=0) -> dict:
    return {'anchors': anchors, 'lines': lines, 'patches': patches, 'rings': rings,
            'imprinted_lines': imprinted, 'higher_order_lines': higher}


def _model(loops, holes=0, closed=0, genus=0, periodic=(0, 0, 0)) -> dict:
    return {'loops': loops, 'surfaces_with_holes': holes, 'closed_surfaces': closed,
            'surfaces_with_genus': genus, 'periodic_points': periodic[0],
            'periodic_curves': periodic[1], 'periodic_surfaces': periodic[2]}


def two_halves() -> CorpusCase:
    labels = _box((6, 6, 6))
    labels[3:] = 2
    return CorpusCase('two_halves', labels, 'Two grains split by one plane.',
                      expected={'model': _model(11),
                                'topology': _topology(12, 20, 11),
                                'euler': {1: [2], 2: [2]}, 'components': {1: 1, 2: 1},
                                'patches': {(1, 2): 1}},
                      tags=('plane',))


def ring_triple_line() -> CorpusCase:
    labels = _box((10, 10, 10))
    labels[:, :, 5:] = 2
    labels[3:7, 3:7, 3:7] = 3
    return CorpusCase('ring_triple_line', labels,
                      'A cube on the plane between two grains: the triple line around it is a closed '
                      'loop without quadruple points, and the 1-2 boundary is a face with a hole.',
                      expected={'model': _model(14, holes=1),
                                'topology': _topology(13, 21, 13, rings=1),
                                'euler': {1: [2], 2: [2], 3: [2]},
                                'patches': {(1, 2): 1, (1, 3): 1, (2, 3): 1}},
                      tags=('ring_triple_line', 'face_with_hole'))


def four_grain_edge() -> CorpusCase:
    labels = _box((8, 8, 8))
    labels[4:, :4] = 2
    labels[4:, 4:] = 3
    labels[:4, 4:] = 4
    return CorpusCase('four_grain_edge', labels,
                      'Four grains meet along a line: two triple paths cross. Diagonal grains touch '
                      'only along that line and share no face.',
                      expected={'model': _model(20),
                                'topology': _topology(18, 33, 20, higher=1),
                                'euler': {g: [2] for g in range(1, 5)},
                                'patches': {(1, 2): 1, (2, 3): 1, (3, 4): 1, (1, 4): 1}},
                      tags=('crossing_triple_paths',))


def eight_grain_point() -> CorpusCase:
    labels = _box((6, 6, 6))
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                labels[3 * i:3 * i + 3, 3 * j:3 * j + 3, 3 * k:3 * k + 3] = 1 + i + 2 * j + 4 * k
    patches = {(g, g ^ bit): 1 for g in range(8) for bit in (1, 2, 4) if g < g ^ bit}
    return CorpusCase('eight_grain_point', labels, 'Eight octant grains meet in one point.',
                      expected={'model': _model(36),
                                'topology': _topology(27, 54, 36, higher=6),
                                'euler': {g: [2] for g in range(1, 9)},
                                'patches': {(a + 1, b + 1): n for (a, b), n in patches.items()}},
                      tags=('checkerboard_point',))


def edge_contact() -> CorpusCase:
    labels = _box((8, 8, 4))
    labels[4:, :4] = 2
    labels[:4, 4:] = 2
    return CorpusCase('edge_contact', labels,
                      'Checkerboard of two grains: each consists of two columns that touch along '
                      'one edge line of 4 voxels.',
                      expected={'components': {1: 1, 2: 1}, 'nonmanifold_edges': {1: 4, 2: 4},
                                'nonmanifold_vertices': {1: 0, 2: 0},
                                'patches': {(1, 2): 1}, 'bodies': {1: 2, 2: 2}},
                      tags=('edge_contact', 'checkerboard'))


def vertex_contact() -> CorpusCase:
    labels = _box((6, 6, 6), fill=2)
    labels[1:3, 1:3, 1:3] = 1
    labels[3:5, 3:5, 3:5] = 1
    return CorpusCase('vertex_contact', labels,
                      'Grain 1 is two cubes touching at a single corner inside grain 2.',
                      expected={'euler': {1: [2, 2], 2: [2, 2, 2]}, 'components': {1: 2, 2: 3},
                                'nonmanifold_edges': {1: 0, 2: 0},
                                'nonmanifold_vertices': {1: 1, 2: 1},
                                'patches': {(1, 2): 2}, 'bodies': {1: 2, 2: 1}},
                      tags=('vertex_contact',))


def grain_in_grain() -> CorpusCase:
    labels = _box((8, 8, 8))
    labels[3:5, 3:5, 3:5] = 2
    return CorpusCase('grain_in_grain', labels,
                      'A grain enclosed by another: their boundary is a closed face without any '
                      'triple line.',
                      expected={'model': _model(6, closed=1),
                                'topology': _topology(8, 12, 7),
                                'euler': {1: [2, 2], 2: [2]}, 'components': {1: 2, 2: 1},
                                'patches': {(1, 2): 1}},
                      tags=('grain_in_grain', 'face_without_boundary'))


def torus() -> CorpusCase:
    labels = _box((10, 10, 6))
    labels[2:8, 2:8, 2:4] = 2
    labels[4:6, 4:6, 2:4] = 1
    return CorpusCase('torus', labels,
                      'A ring-shaped grain inside another: closed boundary of genus 1.',
                      expected={'model': _model(6, closed=1, genus=1),
                                'topology': _topology(8, 12, 7),
                                'euler': {1: [0, 2], 2: [0]}, 'components': {1: 2, 2: 1},
                                'patches': {(1, 2): 1}},
                      tags=('face_without_boundary', 'genus'))


def two_contact_patches() -> CorpusCase:
    labels = _box((10, 6, 8), fill=2)
    labels[:, :, :2] = 1
    labels[:, :, 2:4] = 3
    labels[1:3, 2:4, 2:4] = 1
    labels[7:9, 2:4, 2:4] = 1
    return CorpusCase('two_contact_patches', labels,
                      'Grain 1 reaches grain 2 through two pillars that pierce grain 3: grains 1 and 2 '
                      'touch in two separate faces, grain 3 is a slab with two holes.',
                      expected={'model': _model(22, holes=2),
                                'topology': _topology(18, 30, 18, rings=2),
                                'euler': {1: [2], 2: [2], 3: [-2]},
                                'patches': {(1, 2): 2, (1, 3): 1, (2, 3): 1}},
                      tags=('several_patches', 'face_with_hole', 'genus'))


def thin_grain() -> CorpusCase:
    labels = _box((6, 6, 5))
    labels[:, :, 2] = 2
    labels[:, :, 3:] = 3
    return CorpusCase('thin_grain', labels, 'A grain one voxel thick between two others.',
                      expected={'model': _model(16),
                                'topology': _topology(16, 28, 16),
                                'euler': {1: [2], 2: [2], 3: [2]},
                                'patches': {(1, 2): 1, (2, 3): 1}},
                      tags=('thin_grain',))


def single_voxel() -> CorpusCase:
    labels = _box((5, 5, 5))
    labels[2, 2, 2] = 2
    return CorpusCase('single_voxel', labels, 'A grain of a single voxel.',
                      expected={'model': _model(6, closed=1),
                                'topology': _topology(8, 12, 7),
                                'euler': {1: [2, 2], 2: [2]}, 'patches': {(1, 2): 1}},
                      tags=('thin_grain', 'grain_in_grain'))


def fragmented_grain(periodic: bool = False) -> CorpusCase:
    labels = _box((8, 4, 4))
    labels[2:6] = 2
    if periodic:
        return CorpusCase('periodic_slab', labels,
                          'Grain 1 crosses the periodic x boundary: one body, cut by the box into two '
                          'pieces that must mirror each other.',
                          periodic=(True, True, True),
                          expected={'model': _model(16, periodic=(13, 19, 7)),
                                    'topology': _topology(16, 28, 16),
                                    'euler': {1: [2, 2], 2: [2]}, 'components': {1: 2, 2: 1},
                                    'patches': {(1, 2): 2}, 'bodies': {1: 1, 2: 1}},
                          tags=('periodic', 'box_faces'))
    return CorpusCase('fragmented_grain', labels, 'Grain 1 consists of two separate bodies.',
                      expected={'model': _model(16),
                                'topology': _topology(16, 28, 16),
                                    'euler': {1: [2, 2], 2: [2]}, 'components': {1: 2, 2: 1},
                                'patches': {(1, 2): 2}, 'bodies': {1: 2, 2: 1}},
                      tags=('fragmented',))


def periodic_imprint() -> CorpusCase:
    labels = _box((6, 4, 4))
    labels[3:] = 2
    labels[0, 1:3, 1:3] = 3
    return CorpusCase('periodic_imprint', labels,
                      'Periodic in x only. A one-voxel-thick grain on the lower x face draws a ring on '
                      'that face; the upper x face belongs to another grain and gets the same ring '
                      'imprinted, so both faces can be meshed alike.',
                      periodic=(True, False, False),
                      expected={'model': _model(16, holes=2, periodic=(5, 5, 2)),
                                'euler': {1: [2], 2: [2], 3: [2]},
                                'patches': {(1, 2): 1, (1, 3): 1},
                                'topology': _topology(14, 22, 14, rings=2, imprinted=1)},
                      tags=('periodic', 'ring_triple_line', 'box_faces'))


def cases() -> List[CorpusCase]:
    return [two_halves(), ring_triple_line(), four_grain_edge(), eight_grain_point(), edge_contact(),
            vertex_contact(), grain_in_grain(), torus(), two_contact_patches(), thin_grain(),
            single_voxel(), fragmented_grain(), fragmented_grain(periodic=True), periodic_imprint()]
