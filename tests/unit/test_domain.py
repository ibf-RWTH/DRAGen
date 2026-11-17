import numpy as np
import pytest

from dragen.config import BoxConfig
from dragen.core import VoxelDomain


@pytest.fixture
def cube():
    return VoxelDomain(shape=(4, 4, 4), spacing=(0.5, 0.5, 0.5), periodic=(True, True, True))


def test_from_box():
    dom = VoxelDomain.from_box(BoxConfig(size=(10, 5, 5), resolution=2, periodic=(True, True, False)))
    assert dom.shape == (20, 10, 10)
    assert dom.size == (10, 5, 5)
    assert dom.voxel_volume == 0.125
    assert dom.n_voxels == 2000
    assert dom.new_labels().shape == (20, 10, 10)


def test_wrap_periodic(cube):
    flat, inside = cube.wrap(np.array([[-1, 0, 0], [4, 5, -6], [1, 2, 3]]))
    assert inside.all()
    assert cube.unravel(flat).tolist() == [[3, 0, 0], [0, 1, 2], [1, 2, 3]]


def test_wrap_drops_points_beyond_non_periodic_axis():
    dom = VoxelDomain((4, 4, 4), (1, 1, 1), periodic=(True, True, False))
    flat, inside = dom.wrap(np.array([[-1, 0, -1], [0, 0, 4], [-1, -1, 3]]))
    assert inside.tolist() == [False, False, True]
    assert dom.unravel(flat).tolist() == [[3, 3, 3]]


def test_place_shape_across_corner(cube):
    offsets = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    coords = cube.unravel(cube.place((3, 3, 3), offsets)).tolist()
    assert coords == [[3, 3, 3], [0, 3, 3], [3, 0, 3], [3, 3, 0]]


def test_face_neighbors_periodic_and_open():
    dom = VoxelDomain((3, 3), (1, 1), periodic=(True, False))
    corner = np.ravel_multi_index((0, 0), dom.shape)
    nb = dom.face_neighbors(np.array([corner]))[0]
    # -x wraps to row 2, +x is row 1, -y is beyond the open boundary, +y is column 1
    assert dom.unravel(nb[[0, 1, 3]]).tolist() == [[2, 0], [1, 0], [0, 1]]
    assert nb[2] == -1


def test_every_voxel_has_2ndim_neighbours_in_periodic_box(cube):
    nb = cube.face_neighbors(np.arange(cube.n_voxels))
    assert (nb >= 0).all()
    # neighbourhood is symmetric: each voxel appears 6 times as somebody's neighbour
    assert np.bincount(nb.ravel(), minlength=cube.n_voxels).tolist() == [6] * cube.n_voxels


def test_voxel_centers(cube):
    assert cube.voxel_centers(np.array([0, cube.n_voxels - 1])).tolist() == [[0.25] * 3, [1.75] * 3]
