import sys
import numpy as np

class RVE_utils:
    """Common Representative Volume Element (RVE) operations."""

    def __init__(self, box_size, points_on_edge, bandwidth = 0):
        # LOGGER initialization can be added here
        self.box_size = box_size
        self.points_on_edge = points_on_edge
        self.step_size = box_size/points_on_edge
        self.step_half = self.step_size/2
        self.bandwidth = bandwidth

    def convert_volume(self, radius_a, radius_b, radius_c):
        """Compute the volume for the given radii.
        Parameters :
        radius_a : Integer, radius along x-axis
        radius_b : Integer, radius along y-axis
        radius_b : Integer, radius along z-axis
        """
        grid = np.around(np.arange(-self.box_size + self.step_half, self.box_size, self.step_size))
        mingrid = min([n for n in grid if n > 0])
        x0 = list(grid).index(mingrid)
        grainx, grainy, grainz = np.meshgrid(grid, grid, grid)
        A0 = (1. / radius_a) ** 2
        B0 = (1. / radius_b) ** 2
        C0 = (1. / radius_c) ** 2
        r = np.sqrt(A0 * (grainx-grid[x0]) ** 2 + B0 * (grainy-grid[x0]) ** 2 + C0 * (grainz-grid[x0]) ** 2)
        inside = r <= 1
        # LOGGER : print the returned volume for (radius_a, radius_b, radius_c)
        return len(grainx[inside])

    def periodicity_RSA(self, coordinate, points):
        """Compute the list of periodic coordinates for the given grain coordinates and list of points.
        Parameters :
        coordinate : List, list of grain coordinates
        points : Array, list of points in interval [step_half-box_size, box_size*2) with step_size spacing
        """
        points = list(points)
        periodic_coord = []
        for j, point in enumerate(coordinate):
            if point > self.box_size:
                idx = points.index(point)
                periodic_point = points[idx - self.points_on_edge]

            elif point < self.step_half:
                idx = points.index(point)
                periodic_point = points[idx + self.points_on_edge]

            else:
                periodic_point = point
            periodic_coord.append(periodic_point)

        return periodic_coord

    def periodicity_DT(self, coordinate, points):
        """Computes the list of periodic points and periodic identifiers for the given coordinates
        and list of points.
        Parameters :
        coordinate : List, list of grain coordinates
        points : Array, list of points in interval [step_half-box_size, box_size*2) with step_size spacing
        """
        points = list(points)
        periodic_pt, periodic_identifier_list = ([] for i in range(2))
        for j, point in enumerate(coordinate):
            if point > self.box_size:
                idx = points.index(point)
                periodic_point = points[idx - self.points_on_edge]
                periodic_pt.append(periodic_point)
                periodic_identifier_list.append(1)

            elif point < self.step_half:
                idx = points.index(point)
                periodic_point = points[idx + self.points_on_edge]
                periodic_pt.append(periodic_point)
                periodic_identifier_list.append(-1)

            else:
                periodic_pt.append(point)
                periodic_identifier_list.append(0)

        return periodic_pt, periodic_identifier_list

    def band_generator(self, xyz_grid, plane = 'xy'):
        """Creates a band of given bandwidth for given points in interval [step_half, box_size)
        with step_size spacing along the axis.
        Parameters :
        xyz_grid : Array, list of points in interval [step_half, box_size) with step_size spacing
        plane : String, default is 'xy'
        """
        band_half = self.bandwidth/2
        randidx = int(np.random.rand()*len(xyz_grid))
        band_center = xyz_grid[randidx]
        x, y, z = np.meshgrid(xyz_grid, xyz_grid, xyz_grid)
        if plane == 'xy':
            r = z
        elif plane == 'yz':
            r = x
        elif plane == 'xz':
            r = y
        else:
            print('Error: plane must be defined as xy, yz or xz! Default: xy') # LOGGER
            sys.exit(1)
        left_bound = r >= band_center - band_half
        right_bound = r <= band_center + band_half
        # LOGGER : Print bandwidth, bandcenter-bandhalf, bandcenter+bandhalf, left, right
        left = set([a for a in zip(x[right_bound], y[right_bound], z[right_bound])])
        right = set([a for a in zip(x[left_bound], y[left_bound], z[left_bound])])

        return left.intersection(right)
