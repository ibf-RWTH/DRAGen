import numpy as np
import pandas as pd

from dragen.utilities.generateExodus import NetCDFWrapper
from dragen.generation.PvGridGeneration import MeshingHelper
from dragen.utilities.InputInfo import RveInfo


class MooseMesher(MeshingHelper):

    def __init__(self, rve_shape: tuple, rve: pd.DataFrame, grains_df: pd.DataFrame):
        super().__init__(rve_shape, rve, grains_df)

    def run(self):
        grid = self.gen_blocks()
        grid = self.gen_grains(grid)
        grid = self.smoothen_mesh(grid)

        points = grid.points
        nNodes = grid.n_points
        nElems = grid.n_cells
        nBlocks = len(np.unique(grid.cell_data['GrainID']))
        blockIds = np.unique(grid.cell_data['GrainID'])
        # NetCDF4 starts counting at 1, cell id of 0 leads to errors
        grainIds = ['grain-{}'.format(i + 1) for i in range(nBlocks)]

        assert RveInfo.element_type == 'HEX8', 'only Hexagonal elements supported for Moose'

        nNodeSets = 6
        n_elem_nodes = 8

        sideSetDict = {'bottom': [0, -1, 0], 'top': [0, 1, 0],
                       'left': [-1, 0, 0], 'right': [1, 0, 0],
                       'front': [0, 0, 1], 'back': [0, 0, -1]}

        nodeSetDict = {'bottom': {'y': min(grid.points[:, 1])}, 'top': {'y': max(grid.points[:, 1])},
                       'left': {'x': min(grid.points[:, 0])}, 'right': {'x': max(grid.points[:, 0])},
                       'front': {'z': max(grid.points[:, 2])}, 'back': {'z': min(grid.points[:, 2])}}

        exoFile = NetCDFWrapper('DRAGen_RVE', num_nodes=nNodes, num_elems=nElems, num_blocks=nBlocks,
                                num_node_sets=nNodeSets)

        exoFile.set_coord_names(['x', 'y', 'z'])
        xcoords = np.array(points[:, 0])
        ycoords = np.array(points[:, 1])
        zcoords = np.array(points[:, 2])
        exoFile.set_coords(xcoords, ycoords, zcoords)
        exoFile.set_elem_blk_names(grainIds)
        for idx in blockIds:
            subgrid = grid.extract_cells(np.where(grid.cell_data['GrainID'] == idx))
            n_elems_block_i = subgrid.n_cells

            connectivity_temp = subgrid.cell_connectivity
            ori_connectivity_pts = subgrid.point_data['vtkOriginalPointIds']
            ori_connectivity_pts = [pt + 1 for pt in ori_connectivity_pts]
            connectivity = np.array([ori_connectivity_pts[pt] for pt in connectivity_temp])
            connectivity = connectivity.reshape(-1, n_elem_nodes)
            connectivity = exoFile.fixorder(connectivity)
            exoFile.set_elem_blk_info(idx, 'Hex8', n_elems_block_i, n_elem_nodes)
            exoFile.set_elem_connectivity(idx, connectivity)

        if nNodeSets:
            exoFile.set_node_set_names(list(sideSetDict.keys()))
            for i, key in enumerate(nodeSetDict):
                temp_dict = nodeSetDict[key]
                direction_key = list(temp_dict.keys())[0]
                val = temp_dict[direction_key]
                pt_ids = exoFile.extract_side_nodes(grid, direction_key, val)

                exoFile.set_node_set_info(idx=i, num_node_set_nodes=len(pt_ids))
                exoFile.set_node_set(idx=i, node_set_nodes=pt_ids)
        timestep = 1
        time = 0
        exoFile.put_time(timestep, time)

        exoFile.set_element_variable_number(1)
        exoFile.set_element_variable_name('phaseID', 1)
        for i in range(nBlocks):
            idx = np.where(grid.cell_data['GrainID'] == i+1)[0]
            id = grid.extract_cells(idx).cell_data['phaseID']
            exoFile.set_element_variable_values(i+1, 'phaseID', timestep, id)

        exoFile.close()
