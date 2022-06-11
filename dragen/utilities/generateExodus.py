from netCDF4 import Dataset
import pyvista as pv
import numpy as np
from dragen.utilities.InputInfo import RveInfo

class NetCDFWrapper:
    def __init__(self, file_base, num_nodes, num_elems, num_blocks, num_node_sets=None):
        Version = 2.0
        file_path = RveInfo.store_path+'/'+file_base+'.e'
        self._data = Dataset(file_path, mode='w', format='NETCDF3_64BIT', clobber=True)

        self._data.title = 'DRAGenRVE'
        self._data.version = np.float32(Version)
        self._data.api_version = np.float32(Version)
        self._data.floating_point_word_size = np.int32(8)
        self._data.maximum_name_length = np.int32(32)
        self._data.file_size = 1
        self._data.int64_status = 0

        # Create dimensions
        self._data.createDimension('len_string', 32)
        self._data.createDimension('len_name', 256)

        self._data.createDimension('num_dim', RveInfo.dimension)
        self._data.createDimension('num_nodes', num_nodes)
        self._data.createDimension('num_elem', num_elems)
        self._data.createDimension('num_el_blk', num_blocks)
        self._data.createDimension('time_step', None)
        #print(self._data.dimensions['num_el_blk1'])
        #breakpoint()


        # Create variables
        self._data.createVariable('time_whole', 'f8', 'time_step', shuffle=False)
        self._data.createVariable('coor_names', 'S1', ('num_dim', 'len_name'), shuffle=False)
        self._data.createVariable('coordx', 'f8', 'num_nodes', shuffle=False)
        self._data.createVariable('coordy', 'f8', 'num_nodes', shuffle=False)
        self._data.createVariable('coordz', 'f8', 'num_nodes', shuffle=False)
        self._data.createVariable('eb_status', 'i4', 'num_el_blk', fill_value=0, shuffle=False)
        self._data.createVariable('eb_prop1', 'i4', 'num_el_blk', shuffle=False)
        self._data.variables['eb_prop1'].setncattr('name', 'ID')
        self._data.createVariable('eb_names', 'S1', ('num_el_blk', 'len_name'), shuffle=False)

        if num_node_sets:
            self._data.createDimension('num_node_sets', num_node_sets)
            self._data.createVariable('ns_status', 'i4', 'num_node_sets', fill_value=0)
            self._data.createVariable('ns_prop1', 'i4', 'num_node_sets')
            self._data.variables['ns_prop1'].setncattr('name', 'ID')
            self._data.createVariable('ns_names', 'S1', ('num_node_sets', 'len_name'))

    def set_coord_names(self, names: list):
        """
        :param names: list of coord names
        :type: list
        """
        ndim = RveInfo.dimension
        assert len(names) == ndim, 'The length of the names array must equal the number of dimensions'
        for i in range(ndim):
            self._data.variables['coor_names'][i, 0:len(names[i])] = [c for c in names[i]]
        return

    def set_coords(self, xcoords, ycoords, zcoords):

        assert len(xcoords) == self._data.dimensions[
            'num_nodes'].size, 'Number of Y coords must be equal to numNodes'
        assert len(ycoords) == self._data.dimensions[
            'num_nodes'].size, 'Number of Y coords must be equal to numNodes'
        assert len(zcoords) == self._data.dimensions[
            'num_nodes'].size, 'Number of Z coords must be equal to numNodes'

        self._data.variables['coordx'][:] = xcoords
        self._data.variables['coordy'][:] = ycoords
        self._data.variables['coordz'][:] = zcoords
        return

    def set_elem_blk_names(self, names):
        num_el_blk = self._data.dimensions['num_el_blk'].size
        assert len(names) == num_el_blk, 'The length of the names array must be equal to the number of blocks'

        for i in range(num_el_blk):
            self._data.variables['eb_names'][i, 0:len(names[i])] = [c for c in names[i]]
        return

    def set_elem_blk_info(self, blk_id, elem_type, num_blk_elems, num_elem_nodes):

        # Find first free position in eb_status (first 0 value)
        eb_status = self._data.variables['eb_status']
        eb_status.set_auto_mask(False)
        idx = np.where(eb_status[:] == 0)[0][0]
        self._data.variables['eb_status'][idx] = 1
        self._data.variables['eb_prop1'][idx] = blk_id

        num_elem_in_blk_name = 'num_el_in_blk{}'.format(idx + 1)
        num_nodes_per_elem_name = 'num_nod_per_el{}'.format(idx + 1)

        self._data.createDimension(num_elem_in_blk_name, num_blk_elems)
        self._data.createDimension(num_nodes_per_elem_name, num_elem_nodes)

        var_name = 'connect{}'.format(idx + 1)
        self._data.createVariable(var_name, 'i4', (num_elem_in_blk_name, num_nodes_per_elem_name), shuffle=False)
        self._data.variables[var_name].elem_type = str(elem_type).upper()

        return

    def set_elem_connectivity(self, blk_id, connectivity):

        assert blk_id in self._data.variables['eb_prop1'][:], 'blk_id not in list of block ids'

        # Get idx corresponding to blk_id
        idx = np.where(self._data.variables['eb_prop1'][:] == blk_id)[0][0]

        num_elem_in_blk_name = 'num_el_in_blk{}'.format(idx + 1)
        num_nodes_per_elem_name = 'num_nod_per_el{}'.format(idx + 1)
        num_elem_in_blk = self._data.dimensions[num_elem_in_blk_name].size
        num_nodes_per_elem = self._data.dimensions[num_nodes_per_elem_name].size
        assert connectivity.size == num_elem_in_blk * num_nodes_per_elem, 'Incorrect number of nodes in connectivity'

        var_name = 'connect{}'.format(idx + 1)
        self._data.variables[var_name][:] = connectivity.reshape(num_elem_in_blk, num_nodes_per_elem)
        return

    def set_node_set_names(self, names):

        num_node_sets = self._data.dimensions['num_node_sets'].size
        assert len(names) == num_node_sets, 'The length of the names array must be equal to the number of nodesets'

        for i in range(num_node_sets):
            self._data.variables['ns_names'][i, 0:len(names[i])] = [c for c in names[i]]

        return

    def set_node_set_info(self, idx, num_node_set_nodes):

        assert idx not in self._data.variables['ns_prop1'][:], 'Nodeset id {} already in use'.format(idx)

        # Find first free position in ss_status (first 0 value)
        ns_status = self._data.variables['ns_status']
        ns_status.set_auto_mask(False)
        idx = np.where(ns_status[:] == 0)[0][0]

        self._data.variables['ns_status'][idx] = 1
        self._data.variables['ns_prop1'][idx] = idx

        num_node_ns_name = 'num_nod_ns{}'.format(idx + 1)
        node_ns_name = 'node_ns{}'.format(idx + 1)

        self._data.createDimension(num_node_ns_name, num_node_set_nodes)
        self._data.createVariable(node_ns_name, 'i4', num_node_ns_name)

        self._data.variables['ns_status'][idx] = 1
        self._data.variables['ns_prop1'][idx] = idx

        return

    def set_node_set(self, idx, node_set_nodes):

        assert idx in self._data.variables['ns_prop1'][:], 'Nodeset id {} not present'.format(idx)

        # Get idx corresponding to id
        idx = np.where(self._data.variables['ns_prop1'][:] == idx)[0][0]

        node_ns_name = 'node_ns{}'.format(idx + 1)

        self._data.variables[node_ns_name][:] = node_set_nodes

        return

    def set_element_variable_name(self, name, index):

        self._data.variables['name_elem_var'][index - 1, 0:len(name)] = [c for c in name]
        return

    def set_element_variable_number(self, number):

        self._data.createDimension('num_elem_var', number)
        self._data.createVariable('name_elem_var', 'S1', ('num_elem_var', 'len_name'), shuffle=False)

        return

    def set_element_variable_values(self, blk_id, name, step, values):

        var_names = self.get_element_variable_name()
        block_ids = self._data.variables['eb_prop1'][:]
        assert name in var_names, 'Variable {} not found in list of element variables'.format(name)
        assert blk_id in block_ids, 'Block id {} not found'.format(blk_id)

        idx = np.where(block_ids == blk_id)[0][0]
        var_idx = var_names.index(name)

        var_name = 'vals_elem_var{}eb{}'.format(var_idx + 1, idx + 1)
        num_elem_in_blk = 'num_el_in_blk{}'.format(idx + 1)

        if var_name not in self._data.variables:
            self._data.createVariable(var_name, 'f8', ('time_step', num_elem_in_blk), shuffle=False)

        self._data.variables[var_name][step - 1] = values

        return

    def get_element_variable_name(self):

        name_elem_var = self._data.variables['name_elem_var']
        name_elem_var.set_auto_mask(False)

        names = [b''.join(c).strip().decode() for c in name_elem_var[:]]
        return names

    @staticmethod
    def extract_side_nodes(grid, key='x', val=0):
        """
        :param grid: grid containing all points in model
        :param key: direction node set is vertical to
        :param val: value node plane must lie on eg. x_min=0 or x_max=10
        :return: array with all point ids in the set
        """
        assert key in ['x', 'y', 'z'], 'key must be "x", "y" or "z" but is {}'.format(key)

        pts = grid.points
        if key == 'x':
            pt_ids = np.where(pts[:, 0] == val)
        if key == 'y':
            pt_ids = np.where(pts[:, 1] == val)
        if key == 'z':
            pt_ids = np.where(pts[:, 2] == val)

        pt_ids = [pt+1 for pt in pt_ids[0]]
        return pt_ids

    @staticmethod
    def fixorder(array):
        for i in range(array.shape[0]):
            array[i, :] = array[i, [0, 1, 3, 2, 4, 5, 7, 6]]
        return array

    def put_time(self, step, value):
        self._data.variables['time_whole'][step - 1] = value
        return

    def close(self):
        self._data.close()
        return
