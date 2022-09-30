import numpy as np
import pandas as pd
import pyvista as pv
import datetime
import os
from dragen.generation.PvGridGeneration import MeshingHelper
from dragen.utilities.InputInfo import RveInfo

class AbaqusMesher(MeshingHelper):

    def __init__(self, rve_shape: tuple, rve: pd.DataFrame, grains_df: pd.DataFrame):
        super().__init__(rve_shape, rve, grains_df)

    def make_assembly(self) -> None:

        """simple function to write the assembly definition in the input file"""

        f = open(RveInfo.store_path + '/DRAGen_RVE.inp', 'a')
        f.write('*End Part\n')
        f.write('**\n')
        f.write('** ASSEMBLY\n')
        f.write('**\n')
        f.write('*Assembly, name=Assembly\n')
        f.write('**\n')
        f.write('*Instance, name=Part-1-1, part=Part-1\n')
        f.write('*End Instance\n')
        f.write('**\n')
        if RveInfo.submodel_flag:
            f.write('*Include, Input=HullPointSet.inp\n')
        if RveInfo.pbc_flag:
            f.write('*Include, Input=Nsets.inp\n')
            f.write('*Include, input=LeftToRight.inp\n')
            f.write('*Include, input=BottomToTop.inp\n')
            f.write('*Include, input=FrontToRear.inp\n')
            f.write('*Include, input=Edges.inp\n')
            f.write('*Include, input=Corners.inp\n')
            f.write('*Include, input=VerticeSets.inp\n')
        f.write('*End Assembly\n')
        f.write('** INCLUDE MATERIAL FILE **\n')
        f.write('*Include, input=Materials.inp\n')
        f.write('** INCLUDE STEP FILE **\n')
        f.write('*Include, input=Step.inp\n')
        f.close()

    def submodelSet(self, grid_hull_df: pd.DataFrame) -> None:
        OutPutFile = open(RveInfo.store_path + '/HullPointSet.inp', 'w')
        grid_hull_df.sort_values(by=['x', 'y', 'z'], inplace=True)
        grid_hull_df.index.rename('pointNumber', inplace=True)
        grid_hull_df = grid_hull_df.reset_index()
        OutPutFile.write('*Nset, nset=SET-Hull, instance=PART-1-1\n')
        for i in grid_hull_df.index:
            OutPutFile.write(' {},'.format(int(grid_hull_df.loc[i]['pointNumber'] + 1)))
            if (i+1) % 16 == 0:
                OutPutFile.write('\n')
        OutPutFile.write('\n')
        OutPutFile.write('**\n')
        OutPutFile.write('*Submodel, type=NODE, exteriorTolerance=0.05\n')
        OutPutFile.write('SET-HULL, \n')
        OutPutFile.close()

    def pbc(self, rve: pv.UnstructuredGrid, grid_hull_df: pd.DataFrame) -> None:

        """function to define the periodic boundary conditions
        if errors appear or equations are wrong check ppt presentation from ICAMS
        included in the docs folder called PBC_docs"""

        min_x = min(rve.points[:, 0])
        min_y = min(rve.points[:, 1])
        min_z = min(rve.points[:, 2])
        max_x = max(rve.points[:, 0])
        max_y = max(rve.points[:, 1])
        max_z = max(rve.points[:, 2])

        numberofgrains = self.n_grains
        ########## write Equation - sets ##########
        grid_hull_df.sort_values(by=['x', 'y', 'z'], inplace=True)
        grid_hull_df.index.rename('pointNumber', inplace=True)
        grid_hull_df = grid_hull_df.reset_index()
        grid_hull_df.index.rename('Eqn-Set', inplace=True)
        grid_hull_df = grid_hull_df.reset_index()

        ########## Define Corner Sets ###########
        corner_df = grid_hull_df.loc[((grid_hull_df['x'] == max_x) | (grid_hull_df['x'] == min_x)) &
                                     ((grid_hull_df['y'] == max_y) | (grid_hull_df['y'] == min_y)) &
                                     ((grid_hull_df['z'] == max_z) | (grid_hull_df['z'] == min_z))]

        V1_df = corner_df.loc[(corner_df['x'] == min_x) & (corner_df['y'] == min_y) & (corner_df['z'] == max_z)]
        V1 = V1_df['pointNumber'].values[0]
        V1Eqn = V1_df['Eqn-Set'].values[0]

        V2_df = corner_df.loc[(corner_df['x'] == max_x) & (corner_df['y'] == min_y) & (corner_df['z'] == max_z)]
        V2 = V2_df['pointNumber'].values[0]
        V2Eqn = V2_df['Eqn-Set'].values[0]

        V3_df = corner_df.loc[(corner_df['x'] == max_x) & (corner_df['y'] == max_y) & (corner_df['z'] == max_z)]
        V3 = V3_df['pointNumber'].values[0]
        V3Eqn = V3_df['Eqn-Set'].values[0]

        V4_df = corner_df.loc[(corner_df['x'] == min_x) & (corner_df['y'] == max_y) & (corner_df['z'] == max_z)]
        V4 = V4_df['pointNumber'].values[0]
        V4Eqn = V4_df['Eqn-Set'].values[0]

        H1_df = corner_df.loc[(corner_df['x'] == min_x) & (corner_df['y'] == min_y) & (corner_df['z'] == min_z)]
        H1 = H1_df['pointNumber'].values[0]
        H1Eqn = H1_df['Eqn-Set'].values[0]

        H2_df = corner_df.loc[(corner_df['x'] == max_x) & (corner_df['y'] == min_y) & (corner_df['z'] == min_z)]
        H2 = H2_df['pointNumber'].values[0]
        H2Eqn = H2_df['Eqn-Set'].values[0]

        H3_df = corner_df.loc[(corner_df['x'] == max_x) & (corner_df['y'] == max_y) & (corner_df['z'] == min_z)]
        H3 = H3_df['pointNumber'].values[0]
        H3Eqn = H3_df['Eqn-Set'].values[0]

        H4_df = corner_df.loc[(corner_df['x'] == min_x) & (corner_df['y'] == max_y) & (corner_df['z'] == min_z)]
        H4 = H4_df['pointNumber'].values[0]
        H4Eqn = H4_df['Eqn-Set'].values[0]

        ############ Define Edge Sets ###############
        edges_df = grid_hull_df.loc[(((grid_hull_df['x'] == max_x) | (grid_hull_df['x'] == min_x)) &
                                     ((grid_hull_df['y'] == max_y) | (grid_hull_df['y'] == min_y)) &
                                     ((grid_hull_df['z'] != max_z) & (grid_hull_df['z'] != min_z))) |

                                    (((grid_hull_df['x'] == max_x) | (grid_hull_df['x'] == min_x)) &
                                     ((grid_hull_df['y'] != max_y) & (grid_hull_df['y'] != min_y)) &
                                     ((grid_hull_df['z'] == max_z) | (grid_hull_df['z'] == min_z))) |

                                    (((grid_hull_df['x'] != max_x) & (grid_hull_df['x'] != min_x)) &
                                     ((grid_hull_df['y'] == max_y) | (grid_hull_df['y'] == min_y)) &
                                     ((grid_hull_df['z'] == max_z) | (grid_hull_df['z'] == min_z)))]
        # edges_df.sort_values(by=['x', 'y', 'z'], inplace=True)

        # Top front Edge
        E_T1 = edges_df.loc[(edges_df['y'] == max_y) & (edges_df['z'] == max_z)]['Eqn-Set'].to_list()

        # Top right Edge
        E_T2 = edges_df.loc[(edges_df['x'] == max_x) & (edges_df['y'] == max_y)]['Eqn-Set'].to_list()

        # Top back Edge
        E_T3 = edges_df.loc[(edges_df['y'] == max_y) & (edges_df['z'] == min_z)]['Eqn-Set'].to_list()

        # Top left Edge
        E_T4 = edges_df.loc[(edges_df['x'] == min_x) & (edges_df['y'] == max_y)]['Eqn-Set'].to_list()

        # bottm front edge
        E_B1 = edges_df.loc[(edges_df['y'] == min_y) & (edges_df['z'] == max_z)]['Eqn-Set'].to_list()

        # bottm right edge
        E_B2 = edges_df.loc[(edges_df['x'] == max_x) & (edges_df['y'] == min_y)]['Eqn-Set'].to_list()

        # bottm back edge
        E_B3 = edges_df.loc[(edges_df['y'] == min_y) & (edges_df['z'] == min_z)]['Eqn-Set'].to_list()

        # bottm left edge
        E_B4 = edges_df.loc[(edges_df['x'] == min_x) & (edges_df['y'] == min_y)]['Eqn-Set'].to_list()

        # left front edge
        E_M1 = edges_df.loc[(edges_df['x'] == min_x) & (edges_df['z'] == max_z)]['Eqn-Set'].to_list()

        # right front edge
        E_M2 = edges_df.loc[(edges_df['x'] == max_x) & (edges_df['z'] == max_z)]['Eqn-Set'].to_list()

        # left rear edge
        E_M4 = edges_df.loc[(edges_df['x'] == min_x) & (edges_df['z'] == min_z)]['Eqn-Set'].to_list()

        # right rear edge
        E_M3 = edges_df.loc[(edges_df['x'] == max_x) & (edges_df['z'] == min_z)]['Eqn-Set'].to_list()

        ######### Define Surface Sets #############
        faces_df = grid_hull_df.loc[(((grid_hull_df['x'] == max_x) | (grid_hull_df['x'] == min_x)) &
                                     ((grid_hull_df['y'] != max_y) & (grid_hull_df['y'] != min_y)) &
                                     ((grid_hull_df['z'] != max_z) & (grid_hull_df['z'] != min_z))) |

                                    (((grid_hull_df['x'] != max_x) & (grid_hull_df['x'] != min_x)) &
                                     ((grid_hull_df['y'] != max_y) & (grid_hull_df['y'] != min_y)) &
                                     ((grid_hull_df['z'] == max_z) | (grid_hull_df['z'] == min_z))) |

                                    (((grid_hull_df['x'] != max_x) & (grid_hull_df['x'] != min_x)) &
                                     ((grid_hull_df['y'] == max_y) | (grid_hull_df['y'] == min_y)) &
                                     ((grid_hull_df['z'] != max_z) & (grid_hull_df['z'] != min_z)))]

        # left set
        LeftSet = faces_df.loc[faces_df['x'] == min_x]['Eqn-Set'].to_list()
        # right set
        RightSet = faces_df.loc[faces_df['x'] == max_x]['Eqn-Set'].to_list()
        # bottom set
        BottomSet = faces_df.loc[faces_df['y'] == min_y]['Eqn-Set'].to_list()
        # top set
        TopSet = faces_df.loc[faces_df['y'] == max_y]['Eqn-Set'].to_list()
        # front set
        RearSet = faces_df.loc[faces_df['z'] == min_z]['Eqn-Set'].to_list()
        # rear set
        FrontSet = faces_df.loc[faces_df['z'] == max_z]['Eqn-Set'].to_list()

        OutPutFile = open(RveInfo.store_path + '/Nsets.inp', 'w')
        for i in grid_hull_df.index:
            OutPutFile.write('*Nset, nset=Eqn-Set-{}, instance=PART-1-1\n'.format(i + 1))
            OutPutFile.write(' {},\n'.format(int(grid_hull_df.loc[i]['pointNumber'] + 1)))
        OutPutFile.close()

        ############### Define Equations ###################################
        OutPutFile = open(RveInfo.store_path + '/LeftToRight.inp', 'w')

        OutPutFile.write('**** X-DIR \n')
        for i in range(len(LeftSet)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(RightSet[i] + 1) + ',1, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(LeftSet[i] + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')

        OutPutFile.write('**** \n')
        OutPutFile.write('**** Y-DIR \n')
        for i in range(len(LeftSet)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(RightSet[i] + 1) + ',2, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(LeftSet[i] + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')

        OutPutFile.write('**** \n')
        OutPutFile.write('**** Z-DIR \n')
        for i in range(len(LeftSet)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(RightSet[i] + 1) + ',3, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(LeftSet[i] + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')
        OutPutFile.close()

        OutPutFile = open(RveInfo.store_path + '/BottomToTop.inp', 'w')

        OutPutFile.write('**** X-DIR \n')
        for i in range(len(BottomSet)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(BottomSet[i] + 1) + ',1,1 \n')
            OutPutFile.write('Eqn-Set-' + str(TopSet[i] + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',1,1 \n')

        OutPutFile.write('**** \n')
        OutPutFile.write('**** Y-DIR \n')
        for i in range(len(BottomSet)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(BottomSet[i] + 1) + ',2, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(TopSet[i] + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',2, 1 \n')

        OutPutFile.write('**** \n')
        OutPutFile.write('**** Z-DIR \n')
        for i in range(len(BottomSet)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(BottomSet[i] + 1) + ',3, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(TopSet[i] + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',3, 1 \n')
        OutPutFile.close()

        OutPutFile = open(RveInfo.store_path + '/FrontToRear.inp', 'w')

        OutPutFile.write('**** X-DIR \n')
        for i in range(len(RearSet)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(RearSet[i] + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(FrontSet[i] + 1) + ',1,1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',1,1 \n')

        OutPutFile.write('**** \n')
        OutPutFile.write('**** Y-DIR \n')
        for i in range(len(RearSet)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(RearSet[i] + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(FrontSet[i] + 1) + ',2,1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',2,1 \n')

        OutPutFile.write('**** \n')
        OutPutFile.write('**** Z-DIR \n')
        for i in range(len(RearSet)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(RearSet[i] + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(FrontSet[i] + 1) + ',3,1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',3,1 \n')
        OutPutFile.close()

        OutPutFile = open(RveInfo.store_path + '/Edges.inp', 'w')

        # Edges in x-y Plane
        # right top edge to left top edge
        OutPutFile.write('**** X-DIR \n')
        for i in range(len(E_T2)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T2[i] + 1) + ',1, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T4[i] + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')

        OutPutFile.write('**** Y-DIR \n')
        for i in range(len(E_T2)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T2[i] + 1) + ',2, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T4[i] + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')

        OutPutFile.write('**** Z-DIR \n')
        for i in range(len(E_T2)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T2[i] + 1) + ',3, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T4[i] + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')

        # right bottom edge to left bottom edge
        OutPutFile.write('**** X-DIR \n')
        for i in range(len(E_B2)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B2[i] + 1) + ',1, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B4[i] + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')

        OutPutFile.write('**** Y-DIR \n')
        for i in range(len(E_B2)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B2[i] + 1) + ',2, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B4[i] + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')

        OutPutFile.write('**** Z-DIR \n')
        for i in range(len(E_B2)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B2[i] + 1) + ',3, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B4[i] + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')

        # left top edge to left bottom edge
        OutPutFile.write('**** X-DIR \n')
        for i in range(len(E_T4)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T4[i] + 1) + ',1, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B4[i] + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')

        OutPutFile.write('**** Y-DIR \n')
        for i in range(len(E_T4)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T4[i] + 1) + ',2, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B4[i] + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')

        OutPutFile.write('**** Z-DIR \n')
        for i in range(len(E_T4)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T4[i] + 1) + ',3, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B4[i] + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')

        # Edges in y-z Plane
        # top back edge to top front edge
        OutPutFile.write('**** X-DIR \n')
        for i in range(len(E_T3)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T3[i] + 1) + ',1, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T1[i] + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')

        OutPutFile.write('**** Y-DIR \n')
        for i in range(len(E_T3)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T3[i] + 1) + ',2, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T1[i] + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')

        OutPutFile.write('**** Z-DIR \n')
        for i in range(len(E_T3)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T3[i] + 1) + ',3, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T1[i] + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')

        # Botom back edge to bottom front edge
        OutPutFile.write('**** X-DIR \n')
        for i in range(len(E_B3)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B3[i] + 1) + ',1, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B1[i] + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')

        OutPutFile.write('**** Y-DIR \n')
        for i in range(len(E_B3)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B3[i] + 1) + ',2, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B1[i] + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')

        OutPutFile.write('**** Z-DIR \n')
        for i in range(len(E_B3)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B3[i] + 1) + ',3, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B1[i] + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')

        # top front edge to bottom front edge
        OutPutFile.write('**** X-DIR \n')
        for i in range(len(E_T1)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T1[i] + 1) + ',1, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B1[i] + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')

        OutPutFile.write('**** Y-DIR \n')
        for i in range(len(E_T1)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T1[i] + 1) + ',2, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B1[i] + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')

        OutPutFile.write('**** Z-DIR \n')
        for i in range(len(E_T1)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_T1[i] + 1) + ',3, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_B1[i] + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')

        # Edges in x-z Plane
        # rear right edge to rear left edge
        OutPutFile.write('**** X-DIR \n')
        for i in range(len(E_M3)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M3[i] + 1) + ',1, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M4[i] + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')

        OutPutFile.write('**** Y-DIR \n')
        for i in range(len(E_M3)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M3[i] + 1) + ',2, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M4[i] + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')

        OutPutFile.write('**** Z-DIR \n')
        for i in range(len(E_M3)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M3[i] + 1) + ',3, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M4[i] + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')

        # front right edge to front left edge
        OutPutFile.write('**** X-DIR \n')
        for i in range(len(E_M2)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M2[i] + 1) + ',1, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M1[i] + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')

        OutPutFile.write('**** Y-DIR \n')
        for i in range(len(E_M2)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M2[i] + 1) + ',2, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M1[i] + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')

        OutPutFile.write('**** Z-DIR \n')
        for i in range(len(E_M2)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M2[i] + 1) + ',3, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M1[i] + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')

        # top front edge to bottom front edge
        OutPutFile.write('**** X-DIR \n')
        for i in range(len(E_M4)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M4[i] + 1) + ',1, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M1[i] + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',1,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')

        OutPutFile.write('**** Y-DIR \n')
        for i in range(len(E_M4)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M4[i] + 1) + ',2, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M1[i] + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',2,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')

        OutPutFile.write('**** Z-DIR \n')
        for i in range(len(E_M4)):
            # print item
            OutPutFile.write('*Equation \n')
            OutPutFile.write('4 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M4[i] + 1) + ',3, 1 \n')
            OutPutFile.write('Eqn-Set-' + str(E_M1[i] + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',3,-1 \n')
            OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')
        OutPutFile.close()

        OutPutFile = open(RveInfo.store_path + '/Corners.inp', 'w')

        # V3 zu V4
        OutPutFile.write('**** X-DIR \n')
        OutPutFile.write('*Equation \n')
        OutPutFile.write('4 \n')
        OutPutFile.write('Eqn-Set-' + str(V3Eqn + 1) + ',1, 1 \n')
        OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',1,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',1,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')
        OutPutFile.write('**** y-DIR \n')
        OutPutFile.write('*Equation \n')
        OutPutFile.write('4 \n')
        OutPutFile.write('Eqn-Set-' + str(V3Eqn + 1) + ',2, 1 \n')
        OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',2,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',2,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')
        OutPutFile.write('**** z-DIR \n')
        OutPutFile.write('*Equation \n')
        OutPutFile.write('4 \n')
        OutPutFile.write('Eqn-Set-' + str(V3Eqn + 1) + ',3, 1 \n')
        OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',3,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',3,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')

        # H4 zu V4
        OutPutFile.write('**** X-DIR \n')
        OutPutFile.write('*Equation \n')
        OutPutFile.write('4 \n')
        OutPutFile.write('Eqn-Set-' + str(H4Eqn + 1) + ',1, 1 \n')
        OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',1,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',1,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')
        OutPutFile.write('**** y-DIR \n')
        OutPutFile.write('*Equation \n')
        OutPutFile.write('4 \n')
        OutPutFile.write('Eqn-Set-' + str(H4Eqn + 1) + ',2, 1 \n')
        OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',2,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',2,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')
        OutPutFile.write('**** z-DIR \n')
        OutPutFile.write('*Equation \n')
        OutPutFile.write('4 \n')
        OutPutFile.write('Eqn-Set-' + str(H4Eqn + 1) + ',3, 1 \n')
        OutPutFile.write('Eqn-Set-' + str(V4Eqn + 1) + ',3,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',3,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')

        # H3 zu V3
        OutPutFile.write('**** X-DIR \n')
        OutPutFile.write('*Equation \n')
        OutPutFile.write('4 \n')
        OutPutFile.write('Eqn-Set-' + str(H3Eqn + 1) + ',1, 1 \n')
        OutPutFile.write('Eqn-Set-' + str(V3Eqn + 1) + ',1,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',1,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')
        OutPutFile.write('**** y-DIR \n')
        OutPutFile.write('*Equation \n')
        OutPutFile.write('4 \n')
        OutPutFile.write('Eqn-Set-' + str(H3Eqn + 1) + ',2, 1 \n')
        OutPutFile.write('Eqn-Set-' + str(V3Eqn + 1) + ',2,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',2,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')
        OutPutFile.write('**** z-DIR \n')
        OutPutFile.write('*Equation \n')
        OutPutFile.write('4 \n')
        OutPutFile.write('Eqn-Set-' + str(H3Eqn + 1) + ',3, 1 \n')
        OutPutFile.write('Eqn-Set-' + str(V3Eqn + 1) + ',3,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',3,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')

        # H2 zu V2
        OutPutFile.write('**** X-DIR \n')
        OutPutFile.write('*Equation \n')
        OutPutFile.write('4 \n')
        OutPutFile.write('Eqn-Set-' + str(H2Eqn + 1) + ',1, 1 \n')
        OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',1,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',1,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',1, 1 \n')
        OutPutFile.write('**** y-DIR \n')
        OutPutFile.write('*Equation \n')
        OutPutFile.write('4 \n')
        OutPutFile.write('Eqn-Set-' + str(H2Eqn + 1) + ',2, 1 \n')
        OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',2,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',2,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',2, 1 \n')
        OutPutFile.write('**** z-DIR \n')
        OutPutFile.write('*Equation \n')
        OutPutFile.write('4 \n')
        OutPutFile.write('Eqn-Set-' + str(H2Eqn + 1) + ',3, 1 \n')
        OutPutFile.write('Eqn-Set-' + str(V2Eqn + 1) + ',3,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(H1Eqn + 1) + ',3,-1 \n')
        OutPutFile.write('Eqn-Set-' + str(V1Eqn + 1) + ',3, 1 \n')
        OutPutFile.close()

        OutPutFile = open(RveInfo.store_path + '/VerticeSets.inp', 'w')
        OutPutFile.write('*Nset, nset=V1, instance=PART-1-1\n')
        OutPutFile.write(' {},\n'.format(V1 + 1))
        OutPutFile.write('*Nset, nset=V2, instance=PART-1-1\n')
        OutPutFile.write(' {},\n'.format(V2 + 1))
        OutPutFile.write('*Nset, nset=V3, instance=PART-1-1\n')
        OutPutFile.write(' {},\n'.format(V3 + 1))
        OutPutFile.write('*Nset, nset=V4, instance=PART-1-1\n')
        OutPutFile.write(' {},\n'.format(V4 + 1))
        OutPutFile.write('*Nset, nset=H1, instance=PART-1-1\n')
        OutPutFile.write(' {},\n'.format(H1 + 1))
        OutPutFile.write('*Nset, nset=H2, instance=PART-1-1\n')
        OutPutFile.write(' {},\n'.format(H2 + 1))
        OutPutFile.write('*Nset, nset=H3, instance=PART-1-1\n')
        OutPutFile.write(' {},\n'.format(H3 + 1))
        OutPutFile.write('*Nset, nset=H4, instance=PART-1-1\n')
        OutPutFile.write(' {},\n'.format(H4 + 1))
        OutPutFile.close()

    def write_material_def(self) -> None:

        """simple function to write material definition in Input file
        needs to be adjusted for multiple phases"""
        phase1_idx = 0
        phase2_idx = 0
        phase3_idx = 0
        phase4_idx = 0
        numberofgrains = self.n_grains

        phase = [self.rve.loc[self.rve['GrainID'] == i].phaseID.values[0] for i in range(1, numberofgrains+1)]
        f = open(RveInfo.store_path + '/Materials.inp', 'w+')  # open in write mode to overwrite old files in case ther are any
        f.write('** MATERIALS\n')
        f.write('**\n')
        f.close()
        f = open(RveInfo.store_path + '/Materials.inp', 'a')

        for i in range(numberofgrains):
            ngrain = i+1
            if phase[i] == 1:
                phase1_idx += 1
                f.write('*Material, name=Ferrite_{}\n'.format(phase1_idx))
                f.write('*Depvar\n')
                f.write('    176,\n')
                f.write('*User Material, constants=2\n')
                f.write('{}.,3.\n'.format(phase1_idx))
            elif phase[i] == 2:
                if not RveInfo.phase2iso_flag:
                    phase2_idx += 1
                    f.write('*Material, name=Martensite_{}\n'.format(phase2_idx))
                    f.write('*Depvar\n')
                    f.write('    176,\n')
                    f.write('*User Material, constants=2\n')
                    f.write('{}.,4.\n'.format(ngrain))
            elif phase[i] == 3:
                if not RveInfo.phase2iso_flag:
                    phase3_idx += 1
                    f.write('*Material, name=Pearlite_{}\n'.format(phase3_idx))
                    f.write('*Depvar\n')
                    f.write('    176,\n')
                    f.write('*User Material, constants=2\n')
                    f.write('{}.,4.\n'.format(ngrain))
            elif phase[i] == 4:
                if not RveInfo.phase2iso_flag:
                    phase4_idx += 1
                    f.write('*Material, name=Bainite_{}\n'.format(phase4_idx))
                    f.write('*Depvar\n')
                    f.write('    176,\n')
                    f.write('*User Material, constants=2\n')
                    f.write('{}.,4.\n'.format(ngrain))

        if RveInfo.phase2iso_flag and RveInfo.phase_ratio[2] > 0:
            f.write('**\n')
            f.write('*Include, Input=Martensite.inp\n')

        if RveInfo.phase2iso_flag and RveInfo.phase_ratio[3] > 0:
            f.write('**\n')
            f.write('*Material, name=Pearlite\n')
            f.write('*Elastic\n')
            f.write('210000, 0.3\n')
            f.write('**')
        if RveInfo.phase2iso_flag and RveInfo.phase_ratio[4] > 0:
            f.write('**\n')
            f.write('*Material, name=Bainite\n')
            f.write('*Elastic\n')
            f.write('210000, 0.3\n')
            f.write('**')
        f.close()

    def write_submodel_step_def(self) -> None:

        """simple function to write step definition
        variables should be introduced to give the user an option
        to modify amplidtude, and other parameters"""

        f = open(RveInfo.store_path + '/Step.inp', 'w+')
        f.write('**\n')
        f.write('** ----------------------------------------------------------------\n')
        f.write('**\n')
        f.write('** STEP: Step-1\n')
        f.write('**\n')
        f.write('*Step, name=Step-1, nlgeom=YES, inc=10000000, solver=ITERATIVE\n')
        f.write('*Static\n')
        f.write('** Step time needs to be adjusted to global .odb\n')
        f.write('0.001, 1., 1.05e-15, 0.25\n')
        f.write('**\n')
        f.write('** CONTROLS\n')
        f.write('**\n')
        f.write('*Controls, reset\n')
        f.write('*CONTROLS, PARAMETER=TIME INCREMENTATION\n')
        f.write('35, 50, 9, 50, 28, 5, 12, 45\n')
        f.write('**\n')
        f.write('*CONTROLS, PARAMETERS=LINE SEARCH\n')
        f.write('10\n')
        f.write('*SOLVER CONTROL\n')
        f.write('1e-5,200,\n')
        f.write('**\n')
        f.write('** BOUNDARY CONDITIONS\n')
        f.write('**\n')
        f.write('** Name: Sub-BC-1 Type: Submodel\n')
        f.write('*Boundary, submodel, step=1\n')
        f.write('Set-Hull, 1, 1\n')
        f.write('Set-Hull, 2, 2\n')
        f.write('Set-Hull, 3, 3\n')
        f.write('**\n')
        f.write('** OUTPUT REQUESTS\n')
        f.write('**\n')
        f.write('*Restart, write, frequency=0\n')
        f.write('**\n')
        f.write('** FIELD OUTPUT: F-Output-1\n')
        f.write('**\n')
        f.write('*Output, field, frequency=10\n')
        f.write('*Node Output\n')
        f.write('CF, COORD, RF, U\n')
        f.write('*Element Output, directions=YES\n')
        f.write('EVOL, LE, PE, PEEQ, PEMAG, S, SDV\n')
        f.write('*Contact Output\n')
        f.write('CDISP, CSTRESS\n')
        f.write('**\n')
        f.write('** HISTORY OUTPUT: H-Output-1\n')
        f.write('**\n')
        f.write('*Output, history, variable=PRESELECT\n')
        f.write('*End Step\n')
        f.close()

    def write_pbc_step_def(self) -> None:

        """simple function to write step definition
        variables should be introduced to give the user an option
        to modify amplidtude, and other parameters"""

        f = open(RveInfo.store_path + '/Step.inp', 'w+')
        f.write('**\n')
        f.write('** Step time needs to be in agreement with amplitude\n')
        f.write('*Amplitude, name=Amp-1\n')
        f.write('             0.,              0.,           1.,        1.,\n')
        f.write('**\n')
        f.write('**BOUNDARY CONDITIONS\n')
        f.write('**\n')
        f.write('** Name: H1 Type: Displacement/Rotation\n')
        f.write('*Boundary\n')
        f.write('H1, 1\n')
        f.write('H1, 2\n')
        f.write('** Name: V1 Type: Displacement/Rotation\n')
        f.write('*Boundary\n')
        f.write('V1, 1\n')
        f.write('V1, 2\n')
        f.write('V1, 3\n')
        f.write('** Name: V2 Type: Displacement/Rotation\n')
        f.write('*Boundary\n')
        f.write('V2, 2\n')
        f.write('V2, 3\n')
        f.write('**Name: V4 Type: Displacement/Rotatio\n')
        f.write('**\n')
        f.write('*Boundary\n')
        f.write('**V4, 1\n')
        f.write('**V4, 2\n')
        f.write('**V4, 3\n')
        f.write('** ----------------------------------------------------------------\n')
        f.write('**\n')
        f.write('** STEP: Step-1\n')
        f.write('**\n')
        f.write('*Step, name=Step-1, nlgeom=YES, inc=10000000, solver=ITERATIVE\n')
        f.write('*Static\n')
        f.write('0.001, 1., 1.05e-15, 0.25\n')
        f.write('**\n')
        f.write('** CONTROLS\n')
        f.write('**\n')
        f.write('*Controls, reset\n')
        f.write('*CONTROLS, PARAMETER=TIME INCREMENTATION\n')
        f.write('35, 50, 9, 50, 28, 5, 12, 45\n')
        f.write('**\n')
        f.write('*CONTROLS, PARAMETERS=LINE SEARCH\n')
        f.write('10\n')
        f.write('*SOLVER CONTROL\n')
        f.write('1e-5,200,\n')
        f.write('**\n')
        f.write('** BOUNDARY CONDITIONS\n')
        f.write('**\n')
        f.write('** Name: Load Type: Displacement/Rotation\n')
        f.write('*Boundary, amplitude=AMP-1\n')
        f.write('V4, 2, 2, 0.01\n')
        f.write('**\n')
        f.write('** OUTPUT REQUESTS\n')
        f.write('**\n')
        f.write('*Restart, write, frequency=0\n')
        f.write('**\n')
        f.write('** FIELD OUTPUT: F-Output-1\n')
        f.write('**\n')
        f.write('*Output, field, frequency=10\n')
        f.write('*Node Output\n')
        f.write('CF, COORD, RF, U\n')
        f.write('*Element Output, directions=YES\n')
        f.write('EVOL, LE, PE, PEEQ, PEMAG, S, SDV\n')
        f.write('*Contact Output\n')
        f.write('CDISP, CSTRESS\n')
        f.write('**\n')
        f.write('** HISTORY OUTPUT: H-Output-1\n')
        f.write('**\n')
        f.write('*Output, history, variable=PRESELECT\n')
        f.write('*End Step\n')
        f.close()

    def write_grain_data(self) -> None:
        f = open(RveInfo.store_path + '/graindata.inp', 'w+')
        f.write('!MMM Crystal Plasticity Input File\n')
        phase1_idx = 0
        numberofgrains = self.n_grains
        phase = [self.rve.loc[self.rve['GrainID'] == i].phaseID.values[0] for i in range(1, numberofgrains + 1)]
        grainsize = [np.cbrt(self.rve.loc[self.rve['GrainID'] == i].shape[0] *
                             RveInfo.bin_size**3*3/4/np.pi) for i in range(1, numberofgrains + 1)]

        for i in range(numberofgrains):
            ngrain = i+1
            if not RveInfo.phase2iso_flag:
                """phi1 = int(np.random.rand() * 360)
                PHI = int(np.random.rand() * 360)
                phi2 = int(np.random.rand() * 360)"""
                phi1 = self.grains_df['phi1'].tolist()[i]
                PHI = self.grains_df['PHI'].tolist()[i]
                phi2 = self.grains_df['phi2'].tolist()[i]
                f.write('Grain: {}: {}: {}: {}: {}\n'.format(ngrain, phi1, PHI, phi2, grainsize[i]))
            else:
                if phase[i] == 1:
                    phase1_idx += 1
                    """phi1 = int(np.random.rand() * 360)
                    PHI = int(np.random.rand() * 360)
                    phi2 = int(np.random.rand() * 360)"""
                    phi1 = self.grains_df['phi1'].tolist()[i]
                    PHI = self.grains_df['PHI'].tolist()[i]
                    phi2 = self.grains_df['phi2'].tolist()[i]
                    f.write('Grain: {}: {}: {}: {}: {}\n'.format(phase1_idx, phi1, PHI, phi2, grainsize[i]))
        f.close()

    def plot_bot(self, rve_smooth_grid: pv.UnstructuredGrid, min_val: float, max_val: float, direction: int = 1,
                 storename: str = 'default', display=True) -> None:

        """first approach for a visualization helper
        should probably be moved to utilities"""

        # get cell centroids
        cells = rve_smooth_grid.cells.reshape(-1, 5)[:, 1:]
        cell_center = rve_smooth_grid.points[cells].mean(1)

        # extract cells below the 0 xy plane
        mask = np.where((cell_center[:, direction] >= min_val) & (cell_center[:, direction] <= max_val))
        cell_ind = np.asarray(mask)
        subgrid = rve_smooth_grid.extract_cells(cell_ind)

        # advanced plotting
        plotter = pv.Plotter()
        plotter.add_mesh(subgrid, 'lightgrey', lighting=True, show_edges=True, scalars='GrainID')
        plotter.remove_scalar_bar()
        plotter.add_bounding_box()
        if storename == 'default':
            plotter.show()
        elif storename != 'default' and display:
            plotter.show(screenshot=RveInfo.store_path + '/' + storename + '.png')
        else:
            plotter.show(screenshot=RveInfo.store_path + '/' + storename + '.png', auto_close=True)

    def run(self) -> None:
        if RveInfo.gui_flag:
            RveInfo.progress_obj.emit(0)
            RveInfo.infobox_obj.emit('starting mesher')
        GRID = self.gen_blocks()
        if RveInfo.gui_flag:
            RveInfo.progress_obj.emit(25)
        GRID = self.gen_grains(GRID)
        smooth_mesh = self.smoothen_mesh(GRID, n_iter=200)
        pbc_grid = smooth_mesh
        if RveInfo.gui_flag:
            RveInfo.progress_obj.emit(50)
        if RveInfo.roughness_flag:
            # TODO: roghness einbauen
            #grid = self.apply_roughness(grid)
            pass

        f = open(RveInfo.store_path + '/DRAGen_RVE.inp', 'w+')
        f.write('*Heading\n')
        f.write('** Job name: Job-1 Model name: Job-1\n')
        f.write('** Generated by: DRAGen \n')
        f.write('** Date: {}\n'.format(datetime.datetime.now().strftime("%d.%m.%Y")))
        f.write('** Time: {}\n'.format(datetime.datetime.now().strftime("%H:%M:%S")))
        f.write('*Preprint, echo=NO, model=NO, history=NO, contact=NO\n')
        f.write('**\n')
        f.write('** PARTS\n')
        f.write('**\n')
        f.close()

        pv.save_meshio(RveInfo.store_path + '/rve-part.inp', smooth_mesh)
        f = open(RveInfo.store_path + '/rve-part.inp', 'r')
        lines = f.readlines()
        f.close()
        lines = [line.lower() for line in lines]
        startingLine = lines.index('*node\n')
        f = open(RveInfo.store_path + '/DRAGen_RVE.inp', 'a')
        f.write('*Part, name=PART-1\n')
        for line in lines[startingLine:]:
            if line.replace(" ", "") == "*element,type=c3d8rh\n":
                line = "*element,type=c3d8\n"
            if '*end' in line:
                line = line.replace('*end', '**\n')
            f.write(line)
        for i in range(self.n_grains):
            nGrain = i + 1
            cells = np.where(smooth_mesh.cell_data['GrainID'] == nGrain)[0]
            f.write('*Elset, elset=Set-{}\n'.format(nGrain))
            for j, cell in enumerate(cells + 1):
                if (j + 1) % 16 == 0:
                    f.write('\n')
                f.write(' {},'.format(cell))
            f.write('\n')

        phase1_idx = 0
        phase2_idx = 0
        phase3_idx = 0
        phase4_idx = 0
        for i in range(self.n_grains):
            nGrain = i + 1
            if self.rve.loc[GRID.cell_data['GrainID'] == nGrain].phaseID.values[0] == 1:
                phase1_idx += 1
                f.write('** Section: Section - {}\n'.format(nGrain))
                f.write('*Solid Section, elset=Set-{}, material=Ferrite_{}\n'.format(nGrain, phase1_idx))
            elif self.rve.loc[GRID.cell_data['GrainID'] == nGrain].phaseID.values[0] == 2:
                if not RveInfo.phase2iso_flag:
                    phase2_idx += 1
                    f.write('** Section: Section - {}\n'.format(nGrain))
                    f.write('*Solid Section, elset=Set-{}, material=Martensite_{}\n'.format(nGrain, phase2_idx))
                else:
                    f.write('** Section: Section - {}\n'.format(nGrain))
                    f.write('*Solid Section, elset=Set-{}, material=Martensite\n'.format(nGrain))
            elif self.rve.loc[GRID.cell_data['GrainID'] == nGrain].phaseID.values[0] == 3:
                if not RveInfo.phase2iso_flag:
                    phase2_idx += 1
                    f.write('** Section: Section - {}\n'.format(nGrain))
                    f.write('*Solid Section, elset=Set-{}, material=Pearlite_{}\n'.format(nGrain, phase3_idx))
                else:
                    f.write('** Section: Section - {}\n'.format(nGrain))
                    f.write('*Solid Section, elset=Set-{}, material=Pearlite\n'.format(nGrain))
            elif self.rve.loc[GRID.cell_data['GrainID'] == nGrain].phaseID.values[0] == 4:
                if not RveInfo.phase2iso_flag:
                    phase2_idx += 1
                    f.write('** Section: Section - {}\n'.format(nGrain))
                    f.write('*Solid Section, elset=Set-{}, material=Bainite_{}\n'.format(nGrain, phase4_idx))
                else:
                    f.write('** Section: Section - {}\n'.format(nGrain))
                    f.write('*Solid Section, elset=Set-{}, material=Bainite\n'.format(nGrain))

        f.close()
        os.remove(RveInfo.store_path + '/rve-part.inp')
        x_max = max(GRID.points[:, 0])
        x_min = min(GRID.points[:, 0])
        y_max = max(GRID.points[:, 1])
        y_min = min(GRID.points[:, 1])
        z_max = max(GRID.points[:, 2])
        z_min = min(GRID.points[:, 2])

        grid_hull_df = pd.DataFrame(pbc_grid.points, columns=['x', 'y', 'z'])
        grid_hull_df = grid_hull_df.loc[(grid_hull_df['x'] == x_max) | (grid_hull_df['x'] == x_min) |
                                        (grid_hull_df['y'] == y_max) | (grid_hull_df['y'] == y_min) |
                                        (grid_hull_df['z'] == z_max) | (grid_hull_df['z'] == z_min)]

        self.make_assembly()
        if RveInfo.gui_flag:
            RveInfo.progress_obj.emit(75)
        if RveInfo.submodel_flag:
            self.submodelSet(grid_hull_df)
        if RveInfo.pbc_flag:
            self.pbc(GRID, grid_hull_df)
        self.write_material_def()
        if RveInfo.submodel_flag:
            self.write_submodel_step_def()
        elif RveInfo.pbc_flag:
            self.write_pbc_step_def()
        self.write_grain_data()

        if RveInfo.anim_flag:
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(smooth_mesh, scalars='phaseID', scalar_bar_args={'title': 'Phase IDs'},
                             show_edges=True, interpolate_before_map=True)
            plotter.add_axes()
            plotter.show(interactive=True, auto_close=True, window_size=[800, 600],
                         screenshot=RveInfo.store_path+'/Figs/pyvista_smooth_Mesh_phases.png')
            plotter.close()

            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(smooth_mesh, scalars='GrainID', scalar_bar_args={'title':'Grain IDs'},
                             show_edges=True, interpolate_before_map=True)
            plotter.add_axes()
            plotter.show(interactive=True, auto_close=True, window_size=[800, 600],
                         screenshot=RveInfo.store_path + '/Figs/pyvista_smooth_Mesh_grains.png')
            plotter.close()

        if RveInfo.gui_flag:
            RveInfo.progress_obj.emit(100)