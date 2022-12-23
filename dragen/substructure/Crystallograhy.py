import numpy as np
import pandas as pd


class CrystallInfo:
    """
    record the crystallographic information of substructures
    after grains are generated -> collect grains orientations, global habit plane norm
    after packets are generated -> assign and collect habit plane norm 
    after blocks are generated -> assign and collect blocks orientations
    """
    variants = None
    habit_planes = []
    gid2orientation = []
    gid2habit_planes = dict()
    pid2hp_norm = dict()
    pid2gid = dict()
    pid2variants = dict()
    old_pid = []
    T_list = [np.array(0) for i in range(24)]

    def __init__(self, orientation_relationship: str = 'KS'):
        """
        based on KS orientation relationship
        :param orientation_relationship: Kurdjumov–Sachs orientation relationship
        """
        self.orientation_relationship = orientation_relationship
        if self.orientation_relationship == 'KS':
            variants_list = []
            for j in range(1, 5):
                packet_variants_list = []
                for i in range(1, 7):
                    packet_variants_list.append('V' + str(i + 6 * (j - 1)))

                variants_list.append(packet_variants_list)

                CrystallInfo.T_list[0] = np.array([[0.742, 0.667, 0.075],
                                                   [0.650, 0.742, 0.167],
                                                   [0.167, 0.075, 0.983]])

                CrystallInfo.T_list[1] = np.array([[0.075, 0.667, -0.742],
                                                   [-0.167, 0.742, 0.650],
                                                   [0.983, 0.075, 0.167]])

                CrystallInfo.T_list[2] = np.array([[-0.667, -0.075, 0.742, ],
                                                   [0.742, -0.167, 0.650],
                                                   [0.075, 0.983, 0.167]])

                CrystallInfo.T_list[3] = np.array([[0.667, -0.742, 0.075],
                                                   [0.742, 0.650, -0.167],
                                                   [0.075, 0.167, 0.983]])

                CrystallInfo.T_list[4] = np.array([[-0.075, 0.742, -0.667],
                                                   [-0.167, 0.650, 0.742],
                                                   [0.983, 0.167, 0.075]])

                CrystallInfo.T_list[5] = np.array([[-0.742, 0.075, 0.667],
                                                   [0.650, -0.167, 0.742],
                                                   [0.167, 0.983, 0.075]])

                CrystallInfo.T_list[6] = np.array([[-0.075, 0.667, 0.742],
                                                   [-0.167, -0.742, 0.650],
                                                   [0.983, -0.075, 0.167]])

                CrystallInfo.T_list[7] = np.array([[-0.742, -0.667, 0.075],
                                                   [0.650, -0.742, -0.167],
                                                   [0.167, -0.075, 0.983]])

                CrystallInfo.T_list[8] = np.array([[0.742, 0.075, -0.667],
                                                   [0.650, 0.167, 0.742],
                                                   [0.167, -0.983, 0.075]])

                CrystallInfo.T_list[9] = np.array([[0.075, 0.742, 0.667],
                                                   [-0.167, -0.650, 0.742],
                                                   [0.983, -0.167, 0.075]])

                CrystallInfo.T_list[10] = np.array([[-0.667, -0.742, -0.075],
                                                    [0.742, -0.650, -0.167],
                                                    [0.075, -0.167, 0.983]])

                CrystallInfo.T_list[11] = np.array([[0.667, -0.075, -0.742],
                                                    [0.742, 0.167, 0.650],
                                                    [0.075, -0.983, 0.167]])

                CrystallInfo.T_list[12] = np.array([[0.667, 0.742, -0.075],
                                                    [-0.742, 0.650, -0.167],
                                                    [-0.075, 0.167, 0.983]])

                CrystallInfo.T_list[13] = np.array([[-0.667, 0.075, -0.742],
                                                    [-0.742, -0.167, 0.650],
                                                    [-0.075, 0.983, 0.167]])

                CrystallInfo.T_list[14] = np.array([[0.075, -0.667, 0.742],
                                                    [0.167, 0.742, 0.650],
                                                    [-0.983, 0.075, 0.167]])

                CrystallInfo.T_list[15] = np.array([[0.742, 0.667, 0.075],
                                                    [-0.650, 0.742, -0.167],
                                                    [-0.167, 0.075, 0.983]])

                CrystallInfo.T_list[16] = np.array([[-0.742, 0.075, -0.667],
                                                    [-0.650, -0.167, 0.742],
                                                    [-0.167, 0.983, 0.075]])

                CrystallInfo.T_list[17] = np.array([[-0.075, -0.742, 0.667],
                                                    [0.167, 0.650, 0.742],
                                                    [-0.983, 0.167, 0.075]])

                CrystallInfo.T_list[18] = np.array([[0.742, -0.075, 0.667],
                                                    [0.650, -0.167, -0.742],
                                                    [0.167, 0.983, -0.075]])

                CrystallInfo.T_list[19] = np.array([[0.075, -0.742, -0.667],
                                                    [-0.167, 0.650, -0.742],
                                                    [0.983, 0.167, -0.075]])

                CrystallInfo.T_list[20] = np.array([[-0.667, 0.742, 0.075],
                                                    [0.742, 0.650, 0.167],
                                                    [0.075, 0.167, -0.983]])

                CrystallInfo.T_list[21] = np.array([[0.667, 0.075, 0.742],
                                                    [0.742, -0.167, -0.650],
                                                    [0.075, 0.983, -0.167]])

                CrystallInfo.T_list[22] = np.array([[-0.075, -0.667, -0.742],
                                                    [-0.167, 0.742, -0.650],
                                                    [0.983, 0.075, -0.167]])

                CrystallInfo.T_list[23] = np.array([[-0.742, 0.667, -0.075],
                                                    [0.650, 0.742, 0.167],
                                                    [0.167, 0.075, -0.983]])

            CrystallInfo.variants = np.array(variants_list)
            CrystallInfo.habit_planes = [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]]

    @staticmethod
    def lc_to_gc(orientation):
        R1 = np.array([[np.cos(np.deg2rad(orientation[0])), -np.sin(np.deg2rad(orientation[0])), 0],
                       [np.sin(np.deg2rad(orientation[0])), np.cos(np.deg2rad(orientation[0])), 0],
                       [0, 0, 1]], dtype=object)

        R2 = np.array([[1, 0, 0],
                       [0, np.cos(np.deg2rad(orientation[1])), -np.sin(np.deg2rad(orientation[1]))],
                       [0, np.sin(np.deg2rad(orientation[1])), np.cos(np.deg2rad(orientation[1]))]], dtype=object)

        R3 = np.array([[np.cos(np.deg2rad(orientation[2])), -np.sin(np.deg2rad(orientation[2])), 0],
                       [np.sin(np.deg2rad(orientation[2])), np.cos(np.deg2rad(orientation[2])), 0],
                       [0, 0, 1]], dtype=object)

        result = np.dot(R3, R2)
        R = np.matrix(np.dot(result, R1))
        R = R.astype(float)  # new error has to convert data type
        R_I = R.I

        habit_plane_list = np.array(CrystallInfo.habit_planes)
        transferred_hp_list = (R_I.dot(habit_plane_list.T)).T

        return transferred_hp_list

    @staticmethod
    def get_global_hp_norm(GrainID, orientation):
        hp_normal_list = CrystallInfo.lc_to_gc(orientation=orientation)
        CrystallInfo.gid2habit_planes[GrainID] = hp_normal_list

    @staticmethod
    def assign_variants(packet_id: str):
        """
        assign packet with KS variants
        :param packet_id: id of packet before transferring to number
        :return:
        """
        normal = CrystallInfo.pid2hp_norm[packet_id]
        grain_id = CrystallInfo.pid2gid[packet_id]
        h_list = CrystallInfo.gid2habit_planes[grain_id]
        hp_list = []
        for h in h_list:
            hp_list.append(list(np.array(h).squeeze()))
        hp_list = np.array(hp_list)

        possible_v = CrystallInfo.variants[np.where((hp_list == normal).all(axis=1))[0], :]
        CrystallInfo.pid2variants[packet_id] = possible_v

        return possible_v

    @staticmethod
    def assign_bv(packet: pd.DataFrame):

        bid = packet['block_id'].unique()

        vidx = [1 for i in range(len(bid))]

        for i in range(len(bid)):
            pv = [0, 1, 2, 3, 4, 5]
            if i >= 1:
                pv.remove(vidx[i - 1])
            vidx[i] = np.random.choice(pv, 1)[0]

        pid = int(packet.iloc[0]['packet_id'])
        variant_trial_list = CrystallInfo.pid2variants[CrystallInfo.old_pid[pid - 1]]
        bid_to_vidx = dict(zip(bid, vidx))

        packet['block_variant'] = packet.apply(
            lambda p: variant_trial_list[..., bid_to_vidx[p['block_id']]][0], axis=1)

        return packet

    @staticmethod
    def comp_angle(pag_ori, block_variant: str):
        i = int(block_variant.lstrip('V')) - 1
        T = CrystallInfo.T_list[i]
        phi1 = pag_ori[0]
        PHI = pag_ori[1]
        phi2 = pag_ori[2]

        R1 = np.array([[np.cos(np.deg2rad(phi1)), -np.sin(np.deg2rad(phi1)), 0],
                       [np.sin(np.deg2rad(phi1)), np.cos(np.deg2rad(phi1)), 0],
                       [0, 0, 1]])

        R2 = np.array([[1, 0, 0],
                       [0, np.cos(np.deg2rad(PHI)), -np.sin(np.deg2rad(PHI))],
                       [0, np.sin(np.deg2rad(PHI)), np.cos(np.deg2rad(PHI))]])

        R3 = np.array([[np.cos(np.deg2rad(phi2)), -np.sin(np.deg2rad(phi2)), 0],
                       [np.sin(np.deg2rad(phi2)), np.cos(np.deg2rad(phi2)), 0],
                       [0, 0, 1]])

        result = np.dot(R3, R2)
        R = np.matrix(np.dot(result, R1))

        RB = T * R
        N, n, n1, n2 = 1, 1, 1, 1

        if RB[2, 2] > 1:
            N = 1 / RB[2, 2]

        if RB[2, 2] < -1:
            N = -1 / RB[2, 2]

        RB[2, 2] = N * RB[2, 2]
        PHIB = np.degrees(np.arccos(RB[2, 2]))
        sin_PHIB = np.sin(np.deg2rad(PHIB))

        if RB[2, 0] / sin_PHIB > 1 or RB[2, 0] / sin_PHIB < -1:
            n1 = sin_PHIB / RB[2, 0]

        if RB[0, 2] / sin_PHIB > 1 or RB[0, 2] / sin_PHIB < -1:
            n2 = sin_PHIB / RB[0, 2]

        if abs(n1) > abs(n2):

            n = n2

        else:

            n = n1

        # recalculate after scaling
        RB = N * n * RB
        PHIB = np.degrees(np.arccos(RB[2, 2]))
        if PHIB < 0:
            PHIB = PHIB + 360
        sin_PHIB = np.sin(np.deg2rad(PHIB))
        phi1B = np.degrees(np.arcsin(RB[2, 0] / sin_PHIB))
        if phi1B < 0:
            phi1B = phi1B + 360
        phi2B = np.degrees(np.arcsin(RB[0, 2] / sin_PHIB))
        if phi2B < 0:
            phi2B = phi2B + 360

        return phi1B, PHIB, phi2B


if __name__ == "__main__":
    grains_df = pd.read_csv(r"/OutputData/2022-11-16_000/Generation_Data/grain_data_output.csv")
    crystall_info = CrystallInfo()
    print("compute global habit planes norm for each grain")
    grains_df.apply(lambda grain: crystall_info.get_global_hp_norm(GrainID=grain['GrainID'],
                                                                   orientation=(
                                                                       grain['phi1'], grain['PHI'], grain['phi2'])),
                    axis=1)
    print(CrystallInfo.gid2habit_planes)
