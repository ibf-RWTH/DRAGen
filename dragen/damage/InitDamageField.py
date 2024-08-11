import numpy as np
import random
import math
from dragen.utilities.InputInfo import RveInfo
import pyvista as pv


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def generate_grid(width, height, depth, radius):
    cell_size = radius / math.sqrt(3)
    grid = {}
    for x in range(int(width / cell_size) + 1):
        for y in range(int(height / cell_size) + 1):
            for z in range(int(depth / cell_size) + 1):
                grid[(x, y, z)] = None
    return grid, cell_size


def get_cell_coords(point, cell_size):
    return int(point[0] / cell_size), int(point[1] / cell_size), int(point[2] / cell_size)


def in_neighborhood(grid, point, cell_size, radius):
    cell_coords = get_cell_coords(point, cell_size)
    for x in range(cell_coords[0] - 2, cell_coords[0] + 3):
        for y in range(cell_coords[1] - 2, cell_coords[1] + 3):
            for z in range(cell_coords[2] - 2, cell_coords[2] + 3):
                neighbor = grid.get((x, y, z))
                if neighbor is not None and distance(point, neighbor) < radius:
                    return True
    return False


def generate_poisson_points(width, height, depth, radius, k=30):
    grid, cell_size = generate_grid(width, height, depth, radius)
    points = []
    active_list = []

    # Generate the first point
    first_point = (random.uniform(0, width), random.uniform(0, height), random.uniform(0, depth))
    points.append(first_point)
    active_list.append(first_point)
    grid[get_cell_coords(first_point, cell_size)] = first_point

    while active_list:
        current_point = random.choice(active_list)
        found = False
        for _ in range(k):
            phi = random.uniform(0, 2 * math.pi)
            theta = random.uniform(0, math.pi)
            r = random.uniform(radius, 2 * radius)
            new_point = (
                current_point[0] + r * math.sin(theta) * math.cos(phi),
                current_point[1] + r * math.sin(theta) * math.sin(phi),
                current_point[2] + r * math.cos(theta)
            )

            if 0 <= new_point[0] < width and 0 <= new_point[1] < height and 0 <= new_point[2] < depth:
                if not in_neighborhood(grid, new_point, cell_size, radius):
                    points.append(new_point)
                    active_list.append(new_point)
                    grid[get_cell_coords(new_point, cell_size)] = new_point
                    found = True
                    break

        if not found:
            active_list.remove(current_point)

    return points


def generate_random_euler_angles():
    # Generate three random Euler angles
    alpha = np.random.uniform(0, 2 * np.pi)  # Rotation around x-axis
    beta = np.random.uniform(0, np.pi)  # Rotation around y-axis
    gamma = np.random.uniform(0, 2 * np.pi)  # Rotation around z-axis
    return alpha, beta, gamma


def compute_rotation_matrix(alpha, beta, gamma):
    # Compute the rotation matrix components
    R_z = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])

    R_y = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])

    # Combine the rotation matrices to get the final rotation matrix
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def find_points_within_rectangle(corner, v1, v2, len_v1, len_v2, norm, points, tol=1e-6):
    """
    Find points within a rectangle in 3D space.

    Parameters:
        corner (numpy array): Coordinates of one corner of the rectangle.
        v1 (numpy array): Vector along one edge of the rectangle.
        v2 (numpy array): Vector along the adjacent edge of the rectangle.
        norm (numpy array): Unit normal vector to the rectangle plane.
        points (numpy array): Array of points to check, shape (n_points, 3).
        tol (float): Tolerance for numerical precision.

    Returns:
        list: Points that lie within the rectangle.
    """

    # Initialize an empty list to hold points within the rectangle
    points_within = []

    # Loop through each point
    for idx, point in enumerate(points):
        # Check if the point lies on the plane of the rectangle
        if np.abs(np.dot(point - corner, norm)) < tol:
            # Project the point onto the rectangle's local coordinates
            projection = point - corner
            d1 = np.dot(projection, v1) / len_v1 ** 2
            d2 = np.dot(projection, v2) / len_v2 ** 2

            # Check if the projected point lies within the rectangle bounds
            if 0 <= d1 <= 1 and 0 <= d2 <= 1:
                points_within.append(idx)

    return points_within


def set_init_damage_field(mesh: pv.UnstructuredGrid) -> None:
    norm_tol = RveInfo.norm_tol
    num_cracks = int(RveInfo.box_volume * RveInfo.crack_density) + 1
    print("crack number is: ", num_cracks)
    width, height, depth = RveInfo.box_size, RveInfo.box_size_y, RveInfo.box_size_z
    radius = np.power(RveInfo.box_volume / num_cracks, 1 / 3)
    # print("average distance between cracks is:", radius)
    center_points = generate_poisson_points(width, height, depth, radius)
    damage_node_sets = []
    # generate cracks as tiny rectangles
    mean_crack_len = RveInfo.mean_crack_len
    # Set standard deviation of the logarithm of the variable
    sigma = RveInfo.crack_len_sigma  # Adjust this value as needed for different spreads

    # Calculate mu based on the desired mean
    mu = np.log(mean_crack_len) - (sigma ** 2) / 2
    crack_size_list = np.random.lognormal(mean=mu, sigma=sigma, size=(num_cracks, 2))
    # print(crack_size_list)
    rve = mesh
    points = np.asarray(rve.points)
    for i in range(len(center_points)):
        # compute 3 corners of cracks
        center = np.asarray(center_points[i])
        half_length, half_width = crack_size_list[i, :] / 2

        while True:
            # Generate random Euler angles
            alpha, beta, gamma = generate_random_euler_angles()
            # Compute the corresponding rotation matrix
            rotation_matrix = compute_rotation_matrix(alpha, beta, gamma)
            norm, tangent, bitangent = rotation_matrix[:, 0], rotation_matrix[:, 1], rotation_matrix[:, 2]
            corner = center - half_length * tangent - half_width * bitangent  # left bottom
            v1 = 2 * half_length * tangent
            v2 = 2 * half_width * bitangent  # 2 vectors along crack edge
            # find nodes within cracks
            in_points = find_points_within_rectangle(corner, v1, v2, 2 * half_length, 2 * half_width, norm, points,
                                                     tol=norm_tol)
            if len(in_points) > 0:
                damage_node_sets.extend(in_points)
                break

    print("damage node sets generation done!")
    with open(RveInfo.store_path + '/DamageNodesSet.inp', 'w') as f:
        f.write('*Nset, nset=DamageNodes, instance=PART-1-1\n')
        for idx, node in enumerate(damage_node_sets):
            f.write(f"{node},")
            if (idx + 1) % 10 == 0:
                f.write("\n")

    print("finish writing damage nodes inp!")


if __name__ == "__main__":
    # rve_data = pd.read_csv(r'F:\temp\OutputData\2024-08-11_000\periodic_rve_df.csv')
    # print("x range: ",rve_data['x'].min(), rve_data['x'].max())
    # Parameters
    RveInfo.box_size = 0.030
    RveInfo.box_size_y = 0.030
    RveInfo.box_size_z = 0.030
    RveInfo.bin_size = 0.001
    RveInfo.box_volume = RveInfo.box_size * RveInfo.box_size_y * RveInfo.box_size_z
    RveInfo.crack_density = 1e6
    RveInfo.norm_tol = 1e-5
    RveInfo.mean_crack_len = 0.005
    RveInfo.store_path = r'F:\temp\OutputData\2024-08-11_000'
    RveInfo.crack_len_sigma = 0.5
    rve = pv.read(r'F:\temp\OutputData\2024-08-11_000\rve-part.vtk')
    set_init_damage_field(mesh=rve)
    # num_cracks = int(RveInfo.box_volume * RveInfo.crack_density) + 1
    # print("crack number is: ", num_cracks)
    # width, height, depth = RveInfo.box_size, RveInfo.box_size_y, RveInfo.box_size_z
    # radius = np.power(RveInfo.box_volume / num_cracks, 1 / 3)
    # # print("average distance between cracks is:", radius)
    # center_points = generate_poisson_points(width, height, depth, radius)
    # damage_node_sets = []
    # # generate cracks as tiny rectangles
    # mean_crack_len = 5 * RveInfo.bin_size
    # # Set standard deviation of the logarithm of the variable
    # sigma = 0.5  # Adjust this value as needed for different spreads
    #
    # # Calculate mu based on the desired mean
    # mu = np.log(mean_crack_len) - (sigma ** 2) / 2
    # crack_size_list = np.random.lognormal(mean=mu, sigma=sigma, size=(num_cracks, 2))
    # # print(crack_size_list)
    # rve = pv.read(r'F:\temp\OutputData\2024-08-11_000\rve-part.vtk')
    # points = np.asarray(rve.points)
    # for i in range(len(center_points)):
    #     # compute 3 corners of cracks
    #     center = np.asarray(center_points[i])
    #     half_length, half_width = crack_size_list[i, :] / 2
    #
    #     while True:
    #         # Generate random Euler angles
    #         alpha, beta, gamma = generate_random_euler_angles()
    #         # Compute the corresponding rotation matrix
    #         rotation_matrix = compute_rotation_matrix(alpha, beta, gamma)
    #         norm, tangent, bitangent = rotation_matrix[:, 0], rotation_matrix[:, 1], rotation_matrix[:, 2]
    #         corner = center - half_length * tangent - half_width * bitangent  # left bottom
    #         v1 = 2 * half_length * tangent
    #         v2 = 2 * half_width * bitangent  # 2 vectors along crack edge
    #         # find nodes within cracks
    #         in_points = find_points_within_rectangle(corner, v1, v2, 2 * half_length, 2 * half_width, norm, points,
    #                                                  tol=norm_tol)
    #         if len(in_points) > 0:
    #             damage_node_sets.extend(in_points)
    #             break
    #
    # print(damage_node_sets)
    # with open('DamageNodesSet.inp','w') as f:
    #     f.write('*Nset, nset=DamageNodes, instance=PART-1-1\n')
    #     for idx, node in enumerate(damage_node_sets):
    #         f.write(f"{node},")
    #         if (idx + 1) % 10 == 0:
    #             f.write("\n")
