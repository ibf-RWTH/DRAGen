import numpy as np
import random
import math
from dragen.utilities.InputInfo import RveInfo
import pyvista as pv
import matplotlib.pyplot as plt


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

def find_closest_points_in_rectangle(corner, v1, v2, norm, points, tol=1e-6):
    """
    Find points within the rectangle in 3D space by computing the closest z values using vectorized operations.

    Parameters:
        corner (numpy array): Coordinates of one corner of the rectangle.
        v1 (numpy array): Vector along one edge of the rectangle.
        v2 (numpy array): Vector along the adjacent edge of the rectangle.
        norm (numpy array): Unit normal vector to the rectangle plane.
        points (numpy array): Array of points to check, shape (n_points, 3).
        tol (float): Tolerance for numerical precision when comparing z-values.

    Returns:
        numpy array: Points that are closest to the computed z values within the rectangle.
    """

    # Calculate the plane's d constant using the dot product
    d = np.dot(norm, corner)

    # Calculate the other corners of the rectangle
    corner1 = corner + v1
    corner2 = corner + v2
    corner3 = corner + v1 + v2

    # Find the min and max x, y coordinates for the bounding box
    min_x = min(corner[0], corner1[0], corner2[0], corner3[0])
    max_x = max(corner[0], corner1[0], corner2[0], corner3[0])
    min_y = min(corner[1], corner1[1], corner2[1], corner3[1])
    max_y = max(corner[1], corner1[1], corner2[1], corner3[1])

    # Filter points whose x and y coordinates are within the bounding box
    mask = (points[:, 0] >= min_x) & (points[:, 0] <= max_x) & (points[:, 1] >= min_y) & (points[:, 1] <= max_y)
    points_within_xy = points[mask]
    indices_within_xy = np.where(mask)[0]

    # Calculate the correct z value on the rectangle's plane for these (x, y) points
    a, b, c = norm
    computed_z_values = (d - a * points_within_xy[:, 0] - b * points_within_xy[:, 1]) / c

    # Compute the distance in z between the points and the rectangle's plane
    z_distances = np.abs(points_within_xy[:, 2] - computed_z_values)

    # Select the points where the z distance is within the tolerance
    closest_points = indices_within_xy[z_distances < tol]

    return closest_points

def set_init_damage_field(mesh: pv.UnstructuredGrid) -> None:
    norm_tol = RveInfo.norm_tol
    print("tol is", norm_tol)
    num_cracks = int(RveInfo.box_volume * RveInfo.crack_density) + 1
    print("number of cracks is: ", num_cracks)
    width = RveInfo.box_size
    print("box size is", RveInfo.box_size)
    height = RveInfo.box_size if RveInfo.box_size_y is None else RveInfo.box_size_y
    depth = RveInfo.box_size if RveInfo.box_size_z is None else RveInfo.box_size_z
    radius = np.power(RveInfo.box_volume / num_cracks, 1 / 3)
    print("average distance between cracks is:", radius)
    center_points = generate_poisson_points(width, height, depth, radius)
    damage_node_sets = []
    # generate cracks as tiny rectangles
    mean_crack_len = RveInfo.mean_crack_len
    # Set standard deviation of the logarithm of the variable
    sigma = RveInfo.crack_len_sigma  # Adjust this value as needed for different spreads

    # Calculate mu based on the desired mean
    mu = np.log(mean_crack_len) - (sigma ** 2) / 2
    crack_size_list = np.random.lognormal(mean=mu, sigma=sigma, size=(num_cracks, 2))

    rve = mesh
    points = np.asarray(rve.points)
    points = points * 1000
    print("points are:", points)
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
            # print("corner and vecs are", corner, v1, v2)
            # plot_damage(points=points, corner=corner, v1=v1, v2=v2)

            closet_points = find_closest_points_in_rectangle(corner=corner, v1=v1, v2=v2, norm=norm,
                                                             points=points, tol=norm_tol)
            # print("closet points are:", closet_points)
            if len(closet_points) > 0:
                damage_node_sets.extend(closet_points)
                break


        print(f"damaged nodes on crack {i + 1} found!")

    print("damage node sets generation done!")
    with open(RveInfo.store_path + '/DamageNodesSet.inp', 'w') as f:
        f.write('*Nset, nset=DamageNodes, instance=PART-1-1\n')
        for idx, node in enumerate(damage_node_sets):
            f.write(f"{node},")
            if (idx + 1) % 10 == 0:
                f.write("\n")

    print("finish writing damage nodes inp!")


def plot_damage(points, corner, v1, v2):
    corner1 = corner + v1
    corner2 = corner + v2
    corner3 = corner + v1 + v2

    # Collect all corners
    corners = np.array([corner, corner1, corner2, corner3])

    # Plotting the rectangle and the points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the rectangle by connecting the corners
    rectangle_edges = [[corner, corner1], [corner1, corner3], [corner3, corner2], [corner2, corner]]
    for edge in rectangle_edges:
        ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], [edge[0][2], edge[1][2]], 'r-', linewidth=10)

    # Plot the corners of the rectangle
    ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2], color='red', label='Rectangle Corners',s=50)

    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='green', label='Points', marker='x', s=5)

    # Setting labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([0,30])
    ax.set_ylim([0,30])
    ax.set_zlim([0,30])
    ax.set_title('Rectangle and Points in 2D')
    ax.legend()

    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # rve_data = pd.read_csv(r'F:\temp\OutputData\2024-08-11_000\periodic_rve_df.csv')
    # print("x range: ",rve_data['x'].min(), rve_data['x'].max())
    # Parameters
    RveInfo.box_size = 30
    RveInfo.box_size_y = 30
    RveInfo.box_size_z = 30
    RveInfo.bin_size = 0.001
    RveInfo.box_volume = 27000
    RveInfo.crack_density = 1e-3
    RveInfo.norm_tol = 0.1
    RveInfo.mean_crack_len = 5
    RveInfo.store_path = r'C:\temp\OutputData\2024-08-12_000'
    RveInfo.crack_len_sigma = 0.5
    rve = pv.read(r'C:\temp\OutputData\2024-08-12_000\rve-part.vtk')
    set_init_damage_field(mesh=rve)
