#### generate sacculus point cloud


import pims
import cv2
import numpy as np
import open3d as o3d
import pyvista as pv
from matplotlib import pyplot as plt

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    return pcd_down

# Load the image sequence
frames = pims.ImageSequence('F:/test_unet/predicted_mask2/*.png')
org_frames = pims.ImageSequence('F:/cropped_sacculus/*.jpg')

frames = pims.ImageSequence('H:/new/predicted_mask/*.png') #sample2

org_frames = pims.ImageSequence('H:/new/results/*.tif') #sample2
org_size =np.shape(org_frames[0])
# Initialize a list to store contour points
all_points = []
cnts=[]
for i in range(0,len(frames)):
    img = frames[i]
    #img = cv2.resize(img, (2800, 3500), interpolation=cv2.INTER_LINEAR)
    #img = cv2.resize(img, org_size, interpolation=cv2.INTER_LINEAR)
    edged = cv2.Canny(img, 0, 1)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts.append(contours)
    # Extract points from contours and add a z-coordinate based on the frame index
    for contour in contours:
        for point in contour:
            all_points.append([point[0][0], point[0][1], i * 1.0])  # z-coordinate increases with frame index

i=np.random.randint(0,len(frames))
img_copy=frames[i].copy()# np.uint8(np.ones(np.shape(frames[i])))#frames[i].copy()
img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img_copy, cnts[i],-1,color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
cv2.namedWindow('custom window', cv2.WINDOW_KEEPRATIO)
cv2.imshow('custom window',img_copy)
cv2.resizeWindow('custom window', 500, 500)
cv2.waitKey(0)
cv2.destroyAllWindows()



##############################

# Convert list of points to numpy array
all_points = np.array(all_points, dtype=np.float32)


original_shape = (3500, 2800)
resized_shape = (3500,2800)

# Calculate the scaling factors due to resizing
scale_x = original_shape[0] / resized_shape[0]
scale_y = original_shape[1] / resized_shape[1]

# Physical resolution in nm
resolution_xy = 10  # 10 nm in XY plane
resolution_z = 30   # 30 nm in Z plane

# Scale the point cloud
scaled_points = np.zeros_like(all_points)
scaled_points[:, 0] = all_points[:, 0] * scale_x * resolution_xy  # Scale X coordinates
scaled_points[:, 1] = all_points[:, 1] * scale_y * resolution_xy  # Scale Y coordinates
scaled_points[:, 2] = all_points[:, 2] * resolution_z             # Scale Z coordinates

# Create Open3D point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(scaled_points)
#pcd.points = o3d.utility.Vector3dVector(all_points)
o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud('E:/EM_data_results/point_clouds/sample1.ply', pcd)


o3d.visualization.draw_geometries_with_editing([pcd])
pcd_cropped =o3d.io.read_point_cloud('H:/test_unet/isolated_sensilla_point_clouds/chamber1/cropped_1.ply')
o3d.visualization.draw_geometries([pcd_cropped])

##################
#######################

####  crop sensilla and apply transform based on pca 


import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

def transform_cone_point_cloud(points):
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    pca = PCA(n_components=3)
    pca.fit(centered_points)
    principal_axes = pca.components_
    z_axis = np.array([0, 0, 1])
    axis_to_align = principal_axes[2]
    angle = np.arccos(np.dot(axis_to_align, z_axis) / (np.linalg.norm(axis_to_align) * np.linalg.norm(z_axis)))
    rotation_axis = np.cross(axis_to_align, z_axis)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix = np.array([
        [cos_angle + rotation_axis[0]**2 * (1 - cos_angle),
         rotation_axis[0] * rotation_axis[1] * (1 - cos_angle) - rotation_axis[2] * sin_angle,
         rotation_axis[0] * rotation_axis[2] * (1 - cos_angle) + rotation_axis[1] * sin_angle],
        [rotation_axis[1] * rotation_axis[0] * (1 - cos_angle) + rotation_axis[2] * sin_angle,
         cos_angle + rotation_axis[1]**2 * (1 - cos_angle),
         rotation_axis[1] * rotation_axis[2] * (1 - cos_angle) - rotation_axis[0] * sin_angle],
        [rotation_axis[2] * rotation_axis[0] * (1 - cos_angle) - rotation_axis[1] * sin_angle,
         rotation_axis[2] * rotation_axis[1] * (1 - cos_angle) + rotation_axis[0] * sin_angle,
         cos_angle + rotation_axis[2]**2 * (1 - cos_angle)]
    ])
    rotated_points = np.dot(centered_points, rotation_matrix.T)
    min_z = np.min(rotated_points[:, 2])
    translated_points = rotated_points - np.array([0, 0, min_z])
    return translated_points

def measure_cone_width_at_z_slice(points, z_slice, slice_width=0.1, jitter_strength=1e-5):
    z_min = z_slice - slice_width / 2
    z_max = z_slice + slice_width / 2
    slice_points = points[(points[:, 2] >= z_min) & (points[:, 2] <= z_max)]
    if len(slice_points) < 3:
        return None
    jitter = np.random.normal(scale=jitter_strength, size=slice_points[:, :2].shape)
    slice_points[:, :2] += jitter
    try:
        hull = ConvexHull(slice_points[:, :2])
        hull_points = slice_points[hull.vertices]
        width = np.max([np.linalg.norm(p1 - p2) for i, p1 in enumerate(hull_points) for p2 in hull_points[i+1:]])
    except Exception as e:
        print(f"Error computing convex hull at z={z_slice}: {e}")
        return None
    return width

def measure_cone_width_along_z(points, slice_width=0.1):
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    z_slices = np.arange(z_min, z_max, slice_width)
    widths = []
    for z in z_slices:
        width = measure_cone_width_at_z_slice(points, z, slice_width)
        if width is not None:
            widths.append((z, width))
    return widths

def remove_cropped_points(pcd, pcd_cropped, distance_threshold=0.01):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_cropped)
    diff_points = []
    for point in pcd.points:
        [k, idx, _] = pcd_tree.search_radius_vector_3d(point, distance_threshold)
        if k == 0:
            diff_points.append(point)
    pcd_diff = o3d.geometry.PointCloud()
    pcd_diff.points = o3d.utility.Vector3dVector(np.array(diff_points))
    return pcd_diff

# Load initial point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(scaled_points)
iterations = 5  # Set the number of iterations

for i in range(iterations):
    # Visualize and edit the point cloud to crop a region
    o3d.visualization.draw_geometries_with_editing([pcd])

    # Load the cropped point cloud from the file saved after editing
    pcd_cropped = o3d.io.read_point_cloud(f'H:/test_unet/isolated_sensilla_point_clouds/chamber1/cropped_{i + 1}.ply')
    
    # Transform the cropped point cloud
    points = np.asarray(pcd_cropped.points)
    transformed_points = transform_cone_point_cloud(points)
    pcd_transformed = o3d.geometry.PointCloud()
    pcd_transformed.points = o3d.utility.Vector3dVector(transformed_points)
    o3d.visualization.draw_geometries([pcd_transformed])

    # # Measure widths along the cone
    # widths = measure_cone_width_along_z(transformed_points, slice_width=1)
    # z_values, width_values = zip(*widths)

    # # Plot the width along z
    # plt.scatter(z_values, width_values, marker='o')
    # plt.xlabel('Z Value')
    # plt.ylabel('Width')
    # plt.title(f'Cone Width along Z-axis (Iteration {i + 1})')
    # plt.show()

    # Remove the cropped points from the original point cloud
    pcd = remove_cropped_points(pcd, pcd_cropped, distance_threshold=0.01)

    # Visualize the updated point cloud
    o3d.visualization.draw_geometries([pcd])

print("Process completed.")

##############################
###################################


#load the sacved sensilla point cloud and do a registration using  
#ransac algo  with the help of a target cone 
## target cone dimension for chamber 2 = 1200 (r), 2000 (h)

import pims
import cv2
import numpy as np
import open3d as o3d
import pyvista as pv
from matplotlib import pyplot as plt
import copy

voxel_size = 6

def create_cone_point_cloud(base_radius, height, num_points=1000):
    # Create arrays to hold the point coordinates
    points = []

    # Generate points on the surface of the cone
    for _ in range(num_points):
        # Generate random angle
        theta = np.random.uniform(0, 2 * np.pi)
        
        # Generate random height position (0 at the tip, height at the base)
        z = np.random.uniform(0, height)
        
        # Radius at height z (linear interpolation between 0 at the tip and base_radius at the base)
        r = (height - z) / height * base_radius
        
        # Convert polar coordinates to Cartesian coordinates
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        points.append([x, y, z])
    
    points = np.array(points)
    
    # Create an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    return pcd

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def evaluate_alignment(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source_temp, target, 0.01)
    print(f"Fitness: {evaluation.fitness}")
    print(f"Inlier RMSE: {evaluation.inlier_rmse}")


def reg_3d(source_pt, target_pt, voxel_size = 6 ):
    cont_3d = source_pt
    cone_points = target_pt

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(cont_3d)

    target_pcd =o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(cone_points)


    draw_registration_result(source_pcd, target_pcd, np.identity(4))
    
    # Preprocess point clouds
    source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)

    # Perform global registration
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

    # Visualize the result
    print("Transformation matrix:")
    print(result_ransac.transformation)

    draw_registration_result(source_down, target_down, result_ransac.transformation)
    draw_registration_result(source_pcd, target_pcd, result_ransac.transformation)

    # Evaluate the alignment
    evaluate_alignment(source_pcd, target_pcd, result_ransac.transformation)


    source_transformed = copy.deepcopy(source_pcd)
    source_transformed.transform(result_ransac.transformation)

    # Visualize the aligned point clouds
    draw_registration_result(source_transformed, target_pcd, np.identity(4))

    # Convert the transformed source point cloud to a NumPy array
    source_transformed_np = np.asarray(source_transformed.points)
    return(source_transformed_np)

def plot_transformed_mesh( source_transformed_np, alpha):
    pc=pv.PolyData(source_transformed_np)
    mesh = pc.delaunay_3d(alpha=alpha)

    # Plot the mesh
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='white', edge_color='black', show_edges=True, opacity=0.5)
    plotter.show_grid()
    plotter.show()
    


def transform_cone_point_cloud(points, n=0):
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    pca = PCA(n_components=3)
    pca.fit(centered_points)
    principal_axes = pca.components_
    
    z_axis = np.array([0, 0, 1])
    
    # Find the principal axis most aligned with the z-axis
    dot_products = [np.abs(np.dot(axis, z_axis)) for axis in principal_axes]
    most_aligned_axis_idx = np.argmax(dot_products)
    axis_to_align = principal_axes[n]
    
    # Compute the rotation matrix
    angle = np.arccos(np.dot(axis_to_align, z_axis) / (np.linalg.norm(axis_to_align) * np.linalg.norm(z_axis)))
    rotation_axis = np.cross(axis_to_align, z_axis)
    if np.linalg.norm(rotation_axis) < 1e-10:
        rotation_axis = np.array([1, 0, 0])  # Handle the case where the vectors are parallel
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix = np.array([
        [cos_angle + rotation_axis[0]**2 * (1 - cos_angle),
         rotation_axis[0] * rotation_axis[1] * (1 - cos_angle) - rotation_axis[2] * sin_angle,
         rotation_axis[0] * rotation_axis[2] * (1 - cos_angle) + rotation_axis[1] * sin_angle],
        [rotation_axis[1] * rotation_axis[0] * (1 - cos_angle) + rotation_axis[2] * sin_angle,
         cos_angle + rotation_axis[1]**2 * (1 - cos_angle),
         rotation_axis[1] * rotation_axis[2] * (1 - cos_angle) - rotation_axis[0] * sin_angle],
        [rotation_axis[2] * rotation_axis[0] * (1 - cos_angle) - rotation_axis[1] * sin_angle,
         rotation_axis[2] * rotation_axis[1] * (1 - cos_angle) + rotation_axis[0] * sin_angle,
         cos_angle + rotation_axis[2]**2 * (1 - cos_angle)]
    ])
    
    rotated_points = np.dot(centered_points, rotation_matrix.T)
    
    # Translate so that the base of the cone is in the xy-plane
    min_z = np.min(rotated_points[:, 2])
    translated_points = rotated_points - np.array([0, 0, min_z])
    
    return translated_points


zs3=[]
ws3=[]

import pyvista as pv
import matplotlib.pyplot as plt
#f'H:/test_unet/isolated_sensilla_point_clouds/chamber1/c3/cropped_8.ply'

def calc_width(path, num_slices=500, alpha=200, n=0):
    
    pcd_cropped = o3d.io.read_point_cloud(path)
    o3d.visualization.draw_geometries([pcd_cropped])
    points=np.asarray(pcd_cropped.points)
    tp = transform_cone_point_cloud(points, n) 

    pcd_transformed = o3d.geometry.PointCloud()
    pcd_transformed.points = o3d.utility.Vector3dVector(tp)
    o3d.visualization.draw_geometries([pcd_transformed])
    points2=np.asarray(pcd_transformed.points)


    target_mesh = pv.PolyData(points2)
    # Plot the mesh
    mesh = target_mesh.delaunay_3d(alpha=200)


    # Plot the mesh
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='white', edge_color='black', show_edges=True, opacity=0.5)
    plotter.show_grid()
    plotter.show()
    

    num_slices = num_slices  # You can adjust this based on your resolution needs
    z_min, z_max = mesh.bounds[4], mesh.bounds[5]
    z_values = np.linspace(z_min, z_max, num_slices)

    widths = []

    for z in z_values:
        # Create a plane at each z value
        plane = pv.Plane(center=(0, 0, z), direction=(0, 0, 1), i_size=1000, j_size=1000)
        # Slice the mesh with the plane
        slice_ = mesh.slice(normal=(0, 0, 1), origin=(0, 0, z))
        
        if slice_.n_points > 0:  # Ensure the slice has points
            # Get the x and y coordinates of the points in the slice
            points = slice_.points
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            
            # Calculate the width as the distance between the furthest points
            max_width = max(np.ptp(x_coords), np.ptp(y_coords))
            widths.append(max_width)
        else:
            # If no points, append 0 as the width
            widths.append(0)

# Plot the widths as a function of the z-axis


    plt.figure()
    plt.plot(z_values, widths, label="Width along z-axis")
    plt.xlabel("Z-axis position")
    plt.ylabel("Width")
    plt.legend()
    plt.show()
    return(z_values,widths, points2)




### for chamber 1

path=f'F:/test_unet/isolated_sensilla_point_clouds/chamber1/c1/cropped_1.ply'
ch1_1 = calc_width(path, 200,200,0)



pcd_source = o3d.io.read_point_cloud(f'F:/test_unet/isolated_sensilla_point_clouds/chamber1/c1/cropped_4.ply')
o3d.visualization.draw_geometries([pcd_source])
# Transform the cropped point cloud
points_source = np.asarray(pcd_source.points)


ch1_target = reg_3d(points_source, ch1_1[2],voxel_size = 30) 
plot_transformed_mesh(ch1_target, alpha=300)

transformed_sensilla = o3d.geometry.PointCloud()
transformed_sensilla.points = o3d.utility.Vector3dVector(ch1_target)
# Save the point cloud to a PLY file
o3d.io.write_point_cloud('H:/test_unet/isolated_sensilla_point_clouds/chamber1/aligned_sensilla/c1/transformed_4.ply', transformed_sensilla)
    

    
### for chamber 2

pcd_source = o3d.io.read_point_cloud(f'H:/test_unet/isolated_sensilla_point_clouds/chamber1/cropped_1.ply')
o3d.visualization.draw_geometries([pcd_source])
# Transform the cropped point cloud
points_source = np.asarray(pcd_source.points)


base_radius = 1000
height = 1700
num_points = 30000

# Generate the point cloud
cone_pcd = create_cone_point_cloud(base_radius, height, num_points)

# Visualize the point cloud
o3d.visualization.draw_geometries([cone_pcd])
points_target = np.asarray(cone_pcd.points)
ch2_1 = reg_3d(points_source, points_target,voxel_size =15) 
plot_transformed_mesh(ch2_1, alpha=300)


pcd_source = o3d.io.read_point_cloud(f'H:/test_unet/isolated_sensilla_point_clouds/chamber1/cropped_5.ply')
o3d.visualization.draw_geometries([pcd_source])
# Transform the cropped point cloud
points_source = np.asarray(pcd_source.points)

ch2_target = reg_3d(points_source, ch2_1,voxel_size = 30) 
plot_transformed_mesh(ch2_target, alpha=300)

transformed_sensilla = o3d.geometry.PointCloud()
transformed_sensilla.points = o3d.utility.Vector3dVector(ch2_target)
# Save the point cloud to a PLY file
o3d.io.write_point_cloud('H:/test_unet/isolated_sensilla_point_clouds/chamber1/aligned_sensilla/c2/transformed_5.ply', transformed_sensilla)
  

### for chamber 3

# chamber1 sensilla used as target for alignment works well here
#  use that to align the first sensilla of chamber three and alingn the rest with the result obtained

#path=f'H:/test_unet/isolated_sensilla_point_clouds/chamber1/c3/cropped_6.ply'
#ch3_1 = calc_width(path, 200,200,0)



pcd_source = o3d.io.read_point_cloud(f'H:/test_unet/isolated_sensilla_point_clouds/chamber1/c3/cropped_6.ply')
o3d.visualization.draw_geometries([pcd_source])
# Transform the cropped point cloud
points_source = np.asarray(pcd_source.points)

base_radius = 1300
height = 4500
num_points = 30000

# Generate the point cloud
cone_pcd = create_cone_point_cloud(base_radius, height, num_points)

# Visualize the point cloud
o3d.visualization.draw_geometries([cone_pcd])
points_target = np.asarray(cone_pcd.points)
ch3_1 = reg_3d(points_source, points_target,voxel_size =20) 
plot_transformed_mesh(ch3_1, alpha=300)


pcd_source = o3d.io.read_point_cloud(f'H:/test_unet/isolated_sensilla_point_clouds/chamber1/c3/cropped_9.ply')
o3d.visualization.draw_geometries([pcd_source])
# Transform the cropped point cloud
points_source = np.asarray(pcd_source.points)

ch3_target = reg_3d(points_source, ch3_1,voxel_size = 30) 
plot_transformed_mesh(ch3_target, alpha=300)

transformed_sensilla = o3d.geometry.PointCloud()
transformed_sensilla.points = o3d.utility.Vector3dVector(ch3_target)
# Save the point cloud to a PLY file
o3d.io.write_point_cloud('H:/test_unet/isolated_sensilla_point_clouds/chamber1/aligned_sensilla/c3/transformed_9.ply', transformed_sensilla)
  
######################

#### width measurement 

#### chamber 1

import os
files = os.listdir('H:/test_unet/isolated_sensilla_point_clouds/chamber1/aligned_sensilla/c1')
width_ch1 =[]
height_ch1 =[]

for i in range (0,len(files)):
    pcd = o3d.io.read_point_cloud(f'H:/test_unet/isolated_sensilla_point_clouds/chamber1/aligned_sensilla/c1/' + str(files[i]))
    o3d.visualization.draw_geometries([pcd])
    points = np.asarray(pcd.points)
    target_mesh = pv.PolyData(points)
    mesh = target_mesh.delaunay_3d(alpha=250)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='white', edge_color='black', show_edges=True, opacity=0.5)
    plotter.show_grid()
    plotter.show() 
    
    num_slices = 500  # You can adjust this based on your resolution needs
    z_min, z_max = mesh.bounds[4], mesh.bounds[5]
    z_values = np.linspace(z_min, z_max, num_slices)

    widths = []

    for z in z_values:
        # Create a plane at each z value
        plane = pv.Plane(center=(0, 0, z), direction=(0, 0, 1), i_size=1000, j_size=1000)
        # Slice the mesh with the plane
        slice_ = mesh.slice(normal=(0, 0, 1), origin=(0, 0, z))
        
        if slice_.n_points > 0:  # Ensure the slice has points
            # Get the x and y coordinates of the points in the slice
            points = slice_.points
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            
            # Calculate the width as the distance between the furthest points
            max_width = max(np.ptp(x_coords), np.ptp(y_coords))
            widths.append(max_width)
        else:
            # If no points, append 0 as the width
            widths.append(0)
    width_ch1.append(widths)
    height_ch1.append(z_values)
    print(i)

# Plot the widths as a function of the z-axis
for i in range(0,len(width_ch1)):
    idx = np.where(width_ch1[i] == np.amax(width_ch1[i]))
    idx=idx[0][0]
    width = width_ch1[i][idx:]
    height = height_ch1[i][idx:]
    height = height - height[0] 
    plt.plot(height, width)
plt.xlabel("Z-axis position")
plt.ylabel("Width")

###### width measurement chamber2 

files = os.listdir('H:/test_unet/isolated_sensilla_point_clouds/chamber1/aligned_sensilla/c2')
width_ch2 =[]
height_ch2 =[]

for i in range (0,len(files)):
    pcd = o3d.io.read_point_cloud(f'H:/test_unet/isolated_sensilla_point_clouds/chamber1/aligned_sensilla/c2/' + str(files[i]))
    o3d.visualization.draw_geometries([pcd])
    points = np.asarray(pcd.points)
    target_mesh = pv.PolyData(points)
    mesh = target_mesh.delaunay_3d(alpha=250)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='white', edge_color='black', show_edges=True, opacity=0.5)
    plotter.show_grid()
    plotter.show() 
    
    num_slices = 500  # You can adjust this based on your resolution needs
    z_min, z_max = mesh.bounds[4], mesh.bounds[5]
    z_values = np.linspace(z_min, z_max, num_slices)

    widths = []

    for z in z_values:
        # Create a plane at each z value
        plane = pv.Plane(center=(0, 0, z), direction=(0, 0, 1), i_size=1000, j_size=1000)
        # Slice the mesh with the plane
        slice_ = mesh.slice(normal=(0, 0, 1), origin=(0, 0, z))
        
        if slice_.n_points > 0:  # Ensure the slice has points
            # Get the x and y coordinates of the points in the slice
            points = slice_.points
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            
            # Calculate the width as the distance between the furthest points
            max_width = max(np.ptp(x_coords), np.ptp(y_coords))
            widths.append(max_width)
        else:
            # If no points, append 0 as the width
            widths.append(0)
    width_ch2.append(widths)
    height_ch2.append(z_values)
    print(i)

# Plot the widths as a function of the z-axis
for i in range(0,len(width_ch2)):
    idx = np.where(width_ch2[i] == np.amax(width_ch2[i]))
    idx=idx[0][0]
    width = width_ch2[i][idx:]
    height = height_ch2[i][idx:]
    height = height - height[0] 
    plt.plot(height, width)
plt.xlabel("Z-axis position")
plt.ylabel("Width")


####################

# width measurement chamber 3

files = os.listdir('H:/test_unet/isolated_sensilla_point_clouds/chamber1/aligned_sensilla/c3')
width_ch3 =[]
height_ch3 =[]

for i in range (0,len(files)):
    pcd = o3d.io.read_point_cloud(f'H:/test_unet/isolated_sensilla_point_clouds/chamber1/aligned_sensilla/c3/' + str(files[i]))
    o3d.visualization.draw_geometries([pcd])
    points = np.asarray(pcd.points)
    target_mesh = pv.PolyData(points)
    mesh = target_mesh.delaunay_3d(alpha=250)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='white', edge_color='black', show_edges=True, opacity=0.5)
    plotter.show_grid()
    plotter.show() 
    
    num_slices = 500  # You can adjust this based on your resolution needs
    z_min, z_max = mesh.bounds[4], mesh.bounds[5]
    z_values = np.linspace(z_min, z_max, num_slices)

    widths = []

    for z in z_values:
        # Create a plane at each z value
        plane = pv.Plane(center=(0, 0, z), direction=(0, 0, 1), i_size=1000, j_size=1000)
        # Slice the mesh with the plane
        slice_ = mesh.slice(normal=(0, 0, 1), origin=(0, 0, z))
        
        if slice_.n_points > 0:  # Ensure the slice has points
            # Get the x and y coordinates of the points in the slice
            points = slice_.points
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            
            # Calculate the width as the distance between the furthest points
            max_width = max(np.ptp(x_coords), np.ptp(y_coords))
            widths.append(max_width)
        else:
            # If no points, append 0 as the width
            widths.append(0)
    width_ch3.append(widths)
    height_ch3.append(z_values)
    print(i)

# Plot the widths as a function of the z-axis
for i in range(0,len(width_ch3)):
    idx = np.where(width_ch3[i] == np.amax(width_ch3[i]))
    idx=idx[0][0]
    width = width_ch3[i][idx:]
    height = height_ch3[i][idx:]
    height = height - height[0] 
    plt.plot(height, width)
plt.xlabel("Z-axis position")
plt.ylabel("Width")
















# Function to calculate FWHM
def calculate_fwhm(x, y):
    max_y = max(y)
    half_max = max_y / 2

    indices = np.where(y >= half_max)[0]
    if len(indices) < 2:
        return None  # Cannot calculate FWHM if there are not enough points
    
    fwhm = x[indices[-1]] - x[indices[0]]
    return fwhm

# Plot the widths and calculate FWHM
for i in range(len(width_ch1)):
    idx = np.where(width_ch1[i] == np.amax(width_ch1[i]))[0][0]
    width = width_ch1[i][idx:]
    height = height_ch1[i][idx:]
    height = height - height[0] 
    plt.plot(height, width)
    
    fwhm = calculate_fwhm(height, width)
    if fwhm is not None:
        plt.axhline(y=max(width) / 2, color='r', linestyle='--')
        plt.title(f'Width vs. Height (FWHM = {fwhm:.2f})')
    else:
        plt.title('Width vs. Height (FWHM could not be determined)')

plt.xlabel("Z-axis position")
plt.ylabel("Width")
plt.show()  
    
    
    
pcd1 = o3d.io.read_point_cloud(f'H:/test_unet/isolated_sensilla_point_clouds/chamber1/aligned_sensilla/c1/transformed_1.ply')
pcd2 = o3d.io.read_point_cloud(f'H:/test_unet/isolated_sensilla_point_clouds/chamber1/aligned_sensilla/c2/transformed_1.ply')
pcd3 = o3d.io.read_point_cloud(f'H:/test_unet/isolated_sensilla_point_clouds/chamber1/aligned_sensilla/c3/transformed_6.ply')
pcd1.paint_uniform_color([1, 0, 0])  # Red
pcd2.paint_uniform_color([0, 1, 0])  # Green
pcd3.paint_uniform_color([0, 0, 1])  # Blue
o3d.visualization.draw_geometries([pcd1,pcd2,pcd3])    




#### calculate and plot width with mean values

from scipy.interpolate import interp1d
import pandas as pd

w_ch1=[]
h_ch1=[]
for i in range(0,len(width_ch1)):
    idx = np.where(width_ch1[i] == np.amax(width_ch1[i]))
    idx=idx[0][0]
    width = width_ch1[i][idx:]
    height = height_ch1[i][idx:]
    height = height - height[0] 
    w_ch1.append(width)
    h_ch1.append(height)
    plt.plot(height, width)
plt.xlabel("Z-axis position")
plt.ylabel("Width")

w_ch1 = [np.array(width) for width in w_ch1]
h_ch1 = [np.array(height) for height in h_ch1]
x=[]
for i in range(0,len(w_ch1)):
    s_val = sorted(set(w_ch1[i]))
    s_val=s_val[1]
    x.append(s_val)
x=np.amax(x)                   



df = pd.DataFrame()
w_concat = np.concatenate((w_ch1))
h_concat = np.concatenate((h_ch1))
df['width'] = w_concat
df['height'] = h_concat

max_width = df.width.max()
width_val =np.arange(x, max_width,10 )

mean_w = []
mean_h = []
for i in range(0, len(width_val)-1):
    d = df[(df['width'] > width_val[i]) & (df['width'] < width_val[i+1])]
    if not d.empty:
        mw = np.mean(d.width)
        mh = np.mean(d.height)
        mean_w.append(mw)
        mean_h.append(mh)
df_mean = pd.DataFrame()
df_mean['width'] = mean_w
df_mean['height'] = mean_h    

df_mean = df_mean.ewm(span=20, adjust=False).mean()  
plt.scatter(df_mean.height, df_mean.width)

for i in range(0,len(w_ch1)):
    plt.plot(h_ch1[i],w_ch1[i], alpha=0.4)
plt.plot(df_mean.height, df_mean.width ,color='black', linewidth=2, label='Mean Width')
plt.xlabel("Height (Z-axis position)")
plt.ylabel("Width")
plt.title("Width vs. Height Profiles and Mean Profile")
plt.legend()
plt.show()


#### chamber 2

w_ch2=[]
h_ch2=[]
for i in range(0,len(width_ch2)):
    idx = np.where(width_ch2[i] == np.amax(width_ch2[i]))
    idx=idx[0][0]
    width = width_ch2[i][idx:]
    height = height_ch2[i][idx:]
    height = height - height[0] 
    w_ch2.append(width)
    h_ch2.append(height)
    plt.plot(height, width)
plt.xlabel("Z-axis position")
plt.ylabel("Width")

w_ch2 = [np.array(width) for width in w_ch2]
h_ch2 = [np.array(height) for height in h_ch2]
x=[]
for i in range(0,len(w_ch2)):
    s_val = sorted(set(w_ch2[i]))
    s_val=s_val[1]
    x.append(s_val)
x=np.amax(x)                   



df = pd.DataFrame()
w_concat = np.concatenate((w_ch2))
h_concat = np.concatenate((h_ch2))
df['width'] = w_concat
df['height'] = h_concat

max_width = df.width.max()
width_val =np.arange(x, max_width,10 )

mean_w = []
mean_h = []
for i in range(0, len(width_val)-1):
    d = df[(df['width'] > width_val[i]) & (df['width'] < width_val[i+1])]
    if not d.empty:
        mw = np.mean(d.width)
        mh = np.mean(d.height)
        mean_w.append(mw)
        mean_h.append(mh)
df_mean2 = pd.DataFrame()
df_mean2['width'] = mean_w
df_mean2['height'] = mean_h    

df_mean2 = df_mean2.ewm(span=20, adjust=False).mean()  
plt.scatter(df_mean2.height, df_mean2.width)

for i in range(0,len(w_ch2)):
    plt.plot(h_ch2[i],w_ch2[i], alpha=0.4)
plt.plot(df_mean2.height, df_mean2.width ,color='black', linewidth=2, label='Mean Width')
plt.xlabel("Height (Z-axis position)")
plt.ylabel("Width")
plt.title("Width vs. Height Profiles and Mean Profile")
plt.legend()
plt.show()

##################### chamner 3

w_ch3=[]
h_ch3=[]
for i in range(0,len(width_ch3)):
    idx = np.where(width_ch3[i] == np.amax(width_ch3[i]))
    idx=idx[0][0]
    width = width_ch3[i][idx:]
    height = height_ch3[i][idx:]
    height = height - height[0] 
    w_ch3.append(width)
    h_ch3.append(height)
    plt.plot(height, width)
plt.xlabel("Z-axis position")
plt.ylabel("Width")

w_ch3 = [np.array(width) for width in w_ch3]
h_ch3 = [np.array(height) for height in h_ch3]
x=[]
for i in range(0,len(w_ch3)):
    s_val = sorted(set(w_ch3[i]))
    s_val=s_val[1]
    x.append(s_val)
x=np.amax(x)                   



df = pd.DataFrame()
w_concat = np.concatenate((w_ch3))
h_concat = np.concatenate((h_ch3))
df['width'] = w_concat
df['height'] = h_concat

max_width = df.width.max()
width_val =np.arange(x, max_width,10 )

mean_w = []
mean_h = []
for i in range(0, len(width_val)-1):
    d = df[(df['width'] > width_val[i]) & (df['width'] < width_val[i+1])]
    if not d.empty:
        mw = np.mean(d.width)
        mh = np.mean(d.height)
        mean_w.append(mw)
        mean_h.append(mh)
df_mean3 = pd.DataFrame()
df_mean3['width'] = mean_w
df_mean3['height'] = mean_h    

df_mean3 = df_mean3.ewm(span=20, adjust=False).mean()  
plt.scatter(df_mean3.height, df_mean3.width)

for i in range(0,len(w_ch3)):
    plt.plot(h_ch3[i],w_ch3[i], alpha=0.4)
plt.plot(df_mean3.height, df_mean3.width ,color='black', linewidth=2, label='Mean Width')
plt.xlabel("Height (Z-axis position)")
plt.ylabel("Width")
plt.title("Width vs. Height Profiles and Mean Profile")
plt.legend()
plt.show()
