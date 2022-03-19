import numpy as np
import cv2
import glob
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image
from matplotlib import pyplot as plt
from camera_calibration import undistort

def create_output(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])

	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')

def downsample_image(image, reduce_factor):
	for i in range(0,reduce_factor):
		#Check if image is color or grayscale
		if len(image.shape) > 2:
			row,col = image.shape[:2]
		else:
			row,col = image.shape

		image = cv2.pyrDown(image, dstsize= (col//2, row // 2))
	return image


def measure_dist_SGBM(img_left, img_right, K_left, K_right, distcoeff_left, distcoeff_right):
    # Get height and width. Note: It assumes that both pictures are the same size. They HAVE to be same size
    h, w = img_left.shape[:2]

    # Get optimal camera matrix for better undistortion
    new_camera_matrix_left, roi_left = cv2.getOptimalNewCameraMatrix(K_left, distcoeff_left, (w, h), 1, (w, h))
    new_camera_matrix_right, roi_eight = cv2.getOptimalNewCameraMatrix(K_right, distcoeff_right, (w, h), 1, (w, h))

    # Undistort images
    img_left_u = undistort(img_left, isLeft=True)
    img_right_u = undistort(img_right, isLeft=False)

    win_size = 5
    min_disp = -33
    max_disp = 111
    num_disp = max_disp - min_disp  # Needs to be divisible by 16

    stereo = cv2.StereoSGBM_create(
                                   minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=5,
                                   uniquenessRatio=0,
                                   speckleWindowSize=0,
                                   speckleRange=0,
                                   disp12MaxDiff=64,
                                   P1=8 * 3 * win_size ** 2,  # 8*3*win_size**2,
                                   P2=32 * 3 * win_size ** 2)  # 32*3*win_size**2)

    print("\nComputing the disparity  map...")
    disparity_map = stereo.compute(img_left_u, img_right_u)

    # Show disparity map before generating 3D cloud to verify that point cloud will be usable.
    plt.imshow(disparity_map, 'gray')
    plt.show()

    #focal length in meters
    focal_length = 0.016

    Q2 = np.float32([[1, 0, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, 0.016, 0],  # Focal length multiplication obtained experimentally.
                     [0, 0, 0, 1]])

    # Reproject points into 3D
    points_3D = cv2.reprojectImageTo3D(disparity_map, Q2)
    # Get color points
    colors = cv2.cvtColor(img_left_u, cv2.COLOR_BGR2RGB)

    # Get rid of points with value 0 (i.e no depth)
    mask_map = disparity_map > disparity_map.min()

    # Mask colors and points.
    output_points = points_3D[mask_map]
    output_colors = colors[mask_map]

    # Define name for output file
    # can open this file with MeshLab to see the 3D reconstruction
    output_file = 'reconstructed.ply'

    # Generate point cloud
    print("\n Creating the output file... \n")
    create_output(output_points, output_colors, output_file)


#can go fundamental --> essential --> rotation and translation matrices (4 solutions but only one will be valid) (this is called pose recovery)
# --> projection matrix --> 3D coordinates
if __name__ == '__main__':
    print("in measure_dist")

    K_left = np.load('left_intrinsic_matrix.npy')
    distcoeff_left = np.load('left_dist_coeff.npy')
    K_right = np.load('right_intrinsic_matrix.npy')
    distcoeff_right = np.load('right_dist_coeff.npy')
    #Specify image paths
    img_path1 = 'frame_left_sarah.png'
    img_path2 = 'frame_right_sarah.png'
    #Load pictures
    img_left = cv2.imread(img_path1)
    img_right = cv2.imread(img_path2)

    #run SGBM open cv algorithm
    measure_dist_SGBM(img_left, img_right, K_left, K_right, distcoeff_left, distcoeff_right)