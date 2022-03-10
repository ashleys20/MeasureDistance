import numpy as np
import cv2

#tested optimal num boards
num_boards = 15
num_squares_width = 9
num_squares_height = 6

#Size of square (in mm)
square_dimension = 25

#Image dimensions
image_size = (3648, 2736)

#Crop mask
# A value of 0 will crop out all the black pixels.  This will result in a loss of some actual pixels.
# A value of 1 will leave in all the pixels.  This maybe useful if there is some important information
# at the corners.  Ideally, you will have to tweak this to see what works for you.
#This means all pixels will be geometrically transformed to be included
crop = 0.3


#values to grab correct images from local computer
image_folder_left = 'Checkerboard_LeftCamera/'
image_folder_right = 'Checkerboard_RightCamera/'
left_camera = "LeftCamera"
right_camera = "RightCamera"


def ImageProcessing(n_boards, num_squares_width, num_squares_height, square_dimension, isLeft):

    #will store true object points
    object_pts = []
    #will store points determined by calibration images
    calib_pts = []
    #terminate at either max iterations or when corner moves by less than epsilon
    #second value is max count, third value is min epsilon val
    max_ct = 30
    epsilon = 0.05
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_ct, epsilon)

    # prepare object points based on the actual dimensions of the calibration board
    objp = np.zeros((num_squares_height * num_squares_width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:(num_squares_width * square_dimension):square_dimension, 0:(num_squares_height * square_dimension):square_dimension].T.reshape(-1, 2)

    # Loop through the images.  Find checkerboard corners and save the data to calib_pts.
    for i in range(1, n_boards + 1):

        # Loading images
        print('Loading... Checkerboard_' + str(i))
        path = (image_folder_left if isLeft else image_folder_right) + 'Checkerboard_' + str(i) + '_' + (left_camera if isLeft else right_camera) + '.JPG'
        print(path)
        image = cv2.imread(path)
        print(i)

        # Converting to grayscale
        grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Find chessboard corners
        found, corners = cv2.findChessboardCorners(grey_image, (num_squares_width, num_squares_height),
                                                   cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        print(found)

        if found == True:
            # Add the "true" checkerboard corners
            object_pts.append(objp)

            # Improve the accuracy of the checkerboard corners found in the image and save them to the calib_pts variable.
            window_size = (40, 40) if isLeft else (34,34)
            cv2.cornerSubPix(grey_image, corners, window_size, (-1, -1), criteria)
            calib_pts.append(corners)

            # Draw chessboard corners
            cv2.drawChessboardCorners(image, (num_squares_width, num_squares_height), corners, found)

            # Show the image with the chessboard corners overlaid.
            image_name = "Corners_" + str(i) + (left_camera if isLeft else right_camera) + '.jpg'
            cv2.imwrite(image_name, image)

    print('')
    print('Finished processes images.')

    # Calibrate the camera
    print('Running Calibrations...')
    print(' ')
    flags = cv2.CALIB_FIX_K6
    #cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_FIX_K3 + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_PRINCIPAL_POINT
    ret, intrinsic_matrix, distCoeff, rvecs, tvecs = cv2.calibrateCamera(object_pts, calib_pts, grey_image.shape[::-1], None, None, flags=flags)

    # Save matrices
    #ideally (2,0) should be half the image width and (2,1) should be half the image height
    print('Intrinsic Matrix: ')
    print(str(intrinsic_matrix))
    print(' ')
    print('Distortion Coefficients: ')
    print(str(distCoeff))
    print(' ')

    # Save data
    print('Saving data file...')
    np.save(('left' if isLeft else 'right') + '_dist_coeff.npy', distCoeff)
    np.save(('left' if isLeft else 'right') + '_intrinsic_matrix.npy', intrinsic_matrix)
    print('Calibration complete')

    # Calculate the total reprojection error.  The closer to zero the better.
    tot_error = 0
    for i in range(len(object_pts)):
        imgpoints2, _ = cv2.projectPoints(object_pts[i], rvecs[i], tvecs[i], intrinsic_matrix, distCoeff)
        error = cv2.norm(calib_pts[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        tot_error += error

    #calculated by comparing true checkerboard points to the undistorted image points
    print("total reprojection error: ", tot_error / len(object_pts))

    # Undistort Images
    #
    #     # Scale the images and create a rectification map.
    newMat, ROI = cv2.getOptimalNewCameraMatrix(intrinsic_matrix, distCoeff, image_size, alpha=crop,
                                                centerPrincipalPoint=1)
    mapx, mapy = cv2.initUndistortRectifyMap(intrinsic_matrix, distCoeff, None, newMat, image_size, m1type=cv2.CV_32FC1)

    for i in range(1, n_boards + 1):
        # Loading images
        print('Loading... Checkerboard_' + str(i))
        path = (image_folder_left if isLeft else image_folder_right) + 'Checkerboard_' + str(i) + '_' + (
            left_camera if isLeft else right_camera) + '.JPG'
        image = cv2.imread(path)

        # undistort
        dst = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

        #undistorted_img = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        image_name = "Undistorted_" + str(i) + (left_camera if isLeft else right_camera) + '.jpg'
        cv2.imwrite(image_name, dst)

#function to undistort image after calibration variables are obtained
def undistort(frame, isLeft):
    intrinsic_matrix = np.load('left_intrinsic_matrix.npy') if isLeft else np.load('right_intrinsic_matrix.npy')
    distCoeff = np.load('left_dist_coeff.npy') if isLeft else np.load('right_dist_coeff.npy')

    if intrinsic_matrix is None or distCoeff is None:
        print("issue with calibration values")
        return

    undistorted_image = cv2.undistort(frame, intrinsic_matrix, distCoeff, None, intrinsic_matrix)
    return undistorted_image

    #BELOW IS CODE TO SCALE - this makes image smaller
    # Scale the images and create a rectification map.
    # newMat, ROI = cv2.getOptimalNewCameraMatrix(intrinsic_matrix, distCoeff, image_size, alpha=crop,centerPrincipalPoint=1)
    #mapx, mapy = cv2.initUndistortRectifyMap(intrinsic_matrix, distCoeff, None, newMat, image_size, m1type=cv2.CV_32FC1)

    #undistorted_image = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    #return undistorted_image

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    isLeft = True
    ImageProcessing(num_boards, num_squares_width, num_squares_height, square_dimension, isLeft)
    isLeft = False
    ImageProcessing(num_boards, num_squares_width, num_squares_height, square_dimension, isLeft)




# See PyCharm help at https://www.jetbrains.com/help/pycharm/