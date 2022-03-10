#class to measure the distance once the pixel coordinates of the shot put are known
import cv2
import numpy as np
import math
import random

def three_d_dist_formula(p0, p1):
    return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 + (p0[2]-p1[2]) ** 2)

def normalize_coordinates(arr, img, K):
    #print("in normalize")
    num_rows, num_cols = img.shape[:2]
    new_arr = []
    for val in arr:
        val = np.concatenate((val, [1]))
        new_val = np.linalg.inv(K) @ val
        #print(new_val)
        #x = float(val[0]) / float(num_cols - 1.0)
        #y = float(float(val[1]) / float(num_rows - 1.0))
        new_arr.append(new_val)

    #print(np.array(new_arr))
    return np.array(new_arr)

def normalize_coordinate(arr, img, K):
    print("in normalize")
    num_rows, num_cols = img.shape[:2]
    new_arr = []
    arr = np.concatenate((arr, [1]))
    new_val = np.linalg.inv(K) @ arr
    print(new_val)
    #x = float(val[0]) / float(num_cols - 1.0)
    #y = float(float(val[1]) / float(num_rows - 1.0))
    new_arr.append(new_val)

    #print(np.array(new_arr))
    return np.array(new_arr)



def get_frames():
    frame_left = cv2.imread('frame_left.png')
    frame_right = cv2.imread('frame_right.png')
    #frame_left = undistort(frame_left, isLeft=True)
    #frame_right = undistort(frame_right, isLeft=False)
    return frame_left, frame_right


def compute_fund_matrix(frame_left, frame_right, eight_coords_left, eight_coords_right):
    F, mask = cv2.findFundamentalMat(eight_coords_left, eight_coords_right, cv2.FM_LMEDS)
    return F, mask


#Source: https://gist.github.com/davegreenwood/e1d2227d08e24cc4e353d95d0c18c914
def triangulate_nviews(P, ip):
    """
    Triangulate a point visible in n camera views.
    P is a list of camera projection matrices.
    ip is a list of homogenised image points. eg [ [x, y, 1], [x, y, 1] ], OR,
    ip is a 2d array - shape nx3 - [ [x, y, 1], [x, y, 1] ]
    len of ip must be the same as len of P
    """
    if not len(ip) == len(P):
        raise ValueError('Number of points and number of cameras not equal.')
    n = len(P)
    M = np.zeros([3*n, 4+n])
    for i, (x, p) in enumerate(zip(ip, P)):
        M[3*i:3*i+3, :4] = p
        M[3*i:3*i+3, 4+i] = -x
    V = np.linalg.svd(M)[-1]
    X = V[-1, :4]
    print(X)
    return X / X[3]

def draw_matching_points(frame_left, frame_right, known_pixel_coords_left, known_pixel_coords_right):
    frame_left_copy = frame_left.copy()
    frame_right_copy = frame_right.copy()
    stop_box_coords_left_h = np.array((1017, 1200, 1))
    stop_box_coords_right_h = np.array((781, 960, 1))
    shot_put_coords_left_h = np.array((833, 967, 1))
    shot_put_coords_right_h = np.array((965, 740, 1))
    cv2.circle(frame_left_copy, (shot_put_coords_left_h[0], shot_put_coords_left_h[1]), 20, (0, 0, 0), 5)
    cv2.circle(frame_right_copy, (shot_put_coords_right_h[0], shot_put_coords_right_h[1]), 20, (0, 0, 0), 5)
    cv2.circle(frame_left_copy, (stop_box_coords_left_h[0], stop_box_coords_left_h[1]), 20, (0, 0, 0), 5)
    cv2.circle(frame_right_copy, (stop_box_coords_right_h[0], stop_box_coords_right_h[1]), 20, (0, 0, 0), 5)

    for val in zip(known_pixel_coords_left, known_pixel_coords_right):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        cv2.circle(frame_left_copy, (val[0][0], val[0][1]), 5, (b, g, r), 5)
        cv2.circle(frame_right_copy, (val[1][0], val[1][1]), 5, (b, g, r), 5)
    cv2.imwrite("frame_left_points_labeled.png", frame_left_copy)
    cv2.imwrite("frame_right_points_labeled.png", frame_right_copy)

# essential = K'.T * fundamental * K where K is intrinsic matrix left camera, K' is intrinsic matrix right camera
def find_essential_matrix(F, K_left, K_right):
    essential_matrix = np.transpose(K_right) @ F @ K_left
    return essential_matrix

def find_essential_matrix_opencv(ptsleft, ptsright, K_left, distcoeffleft, K_right, distcoeffright):
    pts_left, pts_right = np.ascontiguousarray(ptsleft, np.float32), np.ascontiguousarray(ptsright, np.float32)
    E, mask = cv2.findEssentialMat(points1=pts_left, points2=pts_right, cameraMatrix1=K_left, distCoeffs1=distcoeffleft, cameraMatrix2= K_right, distCoeffs2=distcoeffright, method=cv2.RANSAC, threshold=0.0001)
    return E, mask

def find_F_sift_and_lowes(frame_left, frame_right):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(frame_left, None)
    kp2, des2 = sift.detectAndCompute(frame_right, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    #confidence default = 0.99
    F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_LMEDS)

    return F

def check_F_error(F, known_coords_left_h, known_coords_right_h):
    total = 0
    num_coords = len(known_coords_left_h)
    for coords in zip(known_coords_left_h, known_coords_right_h):
        left_coord_h = coords[0]
        right_coord_h = coords[1]
        total += abs(right_coord_h @ F @ left_coord_h.T)
    avg_F_error = total / num_coords
    return avg_F_error


def check_E_error(E, known_coords_left_n, known_coords_right_n):
    total = 0
    num_coords = len(known_coords_left_n)
    for coords in zip(known_coords_left_n, known_coords_right_n):
        left_coord_h = np.concatenate((coords[0],[1]))
        right_coord_h = np.concatenate((coords[1], [1]))
        total += abs(right_coord_h @ E @ left_coord_h.T)
    avg_E_error = total / num_coords
    return avg_E_error

def find_correct_projection_matrix(P_left, P_right_1, P_right_2, P_right_3, P_right_4, known_coords_left_h, known_coords_right_h):
    # triangulate with each projection matrix, then run it back with the projection matrix to get camera coords
    matrices_dict = {
        1: 0,
        2: 0,
        3: 0,
        4: 0
    }
    #get global coordinate through triangulation, then run it back to get pixel coordinate, check depth
    for coord in zip(known_coords_left_h, known_coords_right_h):
        global_1 = triangulate_nviews([P_left, P_right_1], [coord[0], coord[1]])
        x1_left = P_left @ global_1
        x1_right = P_right_1 @ global_1
        if x1_left[2] > 0: matrices_dict[1] = matrices_dict[1] + 1
        if x1_right[2] > 0: matrices_dict[1] = matrices_dict[1] + 1

        global_2 = triangulate_nviews([P_left, P_right_2], [coord[0], coord[1]])
        x2_left = P_left @ global_2
        x2_right = P_right_2 @ global_2
        if x2_left[2] > 0: matrices_dict[2] = matrices_dict[2] + 1
        if x2_right[2] > 0: matrices_dict[2] = matrices_dict[2] + 1

        global_3 = triangulate_nviews([P_left, P_right_3], [coord[0], coord[1]])
        x3_left = P_left @ global_3
        x3_right = P_right_1 @ global_3
        if x3_left[2] > 0: matrices_dict[3] = matrices_dict[3] + 1
        if x3_right[2] > 0: matrices_dict[3] = matrices_dict[3] + 1

        global_4 = triangulate_nviews([P_left, P_right_4], [coord[0], coord[1]])
        x4_left = P_left @ global_4
        x4_right = P_right_1 @ global_4
        if x4_left[2] > 0: matrices_dict[4] = matrices_dict[4] + 1
        if x4_right[2] > 0: matrices_dict[4] = matrices_dict[4] + 1

    max_key = max(matrices_dict, key=matrices_dict.get)
    if max_key == 1: final_right_proj_matrix = P_right_1
    elif max_key == 2: final_right_proj_matrix = P_right_2
    elif max_key == 3: final_right_proj_matrix = P_right_3
    elif max_key == 4: final_right_proj_matrix = P_right_4
    return final_right_proj_matrix

#can go fundamental --> essential --> rotation and translation matrices --> projection matrix --> 3D coordinates
if __name__ == '__main__':
   print("in measure_dist")

   #get frames and click for corresponding points here
   frame_left, frame_right = get_frames()

   #get intrinsic matrices and distortion coefficients from calibration that has already occurred
   K_left = np.load('left_intrinsic_matrix.npy')
   K_right = np.load('right_intrinsic_matrix.npy')
   distcoeff_left = np.load('left_dist_coeff.npy')
   distcoeff_right = np.load('right_dist_coeff.npy')
   # print("left intrsinic matrix: ")
   # print(K_left)
   # print("right intrisic matrix: ")
   # print(K_right)

   #list of known pixel coordinatees, see labeled images to verify they match up
   known_pixel_coords_left = np.array([(1016, 1198), (384, 851), (1028, 925), (587, 702), (932, 757), (552, 727), (983, 545), (1124, 574), (1243, 601), (1342, 565), (1488, 517), (727, 411), (635,748), (814,771), (968,790)])
   known_pixel_coords_right = np.array([(779, 960), (707, 668), (1274, 702), (954, 512), (1267, 530), (930, 543), (1292, 268), (1443, 283), (1580, 292), (1673, 229), (1820, 129), (981, 164), (1000,552), (1154,559), (1307,564)])

   #coordinates circled in black are the coordinates that will be measured
   draw_matching_points(frame_left, frame_right, known_pixel_coords_left, known_pixel_coords_right)

   #homogenize coordinates in case necessary
   new_arr = []
   for val in known_pixel_coords_left:
       new_val = np.concatenate((val, [1]))
       new_arr.append(new_val)
   known_pixel_coords_left_h = np.array((new_arr), dtype=np.float64)

   new_arr = []
   for val in known_pixel_coords_right:
       new_val = np.concatenate((val, [1]))
       new_arr.append(new_val)
   known_pixel_coords_right_h = np.array((new_arr), dtype=np.float64)

   #normalize known coordinates in case necessary
   pts_left, pts_right = np.ascontiguousarray(known_pixel_coords_left, np.float32), np.ascontiguousarray(
       known_pixel_coords_right, np.float32)
   known_pixel_coords_left_n = cv2.undistortPoints(np.expand_dims(pts_left, axis=1), cameraMatrix=K_left,
                                         distCoeffs=distcoeff_left)
   known_pixel_coords_right_n = cv2.undistortPoints(np.expand_dims(pts_right, axis=1), cameraMatrix=K_right,
                                         distCoeffs=distcoeff_right)

   #STEP 1 - compute 3x3 fundamental matrix using SIFT and LMEDS
   F = find_F_sift_and_lowes(frame_left, frame_right)
   error = check_F_error(F, known_pixel_coords_left_h, known_pixel_coords_right_h)
   print("Fundamental Matrix:")
   print(F)
   print("F error = ", error)


   # STEP 2 - compute essential matrix
   E = find_essential_matrix(F, K_left, K_right)
   error = check_E_error(E, known_pixel_coords_left_n[0], known_pixel_coords_right_n[0])
   print("E error = ", error)


   #STEP 3 - extract rotation and translation matrices

   R1, R2, t = cv2.decomposeEssentialMat(E)
   print("results from decompose:")
   print("R1:")
   print(R1)
   print("R2:")
   print(R2)
   print("t:")
   print(t)


   #STEP 4: Construct four possible projection matrix solutions
   P_right_1 = K_right @ np.concatenate((R1, t), axis=1)
   P_right_2 = K_right @ np.concatenate((R1, -1*t), axis=1)
   P_right_3 = K_right @ np.concatenate((R2, t), axis=1)
   P_right_4 = K_right @ np.concatenate((R2, -1*t), axis=1)

   print("P right 1:")
   print(P_right_1)
   print("P right 2:")
   print(P_right_2)
   print("P right 3:")
   print(P_right_3)
   print("P right 4:")
   print(P_right_4)

   #STEP 5 - figure out which solution is correct

   #P_left is identity matrix plus column of 0's since origin is left camera
   P_left = K_left @ np.append(np.identity(3), np.zeros((3,1)), axis=1)
   #print("P left:")
   #print(P_left)
   P_right = find_correct_projection_matrix(P_left, P_right_1, P_right_2, P_right_3, P_right_4, known_pixel_coords_left_h, known_pixel_coords_right_h)
   print("P right:")
   print(P_right)

   #STEP 6 - get global coordinates using correct projection matrix

   stop_box_coords_left_h = np.array((1017, 1200, 1))
   stop_box_coords_right_h = np.array((781, 960, 1))
   shot_put_coords_left_h = np.array((833, 967, 1))
   shot_put_coords_right_h = np.array((965, 740, 1))

   global_stopbox = triangulate_nviews([P_left, P_right], [stop_box_coords_left_h, stop_box_coords_right_h])
   global_shotput = triangulate_nviews([P_left, P_right], [shot_put_coords_left_h, shot_put_coords_right_h])

   #in this global coordinates, the shot put depth should be larger than the stop box depth as it is further away from camera
   print("global stop box coords:", global_stopbox)
   print("global shot put coords: ", global_shotput)


   #STEP 7 - calculate distance between the two
   dist = three_d_dist_formula((global_stopbox[0], global_stopbox[1], global_stopbox[2]), (global_shotput[0], global_shotput[1], global_shotput[2]))
   print("expecting measurement around 5.78 m, 578 cm, 5780 mm")
   print("3d dist between shot put and stop box: ", dist)











