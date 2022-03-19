#class to measure the distance once the pixel coordinates of the shot put are known
import cv2
import numpy as np
import math
import random
from camera_calibration import undistort

def three_d_dist_formula(p0, p1):
    return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 + (p0[2]-p1[2]) ** 2)

def two_d_dist_formula(p0, p1):
    return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)


def get_frames():
    #can also test with frames from Sarah
    frame_left = cv2.imread('frame_left_brian.png')
    frame_right = cv2.imread('frame_right_brian.png')
    frame_left = undistort(frame_left, isLeft=True)
    frame_right = undistort(frame_right, isLeft=False)
    return frame_left, frame_right

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
    #print(X)
    return X / X[3]

def draw_matching_points(frame_left, frame_right, known_pixel_coords_left, known_pixel_coords_right):
    frame_left_copy = frame_left.copy()
    frame_right_copy = frame_right.copy()
    stop_box_coords_left_h = np.array((1017, 1200, 1))
    stop_box_coords_right_h = np.array((781, 960, 1))
    shot_put_coords_left_h = np.array((833, 967, 1))
    shot_put_coords_right_h = np.array((965, 740, 1))
    camera_center_left = (965, 721)
    camera_center_right = (948, 728)
    cv2.circle(frame_left_copy, camera_center_left, 20, (255,255,255), 5)
    cv2.circle(frame_right_copy, camera_center_right, 20, (255, 255, 255), 5)
    #cv2.circle(frame_left_copy, (shot_put_coords_left_h[0], shot_put_coords_left_h[1]), 20, (0, 0, 0), 5)
    #cv2.circle(frame_right_copy, (shot_put_coords_right_h[0], shot_put_coords_right_h[1]), 20, (0, 0, 0), 5)
    #cv2.circle(frame_left_copy, (stop_box_coords_left_h[0], stop_box_coords_left_h[1]), 20, (0, 0, 0), 5)
    #cv2.circle(frame_right_copy, (stop_box_coords_right_h[0], stop_box_coords_right_h[1]), 20, (0, 0, 0), 5)

    for val in zip(known_pixel_coords_left, known_pixel_coords_right):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        cv2.circle(frame_left_copy, (val[0][0], val[0][1]), 5, (b, g, r), 5)
        cv2.circle(frame_right_copy, (val[1][0], val[1][1]), 5, (b, g, r), 5)
    cv2.imwrite("frame_left_points_labeled.png", frame_left_copy)
    cv2.imwrite("frame_right_points_labeled.png", frame_right_copy)

def homogenize_coords(known_pixel_coords_left, known_pixel_coords_right):
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

    return known_pixel_coords_left_h, known_pixel_coords_right_h

# essential = K'.T * fundamental * K where K is intrinsic matrix left camera, K' is intrinsic matrix right camera
def find_essential_matrix(F, K_left, K_right):
    essential_matrix = np.transpose(K_right) @ F @ K_left
    return essential_matrix

def find_essential_matrix_opencv(ptsleft, ptsright, K_left, distcoeffleft, K_right, distcoeffright):
    pts_left, pts_right = np.ascontiguousarray(ptsleft, np.float32), np.ascontiguousarray(ptsright, np.float32)
    E, mask = cv2.findEssentialMat(points1=pts_left, points2=pts_right, cameraMatrix1=K_left, distCoeffs1=distcoeffleft, cameraMatrix2= K_right, distCoeffs2=distcoeffright, method=cv2.RANSAC)
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
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    return F, pts1, pts2

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

def decompose_E(E):
    # first do SVD
    U, S, V = np.linalg.svd(E)
    # there will be slight error because the bottom right value of S is not exactly zero even though it is very close
    # want S[0][0] and S[1][1] to be equal
    S = np.diag(S)
    V_transpose = np.transpose(V)
    #print("U:", U)
    #print("S:", S)
    #print("V:", V)
    #print("V transpose:", V_transpose)

    # construct W, W_t and Z
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    W_transpose = np.transpose(W)
    Z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])

    # translation = U*Z*U_T or -(U*Z*U_T)
    # rotation = U*W*V_T or U*W_T*V_T
    rot_solution_1 = U @ W @ V_transpose
    rot_solution_2 = U @ W_transpose @ V_transpose

    # getting translation vectors, which is the last column of U
    u = np.dot(U, np.array([[0], [0], [1]], order='F'))

    return rot_solution_1, rot_solution_2, u


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
        #if global_1[2] > 0: matrices_dict[1] = matrices_dict[1] + 1


        global_2 = triangulate_nviews([P_left, P_right_2], [coord[0], coord[1]])
        x2_left = P_left @ global_2
        x2_right = P_right_2 @ global_2
        if x2_left[2] > 0: matrices_dict[2] = matrices_dict[2] + 1
        if x2_right[2] > 0: matrices_dict[2] = matrices_dict[2] + 1
        #if global_2[2] > 0: matrices_dict[2] = matrices_dict[2] + 1

        global_3 = triangulate_nviews([P_left, P_right_3], [coord[0], coord[1]])
        x3_left = P_left @ global_3
        x3_right = P_right_1 @ global_3
        if x3_left[2] > 0: matrices_dict[3] = matrices_dict[3] + 1
        if x3_right[2] > 0: matrices_dict[3] = matrices_dict[3] + 1
        #if global_3[2] > 0: matrices_dict[3] = matrices_dict[3] + 1

        global_4 = triangulate_nviews([P_left, P_right_4], [coord[0], coord[1]])
        x4_left = P_left @ global_4
        x4_right = P_right_1 @ global_4
        if x4_left[2] > 0: matrices_dict[4] = matrices_dict[4] + 1
        if x4_right[2] > 0: matrices_dict[4] = matrices_dict[4] + 1
        #if global_4[2] > 0: matrices_dict[4] = matrices_dict[4] + 1

    #the projection matrices with the most coordinates with positive depth is the chosen matrix
    max_key = max(matrices_dict, key=matrices_dict.get)
    print("count of positive depth for each possible projection matrix:")
    print(matrices_dict)
    if max_key == 1: final_right_proj_matrix = P_right_1
    elif max_key == 2: final_right_proj_matrix = P_right_2
    elif max_key == 3: final_right_proj_matrix = P_right_3
    elif max_key == 4: final_right_proj_matrix = P_right_4
    return final_right_proj_matrix

#pipeline is fundamental --> essential --> rotation and translation matrices --> projection matrix --> 3D coordinates
if __name__ == '__main__':
   print("in measure_dist")

   #focal length in meters
   focal_length = 0.016

   #get frames
   frame_left, frame_right = get_frames()

   baseline_meters_brian = 0.24
   baseline_meters_sarah = 2.92

   #get intrinsic matrices and distortion coefficients from calibration that has already occurred
   K_left = np.load('left_intrinsic_matrix.npy')
   K_right = np.load('right_intrinsic_matrix.npy')
   distcoeff_left = np.load('left_dist_coeff.npy')
   distcoeff_right = np.load('right_dist_coeff.npy')
   print("left intrinsic matrix: ")
   print(K_left)
   print("right intrinsic matrix: ")
   print(K_right)

   #focal length in meters divided by average fx and fy in pixels = meters/pixel
   meters_per_pixel_left = focal_length / ((K_left[0][0] + K_left[1][1])/2)
   meters_per_pixel_right = focal_length / ((K_right[0][0] + K_right[1][1])/2)
   print("meters per pixel left and right: ", meters_per_pixel_left, ", ", meters_per_pixel_right)

   #list of known pixel coordinatees, see labeled images to verify they match up
   known_pixel_coords_left_sarah = np.array([(1016, 1198), (384, 851), (1028, 925), (587, 702), (932, 757), (552, 727), (983, 545), (1124, 574), (1243, 601), (1342, 565), (1488, 517), (727, 411), (635,748), (814,771), (968,790)])
   known_pixel_coords_right_sarah = np.array([(779, 960), (707, 668), (1274, 702), (954, 512), (1267, 530), (930, 543), (1292, 268), (1443, 283), (1580, 292), (1673, 229), (1820, 129), (981, 164), (1000,552), (1154,559), (1307,564)])

   known_pixel_coords_left_brian = np.array([(999, 1172), (1251, 1188), (995, 1196), (1259, 1216), (880, 901), (1386, 912), (1067, 783), (1195, 783), (1334, 780), (1470, 779), (1082,592), (1387,579), (845,1303), (1436,1555), (910, 816), (917, 895), (988, 865), (983, 814), (842, 968), (785, 998), (702, 1045)])
   known_pixel_coords_right_brian = np.array([(818, 1141), (1057, 1139), (807, 1167), (1062, 1160), (724, 877), (1210, 866), (912, 752), (1038, 751), (1165, 743), (1294, 742), (925, 561), (1215, 549), (650, 1283), (1212, 1272), (751, 794), (761, 870), (836, 840), (827, 790), (676, 949), (613, 982), (518, 1035)])

   #coordinates circled in black are the coordinates that will be measured
   #draw_matching_points(frame_left, frame_right, known_pixel_coords_left_brian, known_pixel_coords_right_brian)

   #homogenize coordinates in case necessary
   known_pixel_coords_left_h, known_pixel_coords_right_h = homogenize_coords(known_pixel_coords_left_brian, known_pixel_coords_right_brian)

   #normalize known coordinates in case necessary using cv2 undistort points
   pts_left, pts_right = np.ascontiguousarray(known_pixel_coords_left_brian, np.float32), np.ascontiguousarray(
       known_pixel_coords_right_brian, np.float32)
   known_pixel_coords_left_n = cv2.undistortPoints(np.expand_dims(pts_left, axis=1), cameraMatrix=K_left,
                                         distCoeffs=distcoeff_left)
   known_pixel_coords_right_n = cv2.undistortPoints(np.expand_dims(pts_right, axis=1), cameraMatrix=K_right,
                                         distCoeffs=distcoeff_right)

   #STEP 1 - 2 ways to compute 3x3 fundamental matrix using SIFT and LMEDS OR eight point alg with known coordinates
   #error checking by principle that x'*F*x = 0
   F, inliers_left, inliers_right = find_F_sift_and_lowes(frame_left, frame_right)
   error = check_F_error(F, known_pixel_coords_left_h, known_pixel_coords_right_h)
   print("Fundamental Matrix from SIFT:")
   print(F)
   print("F error = ", error)

   # F, mask  = cv2.findFundamentalMat(known_pixel_coords_left_brian, known_pixel_coords_right_brian, method=cv2.FM_8POINT)
   # error = check_F_error(F, known_pixel_coords_left_h, known_pixel_coords_right_h)
   # print("Fundamental Matrix from 8-point alg with known coordinates:")
   # print(F)
   # print("F error = ", error)


   # STEP 2 - 3 ways to compute essential matrix
   # checked that this matrix is determinant of value very close to 0 so is singular
   # checked that it has rank of 2
   E = find_essential_matrix(F, K_left, K_right)
   print("E from math with fundamental matrix:")
   print(E)
   error = check_E_error(E, known_pixel_coords_left_n[0], known_pixel_coords_right_n[0])
   print("E error = ", error)


   E, mask = find_essential_matrix_opencv(known_pixel_coords_left_brian, known_pixel_coords_right_brian, K_left, distcoeff_left, K_right, distcoeff_right)
   print("E opencv with known points:")
   print(E)
   error = check_E_error(E, known_pixel_coords_left_n[0], known_pixel_coords_right_n[0])
   print("E error = ", error)

   E, mask = find_essential_matrix_opencv(inliers_left, inliers_right, K_left,
                                          distcoeff_left, K_right, distcoeff_right)
   print("E opencv with matched points from SIFT:")
   print(E)
   error = check_E_error(E, known_pixel_coords_left_n[0], known_pixel_coords_right_n[0])
   print("E error = ", error)


   #STEP 3 - three ways to extract rotation and translation matrices

   R1, R2, t = cv2.decomposeEssentialMat(E)
   print("results from open cv decompose:")
   print("R1:")
   print(R1)
   print("R2:")
   print(R2)
   print("t:")
   print(t)

   R1, R2, t = decompose_E(E)
   print("results from svd math to decompose E")
   print("R1:")
   print(R1)
   print("R2:")
   print(R2)
   print("t:")
   print(t)

   success, R, t, mask = cv2.recoverPose(E=E, points1=known_pixel_coords_left_brian,
                                        points2=known_pixel_coords_right_brian, cameraMatrix=K_left)
   print("results from open cv recover pose with known points:")
   print("R:")
   print(R)
   print("t:")
   print(t)

   success, R, t, mask = cv2.recoverPose(E=E, points1=pts_left,
                                        points2=pts_right, cameraMatrix=K_left)
   print("results from open cv recover pose with SIFT points:")
   print("R:")
   print(R)
   print("t:")
   print(t)



   #STEP 4: Find best projection matrix - may have to construct four possible projection matrix solutions and pick best one
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
   print("P left:")
   print(P_left)
   P_right = find_correct_projection_matrix(P_left, P_right_1, P_right_2, P_right_3, P_right_4, known_pixel_coords_left_h, known_pixel_coords_right_h)
   print("P right from find correct projection matrix:")
   print(P_right)

   P_right_recover_pose = K_right @ np.concatenate((R, t), axis=1)
   print("P right from open cv recover pose")
   print(P_right_recover_pose)

   #STEP 6 - get global coordinates using correct projection matrix

   #Coordinates if testing on Sarah images
   # stop_box_coords_left_h = np.array((1017, 1200, 1))
   # stop_box_coords_right_h = np.array((781, 960, 1))
   # shot_put_coords_left_h = np.array((833, 967, 1))
   # shot_put_coords_right_h = np.array((965, 740, 1))

   stop_box_coords_left_h = np.array((1125, 1180, 1))
   stop_box_coords_right_h = np.array((938, 1141, 1))
   shot_put_coords_left_h = np.array((1023, 926, 1))
   shot_put_coords_right_h = np.array((866, 902, 1))

   global_stopbox = triangulate_nviews([P_left, P_right], [stop_box_coords_left_h, stop_box_coords_right_h])
   global_shotput = triangulate_nviews([P_left, P_right], [shot_put_coords_left_h, shot_put_coords_right_h])


   #in this global coordinates, the shot put depth should be larger than the stop box depth as it is further away from camera
   #theoretically also, the y values should both be negative since they are below the cameras
   print("global stop box coords:", global_stopbox)
   print("global shot put coords: ", global_shotput)


   #STEP 7 - calculate distance between the two
   dist = three_d_dist_formula((global_stopbox[0], global_stopbox[1], global_stopbox[2]), (global_shotput[0], global_shotput[1], global_shotput[2]))
   print("expecting measurement for Brian around 8.21 m, 821 cm, 8210 mm")
   print("expecting measurement for Sarah around 5.78 m, 578 cm, 5780 mm")
   print("3d dist between shot put and stop box: ", dist)
   dist = two_d_dist_formula((global_stopbox[0], global_stopbox[2]),
                               (global_shotput[0], global_shotput[2]))
   print("2d dist between shot put and stop box: ", dist)

   #TEST P right 1
   print("Test P right 1")

   global_stopbox = triangulate_nviews([P_left, P_right_1], [stop_box_coords_left_h, stop_box_coords_right_h])
   global_shotput = triangulate_nviews([P_left, P_right_1], [shot_put_coords_left_h, shot_put_coords_right_h])

   # in this global coordinates, the shot put depth should be larger than the stop box depth as it is further away from camera
   print("global stop box coords:", global_stopbox)
   print("global shot put coords: ", global_shotput)

   # STEP 7 - calculate distance between the two
   dist = three_d_dist_formula((global_stopbox[0], global_stopbox[1], global_stopbox[2]),
                               (global_shotput[0], global_shotput[1], global_shotput[2]))
   print("3d dist between shot put and stop box: ", dist)

   # TEST P right 2
   print("Test P right 2")

   global_stopbox = triangulate_nviews([P_left, P_right_2], [stop_box_coords_left_h, stop_box_coords_right_h])
   global_shotput = triangulate_nviews([P_left, P_right_2], [shot_put_coords_left_h, shot_put_coords_right_h])

   # in this global coordinates, the shot put depth should be larger than the stop box depth as it is further away from camera
   print("global stop box coords:", global_stopbox)
   print("global shot put coords: ", global_shotput)

   # STEP 7 - calculate distance between the two
   dist = three_d_dist_formula((global_stopbox[0], global_stopbox[1], global_stopbox[2]),
                               (global_shotput[0], global_shotput[1], global_shotput[2]))
   print("3d dist between shot put and stop box: ", dist)

   # TEST P right 3
   print("Test P right 3")

   global_stopbox = triangulate_nviews([P_left, P_right_3], [stop_box_coords_left_h, stop_box_coords_right_h])
   global_shotput = triangulate_nviews([P_left, P_right_3], [shot_put_coords_left_h, shot_put_coords_right_h])

   # in this global coordinates, the shot put depth should be larger than the stop box depth as it is further away from camera
   print("global stop box coords:", global_stopbox)
   print("global shot put coords: ", global_shotput)

   # STEP 7 - calculate distance between the two
   dist = three_d_dist_formula((global_stopbox[0], global_stopbox[1], global_stopbox[2]),
                               (global_shotput[0], global_shotput[1], global_shotput[2]))
   print("3d dist between shot put and stop box: ", dist)

   # TEST P right 4
   print("Test P right 4")

   global_stopbox = triangulate_nviews([P_left, P_right_4], [stop_box_coords_left_h, stop_box_coords_right_h])
   global_shotput = triangulate_nviews([P_left, P_right_4], [shot_put_coords_left_h, shot_put_coords_right_h])

   # in this global coordinates, the shot put depth should be larger than the stop box depth as it is further away from camera
   print("global stop box coords:", global_stopbox)
   print("global shot put coords: ", global_shotput)

   # STEP 7 - calculate distance between the two
   dist = three_d_dist_formula((global_stopbox[0], global_stopbox[1], global_stopbox[2]),
                               (global_shotput[0], global_shotput[1], global_shotput[2]))
   print("3d dist between shot put and stop box: ", dist)

   print("Test P right recover pose")

   global_stopbox = triangulate_nviews([P_left, P_right_recover_pose], [stop_box_coords_left_h, stop_box_coords_right_h])
   global_shotput = triangulate_nviews([P_left, P_right_recover_pose], [shot_put_coords_left_h, shot_put_coords_right_h])

   # in this global coordinates, the shot put depth should be larger than the stop box depth as it is further away from camera
   print("global stop box coords:", global_stopbox)
   print("global shot put coords: ", global_shotput)

   # STEP 7 - calculate distance between the two
   dist = three_d_dist_formula((global_stopbox[0], global_stopbox[1], global_stopbox[2]),
                               (global_shotput[0], global_shotput[1], global_shotput[2]))
   print("3d dist between shot put and stop box: ", dist)














