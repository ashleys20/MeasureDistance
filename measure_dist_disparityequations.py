import numpy as np
from camera_calibration import undistort
import cv2
from matplotlib import pyplot as plt
import math

def three_d_dist_formula(p0, p1):
    return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 + (p0[2]-p1[2]) ** 2)

# Visualize epilines
# Adapted from: https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
def drawlines(img1src, img2src, lines, pts1src, pts2src):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''

    r, c, _ = img1src.shape
    #img1color = cv2.cvtColor(img1src, cv2.COLOR_GRAY2BGR)
    #img2color = cv2.cvtColor(img2src, cv2.COLOR_GRAY2BGR)
    img1color = img1src.copy()
    img2color = img2src.copy()
    # Edit: use the same random seed so that two images are comparable!
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1color = cv2.line(img1color, (x0, y0), (x1, y1), color, 1)
        img1color = cv2.circle(img1color, tuple(pt1), 5, color, -1)
        img2color = cv2.circle(img2color, tuple(pt2), 5, color, -1)
    return img1color, img2color


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

    pts1 = np.multiply(np.int32(pts1), 1)
    pts2 = np.multiply(np.int32(pts2),1)
    #confidence default = 0.99
    F, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_RANSAC)

    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    return F, pts1, pts2

if __name__ == '__main__':
    print("in measure_dist disparity equations")

    K_left = np.load('left_intrinsic_matrix.npy')
    distcoeff_left = np.load('left_dist_coeff.npy')
    K_right = np.load('right_intrinsic_matrix.npy')
    distcoeff_right = np.load('right_dist_coeff.npy')

    focal_length_meters = 0.016

    # focal length in meters divided by average fx and fy in pixels = meters/pixel
    meters_per_pixel_left = focal_length_meters / ((K_left[0][0] + K_left[1][1]) / 2)
    meters_per_pixel_right = focal_length_meters / ((K_right[0][0] + K_right[1][1]) / 2)
    print("meters per pixel left and right: ", meters_per_pixel_left, ", ", meters_per_pixel_right)

    baseline_meters_brian = 0.30

    known_pixel_coords_left_brian = np.array(
        [(999, 1172), (1251, 1188), (995, 1196), (1259, 1216), (880, 901), (1386, 912), (1067, 783), (1195, 783),
         (1334, 780), (1470, 779), (1082, 592), (1387, 579), (845, 1303), (1436, 1555), (910, 816), (917, 895),
         (988, 865), (983, 814), (842, 968), (785, 998), (702, 1045)])
    known_pixel_coords_right_brian = np.array(
        [(818, 1141), (1057, 1139), (807, 1167), (1062, 1160), (724, 877), (1210, 866), (912, 752), (1038, 751),
         (1165, 743), (1294, 742), (925, 561), (1215, 549), (650, 1283), (1212, 1272), (751, 794), (761, 870),
         (836, 840), (827, 790), (676, 949), (613, 982), (518, 1035)])

    #only doing multiply here in case we want to change 1 for the pixel to meters scaling factor
    stop_box_coords_left_h = np.concatenate((np.multiply((1125, 1180), 1), [1]))
    stop_box_coords_right_h = np.concatenate((np.multiply((938, 1141), 1), [1]))
    shot_put_coords_left_h = np.concatenate((np.multiply((1023, 926), 1), [1]))
    shot_put_coords_right_h = np.concatenate((np.multiply((866, 902), 1), [1]))

    frame_left = cv2.imread('frame_left_brian.png')
    frame_left_u = undistort(frame_left, True)
    frame_right = cv2.imread('frame_right_brian.png')
    frame_right_u = undistort(frame_right, False)

    F, pts1, pts2 = find_F_sift_and_lowes(frame_left_u, frame_right_u)
    print("Fundamental Matrix from SIFT:")
    print(F)

    # F, mask = cv2.findFundamentalMat(known_pixel_coords_left_brian, known_pixel_coords_right_brian,
    #                                  method=cv2.FM_8POINT)
    # print("Fundamental Matrix from 8-point alg with known coordinates:")
    # print(F)

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(
        pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(frame_left, frame_right, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(
        pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(frame_right, frame_left, lines2, pts2, pts1)

    #plt.subplot(121), plt.imshow(img5)
    #plt.subplot(122), plt.imshow(img3)
    #plt.suptitle("Epilines in both images")
    #plt.show()

    h1, w1, _ = frame_left.shape
    h2, w2, _ = frame_right.shape

    #Get homography matrices to rectify later
    _, H1, H2 = cv2.stereoRectifyUncalibrated(
        np.float32(pts1), np.float32(pts2), F, imgSize=(w1, h1)
    )
    print("H1:")
    print(H1)
    print("H2:")
    print(H2)

    img_left_rectified = cv2.warpPerspective(frame_left, H1, (w1, h1))
    img_right_rectified = cv2.warpPerspective(frame_right, H2, (w2, h2))

    #still not sure if I can do this to get accurate focal length but I know the third column does equal the homoegenized version of the camera center
    K_left_rectified = H1 @ K_left
    K_right_rectified = H2 @ K_right

    camera_center_rectified_left = K_left[:, 2]
    camera_center_rectified_right = K_right[:, 2]


    #find rectified coordinates of important points and draw them on rectified image
    stop_box_rectified_left = H1 @ np.array(stop_box_coords_left_h)
    stop_box_rectified_left = 1/stop_box_rectified_left[2] * stop_box_rectified_left
    cv2.circle(img_left_rectified, (int(stop_box_rectified_left[0]), int(stop_box_rectified_left[1])), 5, (0, 255, 255), 5)
    shotput_rectified_left = H1 @ np.array(shot_put_coords_left_h)
    shotput_rectified_left = 1 / shotput_rectified_left[2] * shotput_rectified_left
    cv2.circle(img_left_rectified, (int(shotput_rectified_left[0]), int(shotput_rectified_left[1])), 5, (0, 255, 255), 5)
    stop_box_rectified_right = H2 @ np.array(stop_box_coords_right_h)
    stop_box_rectified_right = 1 / stop_box_rectified_right[2] * stop_box_rectified_right
    cv2.circle(img_right_rectified, (int(stop_box_rectified_right[0]), int(stop_box_rectified_right[1])), 5, (0, 255, 255), 5)
    shotput_rectified_right = H2 @ np.array(shot_put_coords_right_h)
    shotput_rectified_right = 1 / shotput_rectified_right[2] * shotput_rectified_right
    cv2.circle(img_right_rectified, (int(shotput_rectified_right[0]), int(shotput_rectified_right[1])), 5, (0, 255, 255), 5)
    camera_center_rectified_left = 1 / camera_center_rectified_left[2] * camera_center_rectified_left
    cv2.circle(img_left_rectified, (int(camera_center_rectified_left[0]), int(camera_center_rectified_left[1])), 5,
               (0, 0, 0),
               5)
    camera_center_rectified_right = 1 / camera_center_rectified_right[2] * camera_center_rectified_right
    cv2.circle(img_right_rectified, (int(camera_center_rectified_right[0]), int(camera_center_rectified_right[1])), 5,
               (0, 0, 0),
               5)
    cv2.imwrite("rectified_left.png", img_left_rectified)
    cv2.imwrite("rectified_right.png", img_right_rectified)

    # Draw the rectified images with lines to show the similar axes
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    axes[0].imshow(img_left_rectified, cmap="gray")
    axes[1].imshow(img_right_rectified, cmap="gray")
    axes[0].axhline(250)
    axes[1].axhline(250)
    axes[0].axhline(450)
    axes[1].axhline(450)
    axes[0].axhline(650)
    axes[1].axhline(650)
    axes[0].axhline(850)
    axes[1].axhline(850)
    axes[0].axhline(1050)
    axes[1].axhline(1050)
    axes[0].axhline(1250)
    axes[1].axhline(1250)
    plt.suptitle("Rectified images")
    plt.savefig("rectified_images.png")
    #plt.show()

    print(camera_center_rectified_left)
    print(camera_center_rectified_right)
    print(stop_box_rectified_left)
    print(stop_box_rectified_right)
    print(shotput_rectified_left)
    print(shotput_rectified_right)

    #order in which I subtract depends on if value is greater or less than camera center x and camera center y
    stop_box_adjusted_left = (stop_box_rectified_left[0] - camera_center_rectified_left[0], camera_center_rectified_left[1] - stop_box_rectified_left[1])
    stop_box_adjusted_right = (stop_box_rectified_right[0] - camera_center_rectified_right[0],
                              camera_center_rectified_right[1] - stop_box_rectified_right[1])
    shot_put_adjusted_left = (shotput_rectified_left[0] - camera_center_rectified_left[0], camera_center_rectified_left[1] - shotput_rectified_left[1])
    shot_put_adjusted_right = (shotput_rectified_right[0] - camera_center_rectified_right[0],
                         camera_center_rectified_right[1] - shotput_rectified_right[1])

    #y-coordinates should be the same for each
    print("adjusted coordinates from camera centers:")
    print(stop_box_adjusted_left)
    print(stop_box_adjusted_right)
    print(shot_put_adjusted_left)
    print(shot_put_adjusted_right)

    # d = x'L - x'R
    stop_box_disparity = stop_box_adjusted_left[0] - stop_box_adjusted_right[0]
    shot_put_disparity = shot_put_adjusted_left[0] - shot_put_adjusted_right[0]
    print("disparities:")
    print(stop_box_disparity)
    print(shot_put_disparity)

    #[x,y,z] = B/d * [x'L, y'L, f]
    global_stop_box = baseline_meters_brian / stop_box_disparity * np.array(
        (stop_box_adjusted_left[0], stop_box_adjusted_left[1], K_left_rectified[0][0]))
    global_shot_put = baseline_meters_brian/shot_put_disparity * np.array((stop_box_adjusted_left[0], stop_box_adjusted_left[1], K_left_rectified[0][0]))
    print("global coordinates, in meters")
    print("stop box: ", global_stop_box)
    print("shot put: ", global_shot_put)

    dist = three_d_dist_formula((global_stop_box[0], global_stop_box[1], global_stop_box[2]),
                                (global_shot_put[0], global_shot_put[1], global_shot_put[2]))
    print("distance ", dist)









