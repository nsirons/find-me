import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


def draw_point(img, points, thick = 5, color = (0,0, 255), point_num = None):
    points = np.array(points, dtype=int)
    img_copy = img #np.copy(img)
    for i in range(len(points)):
        cv2.circle(img_copy, (points[i][0], points[i][1]), thick, color, -1)
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    return img_copy


def draw_line(img, lines, point_num=None, color = (0, 0, 255)):
    img_copy = np.copy(img)
    point_num = point_num
    for i, line in enumerate(lines):
        if point_num is not None:
            # print(line[0],i)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img_copy, str(i), line[1], font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.line(img_copy, line[0], line[1], color, 2, cv2.LINE_AA)

    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    return img_copy

def thomas(gray,img):
    corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
    corners = np.int0(corners)
    points= np.array([])
    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)
        points =np.append(points,[x,y], axis=0)
    plt.imshow(img)
    plt.show()
    return points

def harris(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.08)
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    # Now draw them
    res = np.hstack((centroids, corners))
    res = np.array(res, dtype=int)
    points = res[:, 0:2]
    return points

# edges_res = (200, 200)
# rho_res = 5
# theta_res = np.pi / 180
# line_res = 350  # pixels
# hough_res = [rho_res, theta_res, line_res]
# tune_parameters = np.array([edges_res, hough_res])
# hough_lines = hough_transform(img, tune_params=tune_parameters)

def hough_transform(img, tune_params= np.array([(200, 200), [5, np.pi/180, 350]]) ):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, tune_params[0][0], tune_params[0][1], None, 3)
    point_seq = []
    'Standard Hough Transform'
    lines = cv2.HoughLines(edges, tune_params[1][0], tune_params[1][1], tune_params[1][2], None, 0, 0)
    if lines is not None:
        for i in range(len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            # cv2.line(img, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
            point_seq.append([pt1, pt2])


    # cv2.imwrite('hough_output.jpg', img)
    #
    'Probabilistic Hough Transform'
    # reallines =[]
    # linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=500, maxLineGap=1)
    # for x1, y1, x2, y2 in linesP[0]:
    #     reallines.append([[x1, y1], [x2, y2]])

    return point_seq


def hough_harris(lines, corners):
    corn_upd = []
    for i in range(len(corners)):
        for points in lines:
            m = (points[1][1] - points[0][1]) / (points[1][0] - points[0][0] + 1)
            pt_y = abs(points[0][1] + m * (corners[i][0] - points[0][0]))
            pt_x = abs((pt_y - points[0][1]) / (m + 1) + points[0][0])
            if abs(corners[i][1] - pt_y) < 100 and abs(corners[i][0] - pt_x) < 100:
                corn_upd.append(corners[i])
                break

    return np.array(corn_upd)


def clean_corners(corners, neighbors =0):
    corners_upd = []
    while corners.shape[0] > 0:
        point = corners[0, :]
        corners = np.delete(corners, [0], 0)
        new_corners = np.array([[point[0], point[1]]])
        for other_point in corners:
            if np.linalg.norm(other_point - point) < 10:
                new_corners = np.append(new_corners, [other_point], axis=0)

        if new_corners.shape[0] > neighbors:
            corners_upd.append([np.mean(new_corners[:, 0]), np.mean(new_corners[:, 1])])
            for corner in new_corners[neighbors:, :]:
                corners = np.delete(corners, np.where(corners == corner)[0], 0)
    # corners_upd = np.array(corners_upd)

    return corners_upd


def neighbor(start_point, points, radius_thresh):
    prev_radius = radius_thresh
    next_neighbor = None
    for other_point in points:
        radius = np.linalg.norm(start_point - other_point)
        if radius < prev_radius:
            next_neighbor = other_point
            prev_radius = radius

    return next_neighbor


def slope(start, second):
    return math.degrees(math.atan(abs((second[1] - start[1]) / (second[0] - start[0]+0.1))))

def gaussian_noise(img, mu, sigma):
    return np.clip(img + np.random.normal(mu, sigma, img.shape).astype(np.uint8), 0, 255)

def gradients_lines(corners, radius_thresh):
    corner_gate = []
    lines = []
    cur_slope = 0
    points = np.copy(corners)
    point = points[0]
    start_point = point
    points = np.delete(points, [0], 0)
    corner = 0
    while len(points) != 1:
        next_point = neighbor(point, points, radius_thresh)
        # if no elements are found within radius_tresh then delete and look for first point in list
        if next_point is None:
            points = np.delete(points, np.where(points == point)[0], 0)
            lines =[]
            corner_gate =[]
            if len(points) != 0:
                point = points[0]
        else:
            m = slope(point, next_point)
            # print(abs(cur_slope - m), next_point, point, m)
            if abs(cur_slope - m) > 50:
                # corner found
                corner_gate.append(point)
                cur_slope = m
                # print("CORNER:", point)

            lines.append([tuple(map(int, point)), tuple(map(int, next_point))])
            point = next_point  # variable is kept with this value till following iter
            points = np.delete(points, np.where(points == point)[0], 0)
    last_point = lines[-1][1]
    start_point = lines[0][0]
    m = slope(last_point, start_point)
    if abs(cur_slope - m) > 50:
        # corner found
        corner_gate.append(last_point)
        cur_slope = m
        # print("CORNER:", point)
    lines.append([tuple(map(int, last_point)), tuple(map(int, start_point))])
    return lines, corner_gate


'Function commands to be applied from user'
# img = draw_line(img, hough_lines)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# harris_corners = harris(img)
# harris_img = draw_point(img, harris_corners)
#
# hough_lines = hough_transform(img, tune_params)
# hough_img = draw_line(img, hough_lines)
# hough_harris = hough_harris(hough_lines, harris_corners)
# # print(hough_harris)
#
# hough_harris_img  = draw_point(img, hough_harris)



# img = cv2.imread(img_dir)
# # img = cv2.imread("kitshiron.png")
# edges_res = (200, 200)
# rho_res = 5
# theta_res = np.pi / 180
# line_res = 94  # pixels
# hough_res = [rho_res, theta_res, line_res]
# tune_params = np.array([edges_res, hough_res])
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# harris_corners = harris(img)
# harris_img = draw_point(img, harris_corners)
#
# hough_lines = hough_transform(img, tune_params)
# hough_img = draw_line(img, hough_lines)
#
# max_pt = 0#int(0.03 * harris_corners.shape[0])
# polished_corners = clean_corners(harris_corners, neighbors=max_pt)
# polished_img = draw_point(img, polished_corners)
#
# new_corners = hough_harris(hough_lines, polished_corners)
# new_corns = draw_point(img, new_corners)
#
# gradients, corners_gate = gradients_lines(polished_corners, 200)
# gradients_img = draw_line(img, gradients, point_num=1)
# gradients_img = draw_point(gradients_img, corners_gate, thick= 20, color=(255,255,255))
# # gate_lines = draw_gate(np.array(new_corners, dtype=int))
# # gate =  draw_line(img,gate_lines)
# slopes, intercepts, slopes_h, intercepts_h = find_lines(harris_corners)
#
# plot_corners(slopes, intercepts,slopes_h,intercepts_h, harris_img)

# 'Plotter'
# plot = False
# if plot:
# # Plot nr 1
#     plt.subplot(121)
#     plt.imshow(hough_img)
#     plt.subplot(122)
#     plt.imshow(harris_img)
#     plt.show()
#
#     # Plot nr 2
#     plt.subplot(121)
#     plt.imshow(gradients_img)
#
#     plt.subplot(122)
#     plt.imshow(polished_img)
#
#     plt.show()