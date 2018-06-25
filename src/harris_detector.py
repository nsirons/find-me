import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import os
from scipy import stats
import glob
import re
from src.standard_algs import *
from src.snake_check import *
from src.benchmark import Benchmark

class Harris_detector:
    def __init__(self):
        self.gate_detected = False
        self.img_size = [0,0]

    def get_most_dense(self,lines):
        new_lines = sorted(lines, key=lambda x: len(x[0]), reverse=True)
        if len(new_lines)>1:
            return new_lines[0:2]
        else:
            # print("Not enough lines detected")
            return new_lines[0:2]


    def find_lines(self,corners, initial_points = 0):
        tot_used = 0
        num_points = len(corners[:, 0])
        vert_range = 50  # make it a function of shape of corners
        horiz_range = 50
        corners = np.delete(corners, [0], 0)
        corners = np.copy(corners)
        corners_x = np.copy(corners)
        max_points =0# max(2, int(0.05 * num_points))
        lines = []
        lines_h = []

        'Look for vertical lines'
        while len(corners) > 0:
            point = corners[0]
            # corners = np.delete(corners, [0], 0)
            condition = abs(corners[:, 0] - point[0]) < horiz_range
            x_copy = corners[condition]
            corners = corners[np.logical_not(condition)]

            if x_copy.shape[0] > max_points:
                lines.append([x_copy[:, 0], x_copy[:, 1]])
            # print(lines)

        lines = self.get_most_dense(lines)
        slopes = []
        intercepts = []
        for line in lines:
            tot_used += len(line[0])
            slope, intercept, r_value, p_value, std_err = stats.linregress(line[1], line[0])
            if abs(slope)< 0.3:
                slopes.append(slope+0.00001)
                intercepts.append(intercept)


        ' Set limits to look for horizontal lines'
        if len(lines) > 0:
            temp = np.append(np.array(lines)[:, 0][0], np.array(lines)[:, 0][1%len(lines)])
            x_min = min(temp)-20
            x_max = max(temp)+20
            'Bitwise and performed to do dot product such that range condition is met'
            cond = np.bitwise_and(corners_x[:, 0] < x_max, corners_x[:, 0] > x_min)
            corners_x = corners_x[cond]
            cond = corners_x[:, 1] < self.img_size[0]-35
            corners_x = corners_x[cond]
            'Look for horizontal lines'
            while len(corners_x) > 0:
                point = corners_x[0]
                # corners_x = np.delete(corners_x, [0], 0)
                cond = abs(corners_x[:, 1] - point[1]) < vert_range
                x_copy = corners_x[cond]
                corners_x = corners_x[abs(corners_x[:, 1] - point[1]) > vert_range]
                if x_copy.shape[0] > max_points:
                    lines_h.append([x_copy[:, 0], x_copy[:, 1]])

            lines_h = self.get_most_dense(lines_h)
            slopes_h = []
            intercepts_h = []

            for line in lines_h:
                tot_used += len(line[0])
                slope, intercept, r_value, p_value, std_err = stats.linregress(line[0], line[1])
                if abs(slope)< 0.5:
                    slopes_h.append(slope)
                    intercepts_h.append(intercept)

            ratio = tot_used / num_points
            noisy = initial_points > 1000
            if ratio < 0.1 and not noisy:
                # print("Gate was not detected: the number of points interpolated was below threshold or image might be too noisy")
                self.gate_detected = False
            else:
                self.gate_detected = True
        else:
            slopes, intercepts, slopes_h, intercepts_h, ratio, self.gate_detected = None,None,None,None,None,False
        return slopes, intercepts, slopes_h, intercepts_h, ratio, self.gate_detected


    def plot_lines(self,slopes, intercepts, slopes_h, intercepts_h, img,plot=True):
        x_max = img.shape[1]
        y_max = img.shape[0]
        xs = np.arange(0, x_max, 0.5)
        lines = []
        ys = []
        'Plot lines in image'
        for i in range(len(slopes)):
            m = slopes[i]
            c = intercepts[i]
            ys = []
            for x in xs:
                ys.append(1 / m * x - c / m)

            if plot:
                plt.plot(xs, ys)
            lines.append(ys)

        for i in range(len(slopes_h)):
            m = slopes_h[i]
            c = intercepts_h[i]
            ys = []
            for x in xs:
                ys.append(m * x + c)
            if plot:
                plt.plot(xs, ys)

            lines.append(ys)
        # print(lines)
        # print(len(lines))
        if plot:
            plt.imshow(img)
            plt.xlim([0, x_max])
            plt.ylim([y_max, 0])
            # plt.show()

        return lines


    def line_intersection(self,mv, bv, mh, bh):
        if (mh - 1 / mv) == 0:
            # print("No intersection")
            return 0,0
        else:
            x = (-bh - bv / mv) / (mh - 1 / mv)
            y = mh * x + bh
            return x, y


    def get_intersects(self,slopes_v, intercepts_v, slopes_h, intercept_h, ar_limits):
        if len(slopes_h)>1 and len(slopes_v) >1:
            point1 = self.line_intersection(slopes_v[0], intercepts_v[0], slopes_h[0], intercept_h[0])
            point2 = self.line_intersection(slopes_v[0], intercepts_v[0], slopes_h[1], intercept_h[1])
            point3 = self.line_intersection(slopes_v[1], intercepts_v[1], slopes_h[0], intercept_h[0])
            point4 = self.line_intersection(slopes_v[1], intercepts_v[1], slopes_h[1], intercept_h[1])
            points = [point1, point2, point3, point4]

            if None not in points:
                h_len1 = abs(point1[0] - point3[0])
                h_len2 = abs(point2[0] - point4[0])
                v_len1 = abs(point1[1] - point2[1])
                v_len2 = abs(point3[1] - point4[1])
                # print("ar = ", abs(h_len1+h_len2)/abs(v_len1+v_len2))
                ar = abs(h_len1 + h_len2) / abs(v_len1 + v_len2)
                ar_bool =ar > ar_limits[0] and ar < ar_limits[1]

                if ar_bool:
                    return points, ar_bool
                else:
                    # print("No gate in sight: AR requirement is not met")
                    return points, ar_bool

            else:
                points = list(filter(None, points))
                # print("Not enough lines detected")  # TODO: implemet only points found
            return points


    def outliers_z_score(self,corners, z_score_thshd = 1.5):
        ys = corners[:, 0]
        # print(len(ys))
        threshold = z_score_thshd # thshld = 1 is too low since useful points are trashed
        mean_y = 650#np.mean(ys)
        stdev_y = np.std(ys)
        z_scores = [(y - mean_y) / stdev_y for y in ys]
        corners = np.delete(corners, np.where(np.abs(z_scores) > threshold)[0], 0)
        # print(len(corners[:, 0]))
        return corners


    def find_gate(self,img_dir, plot=False, verbosity=False,num=0):
        corners_ordered = [0, 0], [0, 0], [0, 0], [0, 0]
        img = cv2.imread(img_dir)
        self.img_size = img.shape
        # img =  cv2.GaussianBlur(img, (11, 11), sigmaX=10, sigmaY=10)
        # img = gaussian_noise(img, 0, 20)
        harris_corners = harris(img)
        initial_points = len(harris_corners)
        # TODO: implement outlier detection in harris_corners
        # harris_corners = outliers_z_score(harris_corners, z_score_thshd = .75)
        harris_img = draw_point(img, harris_corners)
        slopes, intercepts, slopes_h, intercepts_h, ratio_pts, self.gate_detected = self.find_lines(harris_corners, initial_points=0)
        if slopes is not None:
            final_lines = self.plot_lines(slopes, intercepts, slopes_h, intercepts_h, harris_img, plot=True)
            if self.gate_detected and len(final_lines) >3:
                corners, ar = self.get_intersects(slopes, intercepts, slopes_h, intercepts_h, [-10,10])#[0.01, 1.3])
                indices = [3, 2, 0, 1]
                # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # check_color_cont(corners,gray)
                if corners is not None and ar:
                    if len(corners) == 4:
                        if verbosity:
                            print("Gate detected! \n")
                    corners_ordered = [corners[i] for i in indices]
                    for i in corners_ordered:
                        plt.plot(i[0], i[1], 'bo', markersize=12)
                else:
                    self.gate_detected = False
                    if verbosity:
                        print("Gate not detected! \n")
                    corners_ordered = [0,0],[0,0],[0,0],[0,0]
        else:
            self.gate_detected= False
            corners_ordered = [0, 0], [0, 0], [0, 0], [0, 0]
            if verbosity:
                print("Gate not detected!\n")
            pass  # error message previously announced
        if plot:
            # pass
            stir=str(num)
            # plt.savefig('./results/image'+stir+'.jpg')
            plt.show()

        q1,q2,q3,q4 = corners_ordered[0],corners_ordered[1],corners_ordered[2],corners_ordered[3]
        return self.gate_detected,q1,q2,q3,q4


def get_accuracy(img_list):
    sample_size = len(img_list)
    my_vals = []
    detector = Harris_detector()
    i=1
    for img_dir in img_list:
        noisy = re.search(r'(noise_)[3-9]0', img_dir)
        if not noisy:
            i+=1
            detector.gate_detected, q1,q2,q3,q4 = detector.find_gate(img_dir, plot=True, verbosity=False,num=i)
            print('This is result:')
            print(detector.gate_detected)
            my_vals.append(detector.gate_detected)

    accuracy = sum(my_vals) / sample_size
    print(
        'The accuracy of correct images detected is {}% (aka True-Positive verification to be implemented)'.format(
            accuracy * 100))

    return accuracy

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.realpath(__file__))
    img_dir = os.path.join(base_path, 'cad_renders_dist\*.jpg')
    import time
    # t0=time.perf_counter()
    img_dir='C:\\Users\Daniel\Google Drive\\targets\*.png'
    img_dir = 'C:\\Users\Daniel\Downloads\GateRenders val\GateRenders 6\GateRenders_animation\*.jpg'
    img_dir = 'C:\\Users\Daniel\Downloads\\20June2\GateRenders360Rotation\GateRenders360Rotation_animation\*.jpg'
    img_list = glob.glob(img_dir)
    accuracy = get_accuracy(img_list)
    # print(time.perf_counter()-t0)


    # 'Benchmark'
    # path = '../data/cad_renders2_dist'
    # # from src.benchmark import Benchmark
    # b = Benchmark(Harris_detector())
    # b.load_data(path)
    # b.test()
    # print(b)
    #
    # 'Try Hough Transform'
    # for img_dir in img_list:
    #     img = cv2.imread(img_dir)
    #     edges = cv2.Canny(img, 200,5, None, 3)
    #     plt.imshow(edges)
    #     plt.show()
    #     hough_lines = hough_transform(img, tune_params= np.array([(200, 200), [5, np.pi/180, 150]]))
    #     img= draw_line(img,hough_lines)
    #     plt.imshow(img)
    #     plt.show()



