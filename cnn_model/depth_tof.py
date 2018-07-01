import cv2
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
def smooth_cnt(frame_gray):
    (thresh, binRed) = cv2.threshold(frame_gray, 50, 255, cv2.THRESH_BINARY)
    img, Rcontours, hier_r = cv2.findContours(binRed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    r_areas = [cv2.contourArea(c) for c in Rcontours]
    max_rarea = np.max(r_areas)
    CntExternalMask = np.ones(binRed.shape[:2], dtype="uint8") * 255

    for c in Rcontours:
        if ((cv2.contourArea(c) > max_rarea * 0.70) and (cv2.contourArea(c) < max_rarea)):
            cv2.drawContours(CntExternalMask, [c], -1, 0, 1)
    return CntExternalMask
def detect_tof(frame):
    frame_gay = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gay = np.abs(255 - frame_gay)
    frame_gay[frame_gay <= 70] = 0
    cv2.imshow("gg", frame_gay)
    image, contours, hierarchy = cv2.findContours(frame_gay, 1, 2)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))
    for i, cnt in enumerate(contours):
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if cv2.contourArea(approx) > 50 and cv2.contourArea(approx) < 10000:  # TODO: Change this
            print(cv2.contourArea(approx))
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            point = np.array((int(rect[0][0]), int(rect[0][1])))
            side = np.array((int(rect[0][0] - rect[1][0] / 4.), int(rect[0][1] - rect[1][1] / 4.)))
            area_diff = rect[1][0] * rect[1][1] - cv2.contourArea(approx)
            col_cent = np.array(frame[point[1], point[0], :]) - np.array([255, 253, 255])
            print(col_cent)
            if area_diff < 3000 and sum(col_cent) != 0:
                cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
            elif sum(col_cent) == 0:
                cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
    return frame
if __name__ == "__main__":
    film ='C:\\Users\Daniel\Downloads\TOFPicoAll.avi'
    cap = cv2.VideoCapture(film)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('C:\\Users\Daniel\Downloads\\tofout.avi', fourcc, 30.0, (224,171))
    while (cap.isOpened()):
        ret, frame = cap.read()
        # frame = np.clip(np.abs(255-frame),100,255)
        # frame[frame == 100] = 0
        frame =detect_tof(frame)
        cv2.imshow('frame', frame)
        out.write(frame)

        if cv2.waitKey(24) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

            #