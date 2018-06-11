import cv2
import numpy as np


class FeatureBasedTempalte:
    """
    Full package of FeatureBased for single image
    """
    # TODO: fix brief, fast
    # 'fast': self.fast_kp_des
    def __init__(self, method):
        self.methods = {'sift': self.sift_kp_des, 'surf': self.surf_kp_des,
                        'orb': self.orb_kp_des}
        self.clean_img = None
        self.img = None
        self.params = None
        if method in self.methods:
            self.method = method
        else:
            raise AttributeError("Invalid method: {}\n Method is not in {}".format(method, self.methods.keys()))

    def load_clean(self, path , params=None):
        self.params = {}
        self.img = cv2.imread(path, 0)  # gray
        kp, des = self.methods[self.method](self.img)
        self.params['kp'] = kp
        self.params['des'] = des
        self.params['size'] = self.img.shape
        self.params['path'] = path

    def sift_kp_des(self, img=None):
        sift = cv2.xfeatures2d.SIFT_create()
        return sift.detectAndCompute(self.img if img is None else img, None)

    def surf_kp_des(self, img=None):
        surf = cv2.xfeatures2d.SURF_create()
        return surf.detectAndCompute(self.img if img is None else img, None)

    def fast_kp_des(self, img=None):
        fast = cv2.FastFeatureDetector_create(nonmaxSuppression=True)  # TODO : change params
        return fast.detectAndCompute(self.img if img is None else img, None)

    def orb_kp_des(self, img=None):
        orb = cv2.ORB_create()  # TODO : chan
        return orb.detectAndCompute(self.img if img is None else img, None)

    # TODO: test those functions
    def brief_kp(self):
        brief = cv2.DescriptorExtractor_create("BRIEF")
        return brief.compute(self.img, self.star_kp())

    def star_kp(self):
        star = cv2.FeatureDetector_create("STAR")
        return star.detect(self.img, None)

    def brute_force_matching(self, des1, des2=None):
        des2 = self.params['des'] if des2 is None else des2
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1, des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)  # distance threshold or first 10 points?
        return matches

    def brute_force_matching_ratio_test(self, des1, des2=None):
        des2 = self.params['des'] if des2 is None else des2
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)  # TODO: 2?
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:  # TODO 0.75?
                good.append([m])
        # return good  # TODO: return type
        return matches

    def FLANN_matching(self, des1, des2=None):
        des2 = self.params['des'] if des2 is None else des2
        # FLANN parameters
        FLANN_INDEX_KDTREE = 2
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=1)
        search_params = dict(checks=1000)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1.astype(np.float32), des2.astype(np.float32), k=2)
        matchesMask = [[0, 0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]

        # img3 = cv2.drawMatchesKnn(self.img, self.params['kp'], , kp2, matches, None, **draw_params)
        # plt.imshow(img3, )
        # plt.show()
        return matches, matchesMask

    def plot_img(self, with_kp=True):
        outImg = np.copy(self.img)
        if with_kp:
            outImg = cv2.drawKeypoints(self.img, self.params['kp'], outImg, color=(255, 0, 0))

        cv2.imshow("Image", outImg)
        k = cv2.waitKey(0)
        if k == 27:  # wait for ESC key to exit
            cv2.destroyAllWindows()

    def plot_compare(self, path, method=None):  # TODO: image Feature Matching method choice

        img2 = cv2.imread(path, 0)
        kp1, des1 = self.methods[self.method](img2)
        # matches, matchesMask = self.FLANN_matching(des1)
        matches = self.brute_force_matching(des1)
        # Apply ratio test
        # good = []
        # for m, n in matches:
        #     if m.distance < 0.8 * n.distance:
        #         good.append([m])



        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           # matchesMask=matchesMask,
                           # flags=2)
                           )
        # outImg = cv2.drawMatchesKnn(self.img, self.params['kp'], img2, kp1, good, outImg)
        # print(matches)

        outImg = np.array([])
        outImg = cv2.drawMatches(self.img, self.params['kp'], img2, kp1, matches[:1], outImg)
        cv2.imshow("Image", outImg)
        k = cv2.waitKey(0)
        if k == 27:  # wait for ESC key to exit
            cv2.destroyAllWindows()

    def __str__(self):
        return "Image: {}\nSize: {}\nMethod: {}\nKeyPoints: {}".\
            format(self.params["path"], self.params["size"], self.method, len(self.params['kp']))

    def bullshit(self):
        import numpy as np
        import cv2
        from matplotlib import pyplot as plt

        img1 = cv2.imread("../data/gate_clean.jpeg", 0)  # queryImage
        img2 = cv2.imread("../data/cad_renders2/GateRenders_0040.jpg", 0)  # trainImage

        # Initiate SIFT detector
        orb = cv2.ORB_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw first 10 matches.
        img3 = np.array([])
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10],img3, flags=2)

        plt.imshow(img3), plt.show()
if __name__ == "__main__":
    path = "../data/gate_clean.jpeg"
    path2 = path
    # path2 = "../data/cad_renders/GateRenders 3.jpg"
    path2 = "../data/cad_renders2/GateRenders_0001.jpg"
    fbt = FeatureBasedTempalte("orb")
    # fbt.load_clean(path)
    # fbt.plot_img(True)
    fbt.bullshit()
    # fbt.plot_compare(path2)
