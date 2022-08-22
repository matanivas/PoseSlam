import cv2
import numpy as np
from VAN_project.code.utils.utils import read_images


class StereoFrame:

    def __init__(self, idx, left_img_m, right_img_m, detector=None):
        self._idx = idx
        self._left_img_m = left_img_m
        self._right_img_m = right_img_m

        # img1 kpts and dscs
        self._img1_kpts = None
        self._img1_dscs = None
        # img2 kpts and dscs
        self._img2_kpts = None
        self._img2_dscs = None

        # bool to indicate if the the kpts are filtered using stereo - for sanity
        self._after_stereo = False

        # triangulation of points
        self._three_d_points = None

        # number of matches between the images
        self._num_matches = 0

        # use the given detector (if given)
        self._detector = detector

        # Process the stereo image
        self.process_stereo()

    def __detect_kpts(self):
        """
        Use cv2 to detect possible keypoints
        """
        # read images
        img1, img2 = read_images(self._idx)

        # detect keypoints
        alg = cv2.AKAZE_create()
        if self._detector:
            alg = self._detector
        self._img1_kpts, self._img1_dscs = alg.detectAndCompute(img1, None)
        self._img2_kpts, self._img2_dscs = alg.detectAndCompute(img2, None)

    def __get_y_distances(self, matches):
        """
        given matches across left and right image - return array with the y distance between matching key-points
        """
        y1 = np.array([self._img1_kpts[match.queryIdx].pt[1] for match in matches])
        y2 = np.array([self._img2_kpts[match.trainIdx].pt[1] for match in matches])
        return list(np.abs(y2-y1))

    def __filter_by_stereo(self, matches):
        """
        given matches between left and right - return only those that match the stereo pattern
        """
        dists = self.__get_y_distances(matches)
        img1_kpts_after_stereo = []
        img1_dscs_after_stereo = []
        img2_kpts_after_stereo = []
        img2_dscs_after_stereo = []
        for i in range(len(dists)):
            if dists[i] < 1.0:
                img1_kpts_after_stereo.append(self._img1_kpts[matches[i].queryIdx].pt)
                img1_dscs_after_stereo.append(self._img1_dscs[matches[i].queryIdx])
                img2_kpts_after_stereo.append(self._img2_kpts[matches[i].trainIdx].pt)
                img2_dscs_after_stereo.append(self._img2_dscs[matches[i].trainIdx])

        # return np.array(img1_kpts), np.array(img1_dscs), np.array(img2_kpts), np.array(img2_dscs)
        self._img1_kpts = np.array(img1_kpts_after_stereo)
        self._img1_dscs = np.array(img1_dscs_after_stereo)
        self._img2_kpts = np.array(img2_kpts_after_stereo)
        self._img2_dscs = np.array(img2_dscs_after_stereo)
        self.after_stereo = True

    def __match(self):
        # match between the key-points of the two images

        # cv2 matcher
        brute_force = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = brute_force.match(self._img1_dscs, self._img2_dscs)

        # filter using stereo
        self.__filter_by_stereo(matches)

    def __triangulate(self):
        cv2T = cv2.triangulatePoints(self._left_img_m, self._right_img_m, self._img1_kpts.T, self._img2_kpts.T).T
        cv2T = cv2.convertPointsFromHomogeneous(cv2T).reshape(self._img1_kpts.shape[0], 3)
        self._three_d_points = cv2T
        self._num_matches = len(self._three_d_points)

    def process_stereo(self):
        """
        Analyse and compute all relevant data in a stereo pair
        """
        # detect the key-points in the stereo image
        self.__detect_kpts()

        # match between the two images, while filtering based on stereo
        self.__match()

        # compute 3d point cloud
        self.__triangulate()

    def get_left_dscs(self):
        return self._img1_dscs

    def get_left_right_point(self, i):
        return self._img1_kpts[i][0], self._img2_kpts[i][0], (self._img1_kpts[i][1] + self._img2_kpts[i][1]) / 2

    def get_right_dscs(self):
        return self._img2_dscs

    def get_left_kpts(self):
        return self._img1_kpts

    def get_right_kpts(self):
        return self._img2_kpts

    def get_three_d_pts(self):
        return self._three_d_points

    def get_left_intrinsic_mat(self):
        return self._left_img_m

    def get_right_intrinsic_mat(self):
        return self._right_img_m

    def get_num_matches(self):
        return self._num_matches

    def get_idx(self):
        return self._idx

