import cv2
import numpy as np
from VAN_project.code.utils.rts_utils import rodriguez_to_mat
from copy import deepcopy
from VAN_project.code.utils.matching_utils import get_ransac_num_iteration
from VAN_project.code.utils.utils import two_transformations_composition

RATIO = 0.8
RANSAC_PROBABILITY = 0.85


class MultiFrame:

    def __init__(self, first_stereo, second_stereo, K, m2):
        self._first_stereo = first_stereo
        self._second_stereo = second_stereo
        self._K = K
        self._m2 = m2

        # data over time
        self._first_left_kpts = None
        self._first_right_kpts = None
        self._second_left_kpts = None
        self._second_right_kpts = None
        self._first_3D = None
        self._second_3D = None

        self._matches = None

        # supporters data (given some T)
        self._transformation = None
        self._supp_num = None
        self._supp_first_left = None
        self._supp_second_left = None
        self._supp_idxs = None

        # Best transformation
        self._best_transformation = None
        self._best_supporters_idx = None

    def match(self):
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        # matches = bf.knnMatch(first_pair_data["left_dscs"], second_pair_data["left_dscs"], k=2)
        self._matches = bf.match(self._first_stereo.get_left_dscs(), self._second_stereo.get_left_dscs())
        # Save the matches that passed
        self._first_left_kpts = [self._first_stereo.get_left_kpts()[match.queryIdx] for match in self._matches]
        self._first_right_kpts = [self._first_stereo.get_right_kpts()[match.queryIdx] for match in self._matches]
        self._second_left_kpts = [self._second_stereo.get_left_kpts()[match.trainIdx] for match in self._matches]
        self._second_right_kpts = [self._second_stereo.get_right_kpts()[match.trainIdx] for match in self._matches]
        self._first_3D = [self._first_stereo.get_three_d_pts()[match.queryIdx] for match in self._matches]
        self._second_3D = [self._second_stereo.get_three_d_pts()[match.trainIdx] for match in self._matches]

    def match_and_filter_using_significance(self):
        """
        Match across time using Knn and filter based on significance test
        """
        # find matches between two lefts
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(self._first_stereo.get_left_dscs(), self._second_stereo.get_left_dscs(), k=2)

        # filter using significance
        good_match_idxs = []
        for i in range(len(matches)):
            m1, m2 = matches[i][0], matches[i][1]
            if m1.distance < RATIO * m2.distance:
                good_match_idxs.append(i)

        # Save the matches that passed
        self._first_left_kpts = [self._first_stereo.get_left_kpts()[matches[i][0].queryIdx] for i in good_match_idxs]
        self._first_right_kpts = [self._first_stereo.get_right_kpts()[matches[i][0].queryIdx] for i in good_match_idxs]
        self._second_left_kpts = [self._second_stereo.get_left_kpts()[matches[i][0].trainIdx] for i in good_match_idxs]
        self._second_right_kpts = [self._second_stereo.get_right_kpts()[matches[i][0].trainIdx] for i in good_match_idxs]
        self._first_3D = [self._first_stereo.get_three_d_pts()[matches[i][0].queryIdx] for i in good_match_idxs]
        self._second_3D = [self._second_stereo.get_three_d_pts()[matches[i][0].trainIdx] for i in good_match_idxs]

    def find_supporters(self, cur_transformation):
        """
        Given transformation - find the supporters of that T
        Save to corresponding fields.
        """
        first_3D = np.array(self._first_3D)

        # get the cameras matrices
        left_intrinsic_mat = self._first_stereo.get_left_intrinsic_mat()
        right_intrinsic_mat = self._first_stereo.get_right_intrinsic_mat()


        # calc the projection of the three-d point onto all 4 cameras
        npl0 = left_intrinsic_mat[:, :3] @ first_3D.T + left_intrinsic_mat[:, -1:]
        npl0 /= npl0[2]

        npr0 = right_intrinsic_mat[:, :3] @ first_3D.T + right_intrinsic_mat[:, -1:]
        npr0 /= npr0[2]

        npl1r1_pre = cur_transformation[:, :3] @ first_3D.T + cur_transformation[:, -1:]

        npl1 = left_intrinsic_mat[:, :3] @ npl1r1_pre + left_intrinsic_mat[:, -1:]
        npl1 /= npl1[2]

        npr1 = right_intrinsic_mat[:, :3] @ npl1r1_pre + right_intrinsic_mat[:, -1:]
        npr1 /= npr1[2]

        l0_pixels = np.array(self._first_left_kpts)
        r0_pixels = np.array(self._first_right_kpts)
        l1_pixels = np.array(self._second_left_kpts)
        r1_pixels = np.array(self._second_right_kpts)

        l0_supporters = np.linalg.norm(npl0[:2] - l0_pixels.T, axis=0) < 2
        r0_supporters = np.linalg.norm(npr0[:2] - r0_pixels.T, axis=0) < 2
        l1_supporters = np.linalg.norm(npl1[:2] - l1_pixels.T, axis=0) < 2
        r1_supporters = np.linalg.norm(npr1[:2] - r1_pixels.T, axis=0) < 2
        supp = l0_supporters & r0_supporters & l1_supporters & r1_supporters

        self._supp_idxs = np.where(supp)[0]
        self._supp_num = self._supp_idxs.shape[0]
        self._supp_first_left = npl0[:2].T[self._supp_idxs]
        self._supp_second_left = npl1[:2].T[self._supp_idxs]

    def solve_pnp(self, n):
        """
        given the number of points to sample
        return the matching transformation using cv2-pnp-solver
        """
        selected_4_matches = np.random.choice(len(self._first_3D), n)

        three_d_of_4_matches_in_left_camera_t0_coords = [self._first_3D[i]
                                                         for i in selected_4_matches]
        pixel_locations_of_4_matches_in_left_camera_t1 = [self._second_left_kpts[i] for i in
                                                          selected_4_matches]

        # solve pnp given those 4 points
        success, pnp_R, pnp_t = cv2.solvePnP(objectPoints=np.array(three_d_of_4_matches_in_left_camera_t0_coords),
                                             imagePoints=np.array(pixel_locations_of_4_matches_in_left_camera_t1),
                                             cameraMatrix=self._K, distCoeffs=np.zeros((4, 1)), flags=cv2.SOLVEPNP_AP3P)

        # build transformation matrix
        if success:
            transformation_over_time = rodriguez_to_mat(pnp_R, pnp_t)
            self._transformation = transformation_over_time
        else:
            self._transformation = None

    def ransac(self):
        """
        perform ransac to extract the best transformation between the StereoFrames
        """
        num_samples = len(self._first_3D)
        num_of_iterations = 100
        max_num_of_supporters = 1

        while num_of_iterations > 1:
            self.solve_pnp(4)
            if self._transformation is not None:  # i.e. succeeded in pnp
                self.find_supporters(self._transformation)
                num_of_iterations -= 1
                if self._supp_num > max_num_of_supporters:
                    max_num_of_supporters = self._supp_num
                    num_of_iterations = get_ransac_num_iteration(num_samples, max_num_of_supporters,
                                                                 probability=RANSAC_PROBABILITY)
                    self._best_transformation = deepcopy(self._transformation)
                    self._best_supporters_idx = deepcopy(self._supp_idxs)

    def refine_transformation(self):
        """
        Refine the best transformation found in the ransac process, using all of the supporters
        """
        inliers_three_d_points_in_left_t0 = np.array([self._first_3D[i] for i in self._best_supporters_idx])
        inliers_pixel_location_in_left_camera_t1 = np.array([[self._second_left_kpts[i] for i in
                                                              self._best_supporters_idx]])
        success, pnp_R, pnp_t = cv2.solvePnP(objectPoints=inliers_three_d_points_in_left_t0,
                                             imagePoints=inliers_pixel_location_in_left_camera_t1,
                                             cameraMatrix=self._K, distCoeffs=np.zeros((4, 1)), flags=cv2.SOLVEPNP_EPNP)

        # build transformation matrix
        if success:
            transformation_over_time = rodriguez_to_mat(pnp_R, pnp_t)
            self._best_transformation = transformation_over_time

    def get_first_three_d(self):
        """
        returns first pair three d points
        AFTER MATCHING OVER TIME
        """
        return self._first_3D

    def get_second_three_d(self):
        """
        returns second pair three d points
        AFTER MATCHING OVER TIME
        """
        return self._second_3D

    def get_first_left_kpts(self):
        """
        returns first pair left kpts
        AFTER MATCHING OVER TIME
        """
        return self._first_left_kpts

    def get_first_right_kpts(self):
        """
        returns first pair righ kpts
        AFTER MATCHING OVER TIME
        """
        return self._first_right_kpts

    def get_second_left_kpts(self):
        """
        returns second pair left kpts
        AFTER MATCHING OVER TIME
        """
        return self._second_left_kpts

    def get_second_right_kpts(self):
        """
        returns second pair right kpts
        AFTER MATCHING OVER TIME
        """
        return self._second_right_kpts

    def get_current_transformation(self):
        return self._transformation

    def get_best_transformation(self):
        return self._best_transformation

    def get_first_stereo(self):
        """
        First stereo data before filtering over time.
        """
        return self._first_stereo

    def get_second_stereo(self):
        """
        Second stereo data before filtering over time.
        """
        return self._second_stereo

    def get_best_supporters_idx(self):
        return self._best_supporters_idx

    def get_best_left_projection(self):
        return self._K @ self._best_transformation

    def get_best_right_projection(self):
        return self._K @ two_transformations_composition(self._best_transformation, self._m2)

    def get_matches(self):
        return self._matches
