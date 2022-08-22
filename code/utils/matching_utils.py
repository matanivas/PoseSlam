import numpy as np
from VAN_project.code.utils import utils
import cv2
from VAN_project.code.utils.rts_utils import rodriguez_to_mat
from VAN_project.code.TracksDir.track import Track
from VAN_project.code.utils.rts_utils import  project

RATIO = 0.8
RANSAC_PROBABILITY = 0.85


class Uid:
    def __init__(self):
        self.uid = -1

    def get_next_uid(self):
        self.uid += 1
        return self.uid


def get_y_distances(matches, kpts1, kpts2):
    """
    given matches across left and right image - return array with the y distance between matching key-points
    """
    ans = []
    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        (x1, y1) = kpts1[img1_idx].pt
        (x2, y2) = kpts2[img2_idx].pt

        ans.append(np.abs(y2 - y1))
    return ans


def use_all_supporters_to_refine_transformation(images_over_time_data, K, best_idxs):
    """
    given the data of images at time t and t+1 - and the indexes of the kpts-inliers in the best transformation so far
    refine the transformation
    """
    inliers_three_d_points_in_left_t0 = np.array([images_over_time_data["t0_3D"][i] for i in best_idxs])
    inliers_pixel_location_in_left_camera_t1 = np.array([[images_over_time_data["left_t1_kpts"][i] for i in
                                                          best_idxs]])
    success, pnp_R, pnp_t = cv2.solvePnP(objectPoints=inliers_three_d_points_in_left_t0,
                                         imagePoints=inliers_pixel_location_in_left_camera_t1,
                                         cameraMatrix=K, distCoeffs=np.zeros((4, 1)), flags=cv2.SOLVEPNP_EPNP)

    # build transformation matrix
    if success:
        transformation_over_time = rodriguez_to_mat(pnp_R, pnp_t)
        return transformation_over_time
    else:
        return False


def get_ransac_num_iteration(num_samples, num_supporters, probability):
    """
    given number of samples, number of supporters and desired probability of success - return the desired number of
    itrerations
    """
    # num of iterations should be > log(0.1)/log(1-(0.1)^(s))
    # assume we want probability 0.9
    epsilon = max(1 - num_supporters / num_samples, 0.01)
    denumerator = min(-0.01, np.log(1 - np.power((1 - epsilon), num_samples)))
    return min(np.ceil(np.log(1 - probability) / denumerator), 200)


def sample_n_points_and_solve_pnp(n, K, images_over_time_data):
    """
    given the data of images at time t and t+1, and the number of points to sample
    return the matching transformation using cv2-pnp-solver
    """
    selected_4_matches = np.random.choice(len(images_over_time_data["t0_3D"]), n)

    three_d_of_4_matches_in_left_camera_t0_coords = [images_over_time_data["t0_3D"][i]
                                                     for i in selected_4_matches]
    pixel_locations_of_4_matches_in_left_camera_t1 = [images_over_time_data["left_t1_kpts"][i] for i in
                                                      selected_4_matches]

    # solve pnp given those 4 points
    success, pnp_R, pnp_t = cv2.solvePnP(objectPoints=np.array(three_d_of_4_matches_in_left_camera_t0_coords),
                                         imagePoints=np.array(pixel_locations_of_4_matches_in_left_camera_t1),
                                         cameraMatrix=K, distCoeffs=np.zeros((4, 1)), flags=cv2.SOLVEPNP_AP3P)

    # build transformation matrix
    if success:
        transformation_over_time = rodriguez_to_mat(pnp_R, pnp_t)
        return transformation_over_time
    else:
        return False


def get_num_supporters(T, left_intrinsic_mat, right_intrinsic_mat, data_over_time):
    """
    T - is the transformation we got from the pnp given 4 points
    The function returns the number
    """
    t0_3d = np.array(data_over_time["t0_3D"])
    # calc the projection of the three-d point onto all 4 cameras

    npl0 = left_intrinsic_mat[:, :3] @ t0_3d.T + left_intrinsic_mat[:, -1:]
    npl0 /= npl0[2]

    npr0 = right_intrinsic_mat[:, :3] @ t0_3d.T + right_intrinsic_mat[:, -1:]
    npr0 /= npr0[2]

    npl1r1_pre = T[:, :3] @ t0_3d.T + T[:, -1:]

    npl1 = left_intrinsic_mat[:, :3] @ npl1r1_pre + left_intrinsic_mat[:, -1:]
    npl1 /= npl1[2]

    npr1 = right_intrinsic_mat[:, :3] @ npl1r1_pre + right_intrinsic_mat[:, -1:]
    npr1 /= npr1[2]

    l0_pixels = np.array(data_over_time["left_t0_kpts"])
    r0_pixels = np.array(data_over_time["right_t0_kpts"])
    l1_pixels = np.array(data_over_time["left_t1_kpts"])
    r1_pixels = np.array(data_over_time["right_t1_kpts"])

    l0_supporters = np.linalg.norm(npl0[:2] - l0_pixels.T, axis=0) < 2
    r0_supporters = np.linalg.norm(npr0[:2] - r0_pixels.T, axis=0) < 2
    l1_supporters = np.linalg.norm(npl1[:2] - l1_pixels.T, axis=0) < 2
    r1_supporters = np.linalg.norm(npr1[:2] - r1_pixels.T, axis=0) < 2
    supp = l0_supporters & r0_supporters & l1_supporters & r1_supporters
    supp_idxs = np.where(supp)[0]
    supp_num = supp_idxs.shape[0]
    supp_l0 = npl0[:2].T[supp_idxs]
    supp_l1 = npl1[:2].T[supp_idxs]

    return supp_num, supp_l0, supp_l1, supp_idxs


def get_kpts_after_significance_for_stereo_images(matches, img1_kpts, img2_kpts, img1_dscs, img2_dscs):
    """
    match between descriptors of right and left images - filtering based on significance
    """
    good_match_idxs = []
    for i in range(len(matches)):
        m1, m2 = matches[i][0], matches[i][1]
        if m1.distance < RATIO * m2.distance:
            good_match_idxs.append(i)
    output = {
        "left_kpts": np.array([img1_kpts[matches[i][0].queryIdx] for i in good_match_idxs]),
        "left_dscs": np.array([img1_dscs[matches[i][0].queryIdx] for i in good_match_idxs]),
        "right_kpts": np.array([img2_kpts[matches[i][0].trainIdx] for i in good_match_idxs]),
        "right_dscs": np.array([img2_dscs[matches[i][0].trainIdx] for i in good_match_idxs])
    }
    return output


def get_kpts_after_significance(matches, first_pair_data, second_pair_data):
    """
    match between descriptors of two left images
    """
    good_match_idxs = []
    for i in range(len(matches)):
        m1, m2 = matches[i][0], matches[i][1]
        if m1.distance < RATIO * m2.distance:
            good_match_idxs.append(i)
    output = {
        "left_t0_kpts": [first_pair_data["left_kpts"][matches[i][0].queryIdx] for i in good_match_idxs],
        "right_t0_kpts": [first_pair_data["right_kpts"][matches[i][0].queryIdx] for i in good_match_idxs],
        "left_t1_kpts": [second_pair_data["left_kpts"][matches[i][0].trainIdx] for i in good_match_idxs],
        "right_t1_kpts": [second_pair_data["right_kpts"][matches[i][0].trainIdx] for i in good_match_idxs],
        "t0_3D": [first_pair_data["cv2T"][matches[i][0].queryIdx] for i in good_match_idxs],
        "t1_3D": [second_pair_data["cv2T"][matches[i][0].trainIdx] for i in good_match_idxs]
    }
    return output


def get_kpts_after_significance_with_track_db(matches, first_pair_data, second_pair_data):
    """
    match between descriptors of two left images - and update track db. Todo split from the db creation to unite functions
    """
    good_match_idxs = [i for i in range(len(matches))]
    # # another option
    # for i in range(len(matches)):
    #     m1, m2 = matches[i][0], matches[i][1]
    #     if m1.distance < ex1.RATIO * m2.distance:
    #         good_match_idxs.append(i)
    output = {
        "left_t0_kpts": [first_pair_data["left_kpts"][matches[i].queryIdx] for i in good_match_idxs],
        "right_t0_kpts": [first_pair_data["right_kpts"][matches[i].queryIdx] for i in good_match_idxs],
        "left_t1_kpts": [second_pair_data["left_kpts"][matches[i].trainIdx] for i in good_match_idxs],
        "right_t1_kpts": [second_pair_data["right_kpts"][matches[i].trainIdx] for i in good_match_idxs],
        "t0_3D": [first_pair_data["cv2T"][matches[i].queryIdx] for i in good_match_idxs],
        "t1_3D": [second_pair_data["cv2T"][matches[i].trainIdx]for i in good_match_idxs]
    }

    # track_ids = [-1] * len(second_pair_data["left_kpts"])
    # for idx in good_match_idxs:
    #     cur_track_id = frame_to_track_dict[str(frameId - 1)]["track_ids"][matches[idx][0].queryIdx]
    #     # if track_ids[matches[idx][0].trainIdx] != -1:
    #     #     continue
    #
    #     # update list of track ids for frame_to_trackids dict
    #     track_ids[matches[idx][0].trainIdx] = cur_track_id
    #     # update tuple of (first_frame, last_frame) for the track-id
    #
    #     # update all tracks dict
    #     coords_to_push = (second_pair_data["left_kpts"][matches[idx][0].trainIdx][0],
    #                       second_pair_data["right_kpts"][matches[idx][0].trainIdx][0],
    #                       (second_pair_data["left_kpts"][matches[idx][0].trainIdx][1] +
    #                       second_pair_data["right_kpts"][matches[idx][0].trainIdx][1]) / 2.0)
    #     all_tracks_dict[str(cur_track_id)].push_point_to_track(coords_to_push)
    # for i, id in enumerate(track_ids):
    #     if id == -1:
    #         next_uid = uid_generator.get_next_uid()
    #         track_ids[i] = next_uid
    #
    #         # push the new track to all_tracks_dict
    #         coords_to_push = (second_pair_data["left_kpts"][i][0],
    #                           second_pair_data["right_kpts"][i][0],
    #                           (second_pair_data["left_kpts"][i][1] +
    #                            second_pair_data["right_kpts"][i][1]) / 2.0)
    #         all_tracks_dict[str(next_uid)] = Track(next_uid, frameId, coords_to_push)
    #
    # frame_to_track_dict[str(frameId)] = {
    #     "left_kpts": second_pair_data["left_kpts"],
    #     "right_kpts": second_pair_data["right_kpts"],
    #     "track_ids": track_ids,
    #     "inliers percentage": (len(output["left_t0_kpts"]) / len(matches)) * 100
    # }
    return output, good_match_idxs


def update_tracks_db(best_projection_left, best_projection_right, images_data_time, first_pair, second_pair, matches,
                     good_match_idxs, frame_to_track_dict, all_tracks_dict, cur_frame_id, uid_generator,
                     filter_tracks_based_on_rt=False, return_filtered_inliers=False):
    """
    update the db of tracks, while using the best RT between the frames to reject matches that are not good enough
    :param best_projection_left: Rt from prev frame system, to coords on the cur left frame
    :param best_projection_right: Rt from prev frame system, to coords on the cur right frame
    :param images_data_time:
    :param second_pair:
    :param matches:
    :param good_match_idxs:
    :param filter_tracks_based_on_rt:
    :return:
    """
    track_ids = [-1] * len(second_pair["left_kpts"])
    number_of_matches_rejected_using_rt = 0
    final_good_match_idxs = [] # the index of matches after filtering based on Rt
    for idx in good_match_idxs:
        # get the track id of the point from the previous frame that matched
        cur_track_id = frame_to_track_dict[str(cur_frame_id - 1)]["track_ids"][matches[idx].queryIdx]


        # get the coordinates that match the point-track in the current frame. (x_left, x_right, y(avg))
        coords_to_push = (second_pair["left_kpts"][matches[idx].trainIdx][0],
                          second_pair["right_kpts"][matches[idx].trainIdx][0],
                          (second_pair["left_kpts"][matches[idx].trainIdx][1] +
                           second_pair["right_kpts"][matches[idx].trainIdx][1]) / 2.0)

        if filter_tracks_based_on_rt:
            # move the 3d point from the coordinates of the previous frame to the coordinates of the
            # current frame, project it on the cameras - and then calc the delta between the projection and
            # coords to push - if is large - continue (i.e. don't update the track (bad point).

            p3_in_prev_frame_world = first_pair["cv2T"][matches[idx].queryIdx]
            # move the p3 to the world of the current frame, and project
            proj_cur_left = project(p3_in_prev_frame_world, best_projection_left)
            proj_cur_right = project(p3_in_prev_frame_world, best_projection_right)
            proj_corrds = np.array([proj_cur_left[0], proj_cur_right[0], (proj_cur_left[1] + proj_cur_right[1]) / 2.0])

            # calc dist between proj coords and detected coords - if large - reject match - i.e.
            # don't extend the prev track, rather create a new track from the point in the current frame
            # (happens in the next for loop)
            diff = np.linalg.norm([coords_to_push - proj_corrds])

            if diff > 1.5:  # todo adjust this threshold
                number_of_matches_rejected_using_rt += 1
                continue
            else:
                # update list of track ids for frame_to_trackids dict
                track_ids[matches[idx].trainIdx] = cur_track_id
                final_good_match_idxs.append(idx)

        # update all tracks dict
        coords_to_push = (second_pair["left_kpts"][matches[idx].trainIdx][0],
                          second_pair["right_kpts"][matches[idx].trainIdx][0],
                          (second_pair["right_kpts"][matches[idx].trainIdx][1] +
                          second_pair["right_kpts"][matches[idx].trainIdx][1]) / 2.0)
        all_tracks_dict[str(cur_track_id)].push_point_to_track(coords_to_push)
    for i, id in enumerate(track_ids):
        if id == -1:  # i.e. it's a point that was not matched to some previous point. >> start a new track
            next_uid = uid_generator.get_next_uid()

            track_ids[i] = next_uid

            # push the new track to all_tracks_dict
            coords_to_push = (second_pair["left_kpts"][i][0],
                              second_pair["right_kpts"][i][0],
                              (second_pair["left_kpts"][i][1] +
                               second_pair["right_kpts"][i][1]) / 2.0)
            all_tracks_dict[str(next_uid)] = Track(next_uid, cur_frame_id, coords_to_push)

    frame_to_track_dict[str(cur_frame_id)] = {
        "left_kpts": second_pair["left_kpts"],
        "right_kpts": second_pair["right_kpts"],
        "track_ids": track_ids,
        "inliers percentage": (len(images_data_time["left_t0_kpts"]) / len(matches)) * 100,
        "inlier_percentage_after_rt": (len(images_data_time["left_t0_kpts"]) - number_of_matches_rejected_using_rt) / len(matches) *100,
        "num_matches_rejected_using_rt": number_of_matches_rejected_using_rt
    }
    # print("At frame {} the number of rejected matches based on rt is {} out of {} possible matches".format(cur_frame_id, number_of_matches_rejected_using_rt, len(images_data_time["left_t0_kpts"])))
    if return_filtered_inliers:
        return final_good_match_idxs
    else:
        return None

def get_kpts_dscs_after_stereo(matches, kpts1, kpts2, dscs1, dscs2):
    """
    given matches between left and right - return only those that match the stereo pattern
    """
    dists = get_y_distances(matches, kpts1, kpts2)
    img1_kpts = []
    img1_dscs = []
    img2_kpts = []
    img2_dscs = []
    for i in range(len(dists)):
        if dists[i] < 1.0:
            img1_kpts.append(kpts1[matches[i].queryIdx].pt)
            img1_dscs.append(dscs1[matches[i].queryIdx])
            img2_kpts.append(kpts2[matches[i].trainIdx].pt)
            img2_dscs.append(dscs2[matches[i].trainIdx])

    return np.array(img1_kpts), np.array(img1_dscs), np.array(img2_kpts), np.array(img2_dscs)


def get_triangulation(idx, left_img_m, right_img_m, triangulate=False):
    """
    get triangualtion given: images index, images intrinsic matrices,
    """

    # read images
    img1, img2 = utils.read_images(idx)

    # detect keypoints
    alg = cv2.AKAZE_create()
    img1_kpts, img1_dscs = alg.detectAndCompute(img1, None)
    img2_kpts, img2_dscs = alg.detectAndCompute(img2, None)

    # match between the keypoints of the two images
    brute_force = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matching = brute_force.match(img1_dscs, img2_dscs)

    # discard matches using stereo pattern
    img1_coords, img1_dscs_after_filter, img2_coords, img2_dscs_after_filter \
        = get_kpts_dscs_after_stereo(matching, img1_kpts, img2_kpts, img1_dscs, img2_dscs)

    # discard matches using significance
    # bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    # second_matching = bf.knnMatch(img1_dscs_after_filter, img2_dscs_after_filter, k=2)
    # output_after_significance_filtering = get_kpts_after_significance_for_stereo_images(second_matching, img1_coords, img2_coords, img1_dscs_after_filter, img2_dscs_after_filter)
    # #kaka
    # img1_coords = output_after_significance_filtering["left_kpts"]
    # img2_coords = output_after_significance_filtering["right_kpts"]

    # get triangulation
    if triangulate:
        cv2T = cv2.triangulatePoints(left_img_m, right_img_m, img1_coords.T, img2_coords.T).T
        cv2T = cv2.convertPointsFromHomogeneous(cv2T).reshape(img1_coords.shape[0], 3)
    else:
        cv2T = None

    # create data structure to store all images relevant data
    output = {"left_kpts": img1_coords,
              # "left_dscs": output_after_significance_filtering["left_dscs"],
              # "right_dscs": output_after_significance_filtering["right_dscs"],
              "left_dscs": img1_dscs_after_filter,
              "right_dscs": img2_dscs_after_filter,
              "right_kpts": img2_coords,
              "cv2T": cv2T,
              "left": img1,
              "right": img2
              }
    return output


def ransac(K, images_over_time_data, img1_m, img2_m):
    """ perform ransac given stereo of t and t+1
    :param K:
    :param images_over_time_data:
    :param img1_m:
    :param img2_m:
    :param num_supporters_at_beginning:
    :param transformation_at_beginning:
    :param supporter_idxs_at_beginning:
    :return:
    """
    num_samples = len(images_over_time_data["t0_3D"])
    # todo remove default values
    num_of_iterations = 5000
    max_num_of_supporters = 1
    best_transformation = None
    best_supporters_idx = None

    while num_of_iterations > 1:
        transformation_over_time = sample_n_points_and_solve_pnp(4, K, images_over_time_data)
        if transformation_over_time is not False:  # i.e. succeeded in pnp
            num_supporters, supporters_l0, supporters_l1, supporters_idxs = \
                get_num_supporters(transformation_over_time, img1_m, img2_m, images_over_time_data)

            num_of_iterations -= 1
            if num_supporters > max_num_of_supporters:
                max_num_of_supporters = num_supporters
                num_of_iterations = get_ransac_num_iteration(num_samples, max_num_of_supporters, probability=RANSAC_PROBABILITY)
                best_transformation = transformation_over_time
                best_supporters_idx = supporters_idxs
                # print("Current #(supporters): {}, implied #(iterations): {}".format(max_num_of_supporters, num_of_iterations))
            # elif num_of_iterations % 100 == 0:
            #     print("Current #(iterations): {}".format(num_of_iterations))
    return best_transformation, best_supporters_idx
