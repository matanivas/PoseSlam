import cv2
import gtsam
import matplotlib.pyplot as plt
import numpy as np
from gtsam.utils import plot
from VAN_project.code.utils import rts_utils
from VAN_project.code.GraphsOptimization.BundleDir.BundleGraph import BundleGraph
from VAN_project.code.utils import db_utils
from VAN_project.code.utils import paths
from VAN_project.code.utils import utils
from VAN_project.code.exercises import ex4_analysis
import random
import statistics
COLORS_LIST = ["blue", "orange", "grey"]


def print_matches_inliers_per_loop_closure(circles_meta, output_dir):
    """
    Write the number of matches found in each loop closure - todo change what we save in the dicts
    Write the percentage of inliers found in each loop closure
    """
    with open(output_dir, 'w') as f:
        for match_key in circles_meta.keys():
            frames = match_key.split('-')
            first_frame = frames[0]
            second_frame = frames[1]
            cur_inliers_data = circles_meta[match_key]["inliers_dict"][second_frame]
            cur_matches_data_first = circles_meta[match_key]["num_matches"][first_frame]
            cur_matches_data_second = circles_meta[match_key]["num_matches"][second_frame]
            f.write("For frames - {}, #(matches) in first frame {}, in second frame {}."
                    "The percentage of inliers: {}".format(match_key, cur_matches_data_first, cur_matches_data_second, cur_inliers_data))
            f.write('\n')
    f.close()


def plot_matches_per_frames(matches_per_frame_dict, output_dir):
    matches = []
    all_frames_idxs = range(len(matches_per_frame_dict.keys()))

    for frame_idx in all_frames_idxs:
        matches.append(matches_per_frame_dict[str(frame_idx)])

    plt.plot(all_frames_idxs, matches)
    plt.xlabel('Frame')
    plt.ylabel('Num. Matches')
    plt.title('Number of matches per frame')
    plt.savefig('{}/num_matches_per_frame.png'.format(output_dir))
    plt.close()


def plot_relative_position_of_cameras(list_of_lists_of_pos, title, output_fig_name):
    """
    plot the track the car drove
    list of poses are in GLOBAL!
    """
    for i, pos_list in enumerate(list_of_lists_of_pos):
        x_left, z_left = pos_list[:, :1], pos_list[:, -1:]
        # plot
        plt.scatter(x_left, z_left, color=COLORS_LIST[i % len(COLORS_LIST)], s=(1.0/(i+1)))
    plt.xlabel("X axis")
    plt.ylabel("Z axis")
    plt.title("{}".format(title))
    plt.savefig(output_fig_name)
    plt.close()


def get_gt_rts_global_positions():
    # read the corresponding GT positions for the keyframes
    gt_transformations = db_utils.read_gt_transformations(paths.GT_PATH)
    gt_locations_in_l0_coords = [-1 * Rt[:, :3].T @ Rt[:, 3:] for Rt in gt_transformations]
    return gt_locations_in_l0_coords


def get_gt_rts_global_rotations():
    # read the corresponding GT rotation matrices for the keyframes
    gt_transformations = db_utils.read_gt_transformations(paths.GT_PATH)
    return [Rt[:, :3] for Rt in gt_transformations]


def plot_pose_graph_poses(result, initial_estimate, keys, title, output_fig_name):
    """
    given the result of a pos graph, the symbols of the poses in it (keys) - create a plot of the trajectory
    :param result: the result of the optimization of the pos graph
    :param keys: the symbols of the poses in the graph
    :param title: the title of the plot
    :param output_fig_name: the path to save the plot in
    :return: Nome
    """
    # get the gt locations
    gt_rts = get_gt_rts_global_positions()

    # extract the optimized locations from the graph
    result_camera_locations = []
    for key in keys:
        camera_location = result.atPose3(key)
        result_camera_locations.append(camera_location)
    positions_all_optimized_in_global = np.array([(rt.x(), rt.y(), rt.z()) for rt in result_camera_locations])

    # extract the optimized locations from the graph
    initial_camera_locations = []
    for key in keys:
        camera_location = initial_estimate.atPose3(key)
        initial_camera_locations.append(camera_location)
    positions_orig_in_global = np.array([(rt.x(), rt.y(), rt.z()) for rt in initial_camera_locations])

    plot_relative_position_of_cameras([np.array(gt_rts), np.array(positions_all_optimized_in_global), np.array(positions_orig_in_global)], title, output_fig_name)


def plot_accepted_and_rejected(frame_1_idx, frame_2_idx, img1_accepted_coors, img1_rejected_coors,
                               img2_accepted_coors, img2_rejected_coors):
    """
    plot on images the accepted and rejected coords
    """
    img1, _ = utils.read_images(frame_1_idx)
    img2, _ = utils.read_images(frame_2_idx)
    figure = plt.figure(figsize=(9, 6))
    figure.add_subplot(2, 1, 1)
    img1_accepted_coors = np.array(img1_accepted_coors)
    img1_rejected_coors = np.array(img1_rejected_coors)
    img2_accepted_coors = np.array(img2_accepted_coors)
    img2_rejected_coors = np.array(img2_rejected_coors)
    plt.title('Frame {} orange: inliers, cyan: outliers'.format(frame_1_idx))
    plt.imshow(img1, cmap='gray')
    plt.scatter(img1_accepted_coors[:, 0], img1_accepted_coors[:, 1], s=2, color='orange')
    plt.scatter(img1_rejected_coors[:, 0], img1_rejected_coors[:, 1], s=2, color='cyan')

    figure.add_subplot(2, 1, 2)
    plt.title('Frame {} orange: inliers, cyan: outliers'.format(frame_2_idx))
    plt.imshow(img2, cmap='gray')
    plt.scatter(img2_accepted_coors[:, 0], img2_accepted_coors[:, 1], s=2, color='orange')
    plt.scatter(img2_rejected_coors[:, 0], img2_rejected_coors[:, 1], s=2, color='cyan')
    plt.show()


def plot_pose_graph(fignum, gtsam_values, frame_idx_iteration, marginals=None):
    plot.plot_trajectory(fignum, gtsam_values, marginals=marginals, scale=1, title="Version of the pose graph after frame {} (global idx)".format(frame_idx_iteration), d2_view=True)
    plt.savefig("Rotated 3D Trajectory at frame {} (global idx)".format(frame_idx_iteration))
    plt.close()

#
# def plot_pose_graph_absolute_location_error(pose_graph, title, output_directory):
#     poses, key_frames = pose_graph.get_all_poses_and_global_frames()
#     gt_rts_global = get_gt_rts_global_positions()
#     key_frames_gt_rts_global = np.array([gt_rts_global[i] for i in key_frames]).reshape((len(poses), 3))
#     poses_locations = np.array([(rt.x(), rt.y(), rt.z()) for rt in poses])
#     plot_absolute_location_error(poses_locations, key_frames, title, output_directory)
#     xs = poses_locations.T[0]
#     ys = poses_locations.T[1]
#     zs = poses_locations.T[2]
#
#     xs_global = key_frames_gt_rts_global.T[0]
#     ys_global = key_frames_gt_rts_global.T[1]
#     zs_global = key_frames_gt_rts_global.T[2]
#
#     xs_error = np.abs(xs - xs_global)
#     ys_error = np.abs(ys - ys_global)
#     zs_error = np.abs(zs - zs_global)
#
#     num_of_poses = len(poses)
#     diff = np.subtract(poses_locations, key_frames_gt_rts_global)
#     norm = np.sum(np.abs(diff) ** 2, axis=-1) ** (1. / 2)
#     plt.plot(range(num_of_poses), norm, 'r', label='norm error')
#     plt.plot(range(num_of_poses), xs_error, 'g', label='x axis error')
#     plt.plot(range(num_of_poses), ys_error, 'b', label='y axis error')
#     plt.plot(range(num_of_poses), zs_error, 'y', label='z axis error')
#     plt.xlabel('Keyframes')
#     plt.ylabel('Distance')
#     plt.title('Distance measure of poses from gt over the keyframes {} '.format(title))
#     plt.legend()
#     plt.savefig('{}/Distance metric of our poses and gt over the keyframes {} '.format(output_directory, title))
#     plt.close()


def plot_location_uncertainty_size_of_pose_graph_no_loops(pose_graph, output_directory):
    marginal = gtsam.Marginals(pose_graph.graph, pose_graph.result)
    prev_symbol = pose_graph.poses_symbols[0]
    key = gtsam.KeyVector()
    key.append(prev_symbol)
    cov = marginal.jointMarginalCovariance(key).at(key[-1], key[-1])
    covs = [cov]
    dets_first = [np.linalg.det(cov)]
    for i, symbol in enumerate(pose_graph.poses_symbols[1:]):
        print(i, symbol)
        key = gtsam.KeyVector()
        key.append(prev_symbol)
        key.append(symbol)
        cov = marginal.jointMarginalCovariance(key).at(key[-1], key[-1])
        covs.append(cov + covs[-1])
        dets_first.append(np.linalg.det(cov) + dets_first[-1])
        prev_symbol = symbol
    dets_covs = [np.linalg.det(cov) for cov in covs]
    plt.plot(range(len(dets_covs)), dets_covs)
    plt.xlabel('Pose')
    plt.ylabel('Measure of pose location uncertainty')
    plt.title('Measure of pose location uncertainty no loops ')
    plt.savefig('{}/Measure_of_pose_location_uncertainty_no_loops'.format(output_directory))
    # plt.show()
    plt.close()
    return dets_covs


def plot_location_uncertainty_size_of_pose_graph_with_loops(pose_graph, output_directory):
    # marginal = gtsam.Marginals(pose_graph.graph, pose_graph.result)
    covs = []
    first_symbol = pose_graph.poses_symbols[0]
    for i, symbol in enumerate(pose_graph.poses_symbols[1:]):
        print(i)
        shortest_path = pose_graph.get_shortest_path(first_symbol, symbol)
        sum_path_covs = get_paths_covs_sum(pose_graph, shortest_path)
        covs.append(sum_path_covs)
    dets_covs = [np.linalg.det(cov) for cov in covs]
    plt.plot(range(len(dets_covs)), dets_covs)
    plt.xlabel('Pose')
    plt.ylabel('Measure of pose location uncertainty')
    plt.title('Measure of pose location uncertainty with loops ')
    # plt.show()
    plt.savefig('{}/Measure_of_pose_location_uncertainty_with_loops'.format(output_directory))
    plt.close()
    return dets_covs


def get_paths_covs_sum(pose_graph, path):
    covs = []
    marginal = gtsam.Marginals(pose_graph.graph, pose_graph.result)
    prev_pose = path[0]
    for pose in path[1:]:
        key = gtsam.KeyVector()
        key.append(prev_pose)
        key.append(pose)
        covs.append(marginal.jointMarginalCovariance(key).at(key[-1], key[-1]))
    sum_covs = covs[0]
    for cov in covs[1:]:
        sum_covs += cov
    return sum_covs


def plot_num_of_matches_per_seccessfull_loop_closure(loop_closure_frames_lst, num_of_matches_lst, output_directory):
    plt.plot(loop_closure_frames_lst, num_of_matches_lst)
    plt.xlabel('Frames with loop closure')
    plt.ylabel('Number of matches')
    plt.title('Number of matches per successful loop closure frame')
    plt.savefig('{}/Number of matches per successful loop closure frame'.format(output_directory))
    plt.close()


def plot_inliers_percentage_per_seccessfull_loop_closure(loop_closure_frames_lst, inliers_percentage_lst, output_directory):
    plt.plot(loop_closure_frames_lst, inliers_percentage_lst)
    plt.xlabel('Frames with loop closure')
    plt.ylabel('inliers_percentage')
    plt.title('Inliers percentage per successful loop closure frame')
    plt.savefig('{}/Inlier percentage per successful loop closure frame'.format(output_directory))
    plt.close()


# def get_PnP_projection_error_per_distances(distances, pnp_data):
def pnp_median_projection_error_per_distance_from_reference(pnp_trans, all_frames, all_tracks, output_dir):
    # cameras matrices
    K, m1, m2 = utils.read_cameras()
    num_of_frames_to_sample = 100
    # Sample number of frames to sample the tracks from
    random.seed(10)
    frames = random.sample(all_frames.keys(), num_of_frames_to_sample)

    max_length_of_dist_per_track = 200
    distances_db = [[] for i in range(max_length_of_dist_per_track)]

    max_frame = max(list(map(int, list(all_frames.keys()))))

    for frame in frames:
        print("Analysing frame {}".format(frame))
        # get the tracks in that frame
        tracks_ids = all_frames[frame]

        for track_id in tracks_ids:
            print("Track id: {}".format(track_id))
            cur_track = all_tracks[track_id]
            projection_error = ex4_analysis.get_projection_dist_of_track(cur_track, pnp_trans, K, m1, m2, max_frame)
            for i, error in enumerate(projection_error):
                distances_db[i].append(error)

    furthest_frame = 0
    for i, dist in enumerate(distances_db):
        if len(dist) == 0:
            furthest_frame = i
            break
    # medians = [statistics.median(error_of_frame) for error_of_frame in distances_db if len(error_of_frame) > 0]
    medians = [statistics.median(distances_db[i]) for i in range(furthest_frame)]

    plt.plot(range(furthest_frame), medians)
    plt.xlabel('distances from reference')
    plt.ylabel('median projection error')
    plt.title('Median projection error - PnP')
    plt.savefig('{}/median_projection_error_pnp.png'.format(output_dir))
    plt.close()

def bundles_median_projection_error_per_distance_from_reference(bundles_manager, all_frames,
                                                        all_tracks, title,  output_directory):
    num_of_frames_to_sample = 100
    # Sample number of frames to sample the tracks from
    random.seed(10)
    frames = random.sample(all_frames.keys(), num_of_frames_to_sample)
    # Sample number of frames to sample the tracks from

    max_length_of_dist_per_track = 30
    distances_db = [[] for i in range(max_length_of_dist_per_track)]

    for frame in frames:
        print("Analysing frame {}".format(frame))
        # get the bundle the frame is in
        relevant_bundle = bundles_manager.get_bundle_that_contains_frame(int(frame))
        # get the tracks in that frame
        tracks_ids = all_frames[frame]

        for track_id in tracks_ids:
            print("Track id: {}".format(track_id))
            cur_track = all_tracks[track_id]
            projection_error = relevant_bundle.get_reprojection_error_of_track(cur_track)
            if projection_error is not None:
                for i, error in enumerate(projection_error):
                    distances_db[i].append(error)

    max_dist = 0
    for i, dist in enumerate(distances_db):
        if len(dist) < 1:
            max_dist = i
            break

    medians = [statistics.median(distances_db[i]) for i in range(max_dist)]

    plt.plot(range(max_dist), medians)
    plt.xlabel('distances from reference')
    plt.ylabel('median projection error')
    plt.title('Median projection error - {}'.format(title))
    plt.savefig('{}/median_projection_error_bundles'.format(output_directory, title))
    plt.close()


def get_gtsam_angle_error(R, Q):
    axis_vec, _ = cv2.Rodrigues(R.transpose().reshape(3, 3) @ Q.transpose().transpose().reshape(3, 3))
    return np.linalg.norm(axis_vec) * 180 / np.pi


def get_np_angle_error(R, Q):
    axis_vec, _ = cv2.Rodrigues(R.T @ Q)
    return np.linalg.norm(axis_vec) * 180 / np.pi

def plot_absolute_pnp_angle_estimation_error(pnp_transformations, output_directory):
    gt_rts_global = get_gt_rts_global_rotations()
    angle_estimation_errors = np.zeros((len(pnp_transformations), 1))
    for i, tr in enumerate(pnp_transformations):
        angle_estimation_errors[i] = get_np_angle_error(tr, gt_rts_global[i])
    idxs = [i for i in range(len(pnp_transformations))]
    plt.plot(idxs, angle_estimation_errors)
    plt.xlabel('Frame')
    plt.ylabel('Absolute angle error estimation')
    plt.title('Absolute angle error estimation- PnP')
    plt.savefig('{}/Absolute angle error estimation- PnP'.format(output_directory))
    plt.close()


def plot_absolute_pose_graph_angle_estimation_error(pose_graph, title, output_directory):
    poses, key_frames = pose_graph.get_all_poses_and_global_frames()
    # poses = utils.from_gtsam_relative_to_global(poses)
    poses = [pos.rotation().transpose() for pos in poses]
    gt_rts_global = get_gt_rts_global_rotations()
    key_frames_gt_rts_global = np.array([gt_rts_global[i] for i in key_frames]).reshape((len(poses), 3, 3))
    angle_estimation_errors = np.zeros((len(poses), 1))
    for i, pose in enumerate(poses):
        angle_estimation_errors[i] = get_gtsam_angle_error(pose, key_frames_gt_rts_global[i])
    plt.plot(key_frames, angle_estimation_errors)
    plt.xlabel('Frame')
    plt.ylabel('Absolute angle error estimation')
    plt.title('Pose graph absolute angle error estimation {}'.format(title))
    plt.savefig('{}/Pose graph absolute angle error estimation {}'.format(output_directory, title))
    plt.close()


def plot_posegraph_absolute_location_error(pose_graph, title,  output_directory):
    poses, key_frames = pose_graph.get_all_poses_and_global_frames()
    poses_locations = np.array([(rt.x(), rt.y(), rt.z()) for rt in poses])
    plot_absolute_location_error(poses_locations, key_frames, 'pose graph' + title, output_directory)


def plot_pnp_absolute_location_error(pnp_locations, output_directory):
    plot_absolute_location_error(pnp_locations, range(len(pnp_locations)), 'PnP', output_directory)


def plot_absolute_location_error(locations, idxs_in_global, title, output_directory):
    gt_rts_global = get_gt_rts_global_positions()
    gt_of_idxs = np.array([gt_rts_global[i] for i in idxs_in_global]).reshape((len(locations), 3))
    # xs = locations[:, 0]
    # ys = locations[:, 1]
    # zs = locations[:, 2]
    xs = locations.T[0]
    ys = locations.T[1]
    zs = locations.T[2]

    xs_global = gt_of_idxs.T[0]
    ys_global = gt_of_idxs.T[1]
    zs_global = gt_of_idxs.T[2]

    xs_error = np.abs(xs - xs_global)
    ys_error = np.abs(ys - ys_global)
    zs_error = np.abs(zs - zs_global)

    diff = np.subtract(locations.reshape(gt_of_idxs.shape), gt_of_idxs)
    norm = np.sum(np.abs(diff) ** 2, axis=-1) ** (1. / 2)

    plt.plot(idxs_in_global, norm, 'r', label='norm error')
    plt.plot(idxs_in_global, xs_error, 'g', label='x axis error')
    plt.plot(idxs_in_global, ys_error, 'b', label='y axis error')
    plt.plot(idxs_in_global, zs_error, 'y', label='z axis error')
    plt.xlabel('Keyframes')
    plt.ylabel('Distance')
    plt.title('Absolute location estimation error- {} '.format(title))
    plt.legend()
    plt.savefig('{}/Absolute location estimation error- {} '.format(output_directory, title))
    plt.close()


def plot_relative_pnp_estimation_location_error(transfomations_of_movie, length_seq, output_directory):
    gt_global = np.array(db_utils.read_gt_transformations(paths.GT_PATH)[1:])
    rts_global = utils.from_relative_to_global_transformtions(transfomations_of_movie)[1:]

    relative_rts = rel_seq(rts_global, length_seq)
    relative_rts_locs = relative_rts[:, :, -1]

    relatives_gt = rel_seq(gt_global, length_seq)
    relatives_gt_locs = relatives_gt[:, :, -1]

    xs = relative_rts_locs.T[0]
    ys = relative_rts_locs.T[1]
    zs = relative_rts_locs.T[2]

    xs_relative_gt = relatives_gt_locs.T[0]
    ys_relative_gt = relatives_gt_locs.T[1]
    zs_relative_gt = relatives_gt_locs.T[2]


    # gt_total_distances =

    xs_error = np.abs(xs - xs_relative_gt)
    ys_error = np.abs(ys - ys_relative_gt)
    zs_error = np.abs(zs - zs_relative_gt)

    xs_percantage_error = 100 * (xs_error / np.abs(xs_relative_gt))
    ys_percantage_error = 100 * (ys_error / np.abs(ys_relative_gt))
    zs_percantage_error = 100 * (zs_error / np.abs(zs_relative_gt))

    diff = np.subtract(relative_rts_locs, relatives_gt_locs)
    gt_norm = np.sum(np.abs(relatives_gt_locs) ** 2, axis=-1) ** (1. / 2)
    norm = np.sum(np.abs(diff) ** 2, axis=-1) ** (1. / 2)
    norm_percentage = 100 * (norm / gt_norm)


    plt.plot(range(len(norm)), norm, 'r', label='norm error')
    plt.plot(range(len(norm)), xs_error, 'g', label='x axis error')
    plt.plot(range(len(norm)), ys_error, 'b', label='y axis error')
    plt.plot(range(len(norm)), zs_error, 'y', label='z axis error')
    plt.xlabel('first transformation')
    plt.ylabel('Distance from GT')
    plt.title('PnP Relative location estimation percentage error- {} '.format(length_seq))
    plt.legend()
    plt.savefig('{}/PnP Relative location estimation error- length = {} '.format(output_directory, str(length_seq)))
    plt.close()

def plot_relative_bundle_estimation_location_error(poses_graph_no_loops, length_seq, output_directory):
    global_gt_rts_ctw = db_utils.read_gt_transformations(paths.GT_PATH)
    global_gt_rts_wtc = [utils.inverse_rt(global_ctw_rt) for global_ctw_rt in global_gt_rts_ctw] # switched to world to cam

    global_poses, global_frames = poses_graph_no_loops.get_all_poses_and_global_frames()
    # get the pairs of frames in global and idxs of them in the keyframes list
    global_frames_pairs, idxs = get_frames_length_closest_to_keyframes(global_frames, length_seq)

    relative_poses_of_length = []
    relative_gt_rts_of_length = []
    for tup_pair in idxs:
    # get relative pose of length
        first_pose = global_poses[tup_pair[0]]
        second_pose = global_poses[tup_pair[1]]
        relative_pose = first_pose.between(second_pose)
        relative_poses_of_length.append(relative_pose)

    # get relative gt rt of length
    for tup in global_frames_pairs:
        first_gt_rt = global_gt_rts_wtc[tup[0]]
        second_gt_rt = global_gt_rts_wtc[tup[1]]
        relative_gt_rts_of_length.append(relative_rt_world_to_cam(first_gt_rt, second_gt_rt))
    relative_gt_rts_of_length = np.array(relative_gt_rts_of_length)

    relative_bundle_estimation_locations = np.array([pose.translation() for pose in relative_poses_of_length])

    relative_gt_locations = relative_gt_rts_of_length[:, :, -1]



    xs = relative_bundle_estimation_locations.T[0]
    ys = relative_bundle_estimation_locations.T[1]
    zs = relative_bundle_estimation_locations.T[2]

    xs_relative_gt = relative_gt_locations.T[0]
    ys_relative_gt = relative_gt_locations.T[1]
    zs_relative_gt = relative_gt_locations.T[2]

    xs_error = np.abs(xs - xs_relative_gt)
    ys_error = np.abs(ys - ys_relative_gt)
    zs_error = np.abs(zs - zs_relative_gt)

    # todo check if that is the meaning of percantage errors
    xs_percantage_error = 100 * (xs_error / np.abs(xs_relative_gt))
    ys_percantage_error = 100 * (ys_error / np.abs(ys_relative_gt))
    zs_percantage_error = 100 * (zs_error / np.abs(zs_relative_gt))

    diff = np.abs(relative_bundle_estimation_locations - relative_gt_locations)
    gt_norm = np.sum(np.abs(relative_gt_locations) ** 2, axis=-1) ** (1. / 2)
    norm = np.sum(np.abs(diff) ** 2, axis=-1) ** (1. / 2)
    norm_percentage = 100 * (norm / gt_norm)


    plt.plot(range(len(norm)), norm_percentage, 'r', label='norm error')
    plt.plot(range(len(norm)), xs_percantage_error, 'g', label='x axis error')
    plt.plot(range(len(norm)), ys_percantage_error, 'b', label='y axis error')
    plt.plot(range(len(norm)), zs_percantage_error, 'y', label='z axis error')
    plt.xlabel('first transformation')
    plt.ylabel('Distance from GT in %')
    plt.title('Bundle Relative location estimation percentage error- {} '.format(length_seq))
    plt.legend()
    plt.savefig('{}/Bundle Relative location estimation error- length = {} '.format(output_directory, str(length_seq)))
    plt.close()




def get_frames_length_closest_to_keyframes(keyframes_lst, length_seq):
    pairs_of_len_seq = []
    idxs = []
    for i, key_frame in enumerate(keyframes_lst):
        if i + length_seq >= 3449:
            break
        designated_frame = key_frame + length_seq
        for j in range(i, len(keyframes_lst) - 1):
            if keyframes_lst[j] <= designated_frame <= keyframes_lst[j + 1]:
                if designated_frame - keyframes_lst[j] < keyframes_lst[j + 1] - designated_frame:
                    pairs_of_len_seq.append((key_frame, keyframes_lst[j]))
                    idxs.append((i, j))
                else:
                    pairs_of_len_seq.append((key_frame, keyframes_lst[j + 1]))
                    idxs.append((i, j + 1))
                break
    return pairs_of_len_seq, idxs



def relative_rt_world_to_cam(first_rt, second_rt):
    return utils.two_transformations_composition(second_rt, utils.inverse_rt(first_rt))



def get_relative_gt(length_seq):
    globl_rts = np.array(db_utils.read_gt_transformations(paths.GT_PATH)[1:])
    relatives = []
    for i in range(len(globl_rts) - length_seq):
        cur = globl_rts[i + length_seq]
        prev = globl_rts[i]
        relatives.append(utils.two_transformations_composition(prev, utils.inverse_rt(cur)))
    return np.array(relatives)
#
def rel_seq(transfomations_of_movie, length_seq):
    relatives = []
    for i in range(len(transfomations_of_movie) - length_seq):
        first = transfomations_of_movie[i]
        firts_inv = utils.inverse_rt(first)
        second = transfomations_of_movie[i + length_seq]
        relatives.append(utils.two_transformations_composition(firts_inv, second))
    return np.array(relatives)


def relative_dist_between_consequtive_rts(first_rt, second_rt):
    return np.abs(second_rt.T[-1] - first_rt.T[-1])


def cumulative_dist_between_rts(rts):
    rts_rel_dists = []
    for i in range(len(rts) - 1):
        gt_i = rts[i].T[-1]
        gt_in = rts[i + 1].T[-1]
        rel = gt_in - gt_i
        rts_rel_dists.append(np.abs(gt_in - gt_i))
    return np.array(rts_rel_dists).sum(axis=0)


def relative_pnp(transfomations_of_movie, length_seq, output_directory):
    gt_global = np.array(db_utils.read_gt_transformations(paths.GT_PATH)[1:])
    rts_global = utils.from_relative_to_global_transformtions(transfomations_of_movie)[1:]
    rts_rel_dists = []
    gt_rel_dists = []
    for i in range(len(rts_global) - length_seq):
        rts_rel_dists.append(cumulative_dist_between_rts(rts_global[i: i + length_seq]))
        gt_rel_dists.append(cumulative_dist_between_rts(gt_global[i: i + length_seq]))
    rts_rel_dists = np.array(rts_rel_dists)
    gt_rel_dists = np.array(gt_rel_dists)

    xs = rts_rel_dists.T[0]
    ys = rts_rel_dists.T[1]
    zs = rts_rel_dists.T[2]

    xs_relative_gt = gt_rel_dists.T[0]
    ys_relative_gt = gt_rel_dists.T[1]
    zs_relative_gt = gt_rel_dists.T[2]

    xs_error = np.abs(xs - xs_relative_gt)
    ys_error = np.abs(ys - ys_relative_gt)
    zs_error = np.abs(zs - zs_relative_gt)

    xs_percantage_error = 100 * (xs_error / np.abs(xs_relative_gt))
    ys_percantage_error = 100 * (ys_error / np.abs(ys_relative_gt))
    zs_percantage_error = 100 * (zs_error / np.abs(zs_relative_gt))

    diff = np.subtract(rts_rel_dists, gt_rel_dists)
    gt_norm = np.sum(np.abs(gt_rel_dists) ** 2, axis=-1) ** (1. / 2)
    norm = np.sum(np.abs(diff) ** 2, axis=-1) ** (1. / 2)
    norm_percentage = 100 * (norm / gt_norm)


    plt.plot(range(len(norm)), norm_percentage, 'r', label='norm error')
    plt.plot(range(len(norm)), xs_percantage_error, 'g', label='x axis error')
    plt.plot(range(len(norm)), ys_percantage_error, 'b', label='y axis error')
    plt.plot(range(len(norm)), zs_percantage_error, 'y', label='z axis error')
    plt.xlabel('first transformation')
    plt.ylabel('Distance from GT')
    plt.title('PnP Relative location estimation percentage error- {} '.format(length_seq))
    plt.legend()
    plt.savefig('{}/PnP_Relative_location_estimation_error-length={}'.format(output_directory, str(length_seq)))
    plt.close()

def relative_bundle(poses_graph_no_loops, length_seq, output_directory):
    global_gt_rts_ctw = db_utils.read_gt_transformations(paths.GT_PATH)
    global_gt_rts_wtc = [utils.inverse_rt(global_ctw_rt) for global_ctw_rt in global_gt_rts_ctw] # switched to world to cam

    global_poses, global_frames = poses_graph_no_loops.get_all_poses_and_global_frames()
    # get the pairs of frames in global and idxs of them in the keyframes list
    global_frames_pairs, idxs = get_frames_length_closest_to_keyframes(global_frames, length_seq)

    relative_dist_of_length = []
    relative_gt_rts_of_length = []
    for tup_pair in idxs:
    # get relative pose of length
        rel_dist = cummulative_dist_between_poses(global_poses, tup_pair)
        relative_dist_of_length.append(rel_dist)
    relative_dist_of_length = np.array(relative_dist_of_length)
    gt_dists_cummulative = []
    # get relative gt rt of length
    for tup in global_frames_pairs:
        gt_dist = cumulative_dist_between_rts(global_gt_rts_wtc[tup[0]:tup[1]])
        gt_dists_cummulative.append(gt_dist)
    gt_dists_cummulative = np.array(gt_dists_cummulative)



    xs = relative_dist_of_length.T[0]
    ys = relative_dist_of_length.T[1]
    zs = relative_dist_of_length.T[2]

    xs_relative_gt = gt_dists_cummulative.T[0]
    ys_relative_gt = gt_dists_cummulative.T[1]
    zs_relative_gt = gt_dists_cummulative.T[2]

    xs_error = np.abs(xs - xs_relative_gt)
    ys_error = np.abs(ys - ys_relative_gt)
    zs_error = np.abs(zs - zs_relative_gt)

    # todo check if that is the meaning of percantage errors
    xs_percantage_error = 100 * (xs_error / np.abs(xs_relative_gt))
    ys_percantage_error = 100 * (ys_error / np.abs(ys_relative_gt))
    zs_percantage_error = 100 * (zs_error / np.abs(zs_relative_gt))

    diff = np.abs(relative_dist_of_length - gt_dists_cummulative)
    gt_norm = np.sum(np.abs(gt_dists_cummulative) ** 2, axis=-1) ** (1. / 2)
    norm = np.sum(np.abs(diff) ** 2, axis=-1) ** (1. / 2)
    norm_percentage = 100 * (norm / gt_norm)


    plt.plot(range(len(norm)), norm_percentage, 'r', label='norm error')
    plt.plot(range(len(norm)), xs_percantage_error, 'g', label='x axis error')
    plt.plot(range(len(norm)), ys_percantage_error, 'b', label='y axis error')
    plt.plot(range(len(norm)), zs_percantage_error, 'y', label='z axis error')
    plt.xlabel('first transformation')
    plt.ylabel('Distance from GT in %')
    plt.title('Bundle Relative location estimation percentage error- {} '.format(length_seq))
    plt.legend()
    plt.savefig('{}/Bundle Relative location estimation error- length = {} '.format(output_directory, str(length_seq)))
    plt.close()


def cummulative_dist_between_poses(poses, idxs):
    sequential_dists = []
    for i in range(idxs[0], idxs[1]):
        loc1 = poses[i].translation()
        loc2 = poses[i+1].translation()
        sequential_dists.append(np.abs(loc2-loc1))
    return np.array(sequential_dists).sum(axis=0)


def relative_angle_pnp(transformations_of_movie, length_seq, output_directory):
    rts_global = utils.from_relative_to_global_transformtions(transformations_of_movie)
    gt_rts_global = db_utils.read_gt_transformations(paths.GT_PATH)
    rts_rel_angle_dists = []
    gt_rel_angle_dists = []
    for i in range(len(gt_rts_global) - length_seq):
        rts_rel_angle_dists.append(angle_relative_cumulative_pnp(rts_global[i: i + length_seq]))
        gt_rel_angle_dists.append(angle_relative_cumulative_pnp(gt_rts_global[i: i + length_seq]))
    rts_rel_angle_dists = np.array(rts_rel_angle_dists)
    gt_rel_angle_dists = np.array(gt_rel_angle_dists)
    angle_errors = np.abs(rts_rel_angle_dists - gt_rel_angle_dists)
    plt.plot(range(len(angle_errors)), angle_errors)
    plt.xlabel('Frame')
    plt.ylabel('Relative angle estimation error')
    plt.title('Relative angle estimation error - {}'.format(length_seq))
    plt.savefig('{}/Relative_angle_estimation_error_PnP-length={} '.format(output_directory, str(length_seq)))



def angle_relative_cumulative_pnp(rts):
    relative_angles = []
    for i in range(len(rts) - 1):
        rot1 = rts[i].T[:3].T
        rot2 = rts[i+1].T[:3].T
        relative_angles.append(get_np_angle_error(rot1, rot2))
    return np.array(relative_angles).sum()

def angle_relative_cumulative_bundle(poses):
    relative_angles = []
    for i in range(len(poses) - 1):
        rot1 = poses[i].rotation()
        rot2 = poses[i+1].rotation()
        relative_angles.append(get_gtsam_angle_error(rot1, rot2))
    return np.array(relative_angles).sum()



def relative_angle_bundle(poses_graph_no_loops, length_seq, output_directory):
    global_gt_rts_ctw = db_utils.read_gt_transformations(paths.GT_PATH)
    global_gt_rts_wtc = [utils.inverse_rt(global_ctw_rt) for global_ctw_rt in global_gt_rts_ctw] # switched to world to cam
    gtsam_gt = utils.from_global_trans_to_gtsam_pose(global_gt_rts_wtc)
    global_poses, global_frames = poses_graph_no_loops.get_all_poses_and_global_frames()
    # get the pairs of frames in global and idxs of them in the keyframes list
    global_frames_pairs, idxs = get_frames_length_closest_to_keyframes(global_frames, length_seq)

    relative_angle_of_length = []
    for tup_pair in idxs:
    # get relative pose of length
        rel_angle = angle_relative_cumulative_bundle(global_poses[tup_pair[0]:tup_pair[1]+1])
        relative_angle_of_length.append(rel_angle)
    relative_angle_of_length = np.array(relative_angle_of_length)
    gt_angles_cummulative = []
    # get relative gt rt of length
    for tup in global_frames_pairs:
        gt_angle = angle_relative_cumulative_bundle(gtsam_gt[tup[0]:tup[1] + 1])
        gt_angles_cummulative.append(gt_angle)
    gt_angles_cummulative = np.array(gt_angles_cummulative)
    angle_errors = np.abs(relative_angle_of_length - gt_angles_cummulative)
    plt.plot(range(len(angle_errors)), angle_errors)
    plt.xlabel('Frame')
    plt.ylabel('Relative angle estimation error')
    plt.title('Relative angle estimation error Bundle- {}'.format(length_seq))
    plt.savefig('{}/Relative_angle_estimation_error_Bundle-length={} '.format(output_directory, str(length_seq)))



