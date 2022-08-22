import matplotlib.pyplot as plt
import numpy as np
import gtsam
from gtsam import symbol
from VAN_project.code.utils import utils
from gtsam.utils.plot import plot_trajectory
from gtsam.utils.plot import plot_3d_points
from VAN_project.code.utils.db_utils import read_gt_transformations
from VAN_project.code.GraphsOptimization.BundleDir.BundleGraph import stereo_to_gs_stereo
from VAN_project.code.GraphsOptimization.BundleDir.BundlesManager import BundlesManager, RADIANS_NOISE
from VAN_project.code.utils import paths
from VAN_project.code.utils.plot_utils import COLORS_LIST, plot_relative_position_of_cameras

x_for_camera_symbol = 'x'
l_for_camera_symbol = 'l'
PXL_AROUND_FEATURE = 100


def plot_top_view(data, output_path, title, x_axis="X axis", y_axis="Z axis"):
    """
    :param data: a list of lists of points to plot
    """
    # plot the cameras and points
    for i, points_list in enumerate(data):
        plt.scatter(points_list[0], points_list[1], color=COLORS_LIST[i])
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.savefig("{}.png".format(output_path))
    plt.close()


def q1():
    # get the cameras matrices
    k, m1, m2 = utils.read_cameras()
    #  create camera's K
    K = gtsam.Cal3_S2Stereo(k[0, 0], k[1, 1], k[0, 1], k[0, 2], k[1, 2], -m2[0, 3])


    # read all tracks db
    all_tracks = utils.read_data_from_pickle(paths.all_tracks_path)

    # sample track longer than 10
    track = utils.sample_track_longer_than_n(10, all_tracks, rand=False)

    # read the relative transformations created in the previous excercises
    relative_transformations = utils.read_data_from_pickle(paths.transformations_of_movie_path)

    # take only the transformation of the track and move them to be "global" compared to the first frame of the track
    trans_of_track = relative_transformations[track.first_frame: track.last_frame]
    trans_of_track_in_global = utils.from_relative_to_global_transformtions(trans_of_track)

    # transform to gtsam Rt's
    poses_of_track = utils.from_global_trans_to_gtsam_pose(trans_of_track_in_global)

    # create gs_frames
    gs_frames = [gtsam.gtsam.StereoCamera(x, K) for x in poses_of_track]

    # get the coordinates in each image of the track
    coords = track.coordinates
    gs_coords = [stereo_to_gs_stereo(x) for x in coords]

    # get last frame in track.
    gs_frame = gs_frames[-1]
    # create stereo point
    sp2 = gs_coords[-1]
    # get 3d point in global (of the first frame in the track - corresponding to the stereo point
    p3 = gs_frame.backproject(sp2)


    # project the 3d point on all cameras in the track
    gs_projections = []
    for cur_gs_frame in gs_frames:
        gs_projections.append(cur_gs_frame.project(p3))

    # re-arange the projections as left/right projections arrays
    gs_left_coors = np.array([(gs_projections[i].uL(), gs_projections[i].v()) for i in range(len(gs_projections))])
    gs_right_coors = np.array([(gs_projections[i].uR(), gs_projections[i].v()) for i in range(len(gs_projections))])

    # re-arange our coordinates-detection as left/right coordinates arrays
    left_coors = np.array([(gs_coords[i].uL(), gs_coords[i].v()) for i in range(len(gs_coords))])
    right_coors = np.array([(gs_coords[i].uR(), gs_coords[i].v()) for i in range(len(gs_coords))])

    # calc projection dist
    left_proj_dist = np.linalg.norm(gs_left_coors - left_coors, axis=1)
    right_proj_dist = np.linalg.norm(gs_right_coors - right_coors, axis=1)
    dists = (left_proj_dist + right_proj_dist) / 2


    # plot the projection distance over time
    plt.plot(range(track.first_frame, track.last_frame + 1), dists)
    plt.xlabel('frame id')
    plt.ylabel('gtsam projection distance')
    plt.title('gtsam projection distances')
    plt.savefig('{}/gtsam_projection_distances.png'.format(paths.plots_output))
    plt.close()


    # Question 1 section 2

    # create keys for variables of track
    xs = []
    for i, frame_idx in enumerate(range(track.first_frame, track.last_frame+1)):
        xs.append(symbol('x', i))
        print("i {} ".format(i))
    l = symbol('l', 0)


    # Create graph container and add factors to it
    graph = gtsam.NonlinearFactorGraph()


    ## add a constraint on the starting pose
    first_pose = gtsam.Pose3()
    noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([RADIANS_NOISE, RADIANS_NOISE, RADIANS_NOISE, 0.1, 0.1, 0.1]))
    first_factor = gtsam.PriorFactorPose3(xs[0], first_pose, noise)

    # add measurements
    factors = []
    for i, coord in enumerate(gs_coords):
        noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, 1.0]))
        factor = gtsam.GenericStereoFactor3D(coord, noise_model, xs[i], l, K)
        factors.append(factor)
        graph.add(factor)


    initialEstimate = gtsam.Values()
    # initialEstimate.insert(l, p3)
    for i, x in enumerate(xs):
        initialEstimate.insert(x, poses_of_track[i])
    initialEstimate.insert(l, p3)


    # plot the projection distance over time
    error = [factor.error(initialEstimate) for factor in factors]
    plt.plot(range(track.first_frame, track.last_frame + 1), error)
    plt.xlabel('frame id')
    plt.ylabel('Factor Error')
    plt.title('Factor error over the frames of the track')
    plt.savefig('{}/factor_error'.format(paths.plots_output))
    plt.close()


def q2():

    bundles_manager = BundlesManager()
    bundles_manager.process_drive()


    first_bundle = bundles_manager.get_bundle(0)
    initialEstimate = first_bundle.initialEstimate
    result = first_bundle.result
    graph = first_bundle.graph

    bundles_manager.save(paths.bundle_graphs_output)
    # path = paths.bundle_graphs_output
    # with open(path, 'wb') as f:
    #     pickle.dump(bundles_manager, f)
    plot_trajectory(fignum=1, values=initialEstimate, title="Trajectory- initialEstimate")
    plot_3d_points(fignum=2, values=initialEstimate, title="Points - initial estimate")

    # gtsam.utils.plot.plot_pose3_on_axes()

    plot_trajectory(fignum=3, values=result, title="Trajectory - after optimization")
    plot_3d_points(fignum=4, values=result, title=" Points - after optimization")
    print("Error init {} final error {}".format(graph.error(initialEstimate), graph.error(result)))

    plt.show()


def q2_q3_plots():
    # Question 2 plots

    # read the optimized data
    bundle_manager = utils.read_data_from_pickle(paths.bundle_graphs_output)
    # all_cameras_poses_by_bundles = []
    # all_3d_points_by_bundle = []


    # first we plot the section 2 plot - top view of the cameras pos and 3d points
    first_bundle = bundle_manager.get_bundle(0)
    cameras_first_bundle = first_bundle.get_poses()
    points_first_bundle = first_bundle.get_points()

    # take only x and z for a top-view-plot
    cameras_location_first_bundle = np.array(
        [(camera_pose.x(), camera_pose.z()) for camera_pose in cameras_first_bundle])
    points_location_first_bundle = np.array([(point[0], point[2]) for point in points_first_bundle])

    # Section 2 - plot the cameras and points of the first bundle
    title = "Top view first bundle"
    output_path = "top_view_first_bundle"
    cameras_2d_locations = (cameras_location_first_bundle[:, :1], cameras_location_first_bundle[:, -1:])
    points_2d = (points_location_first_bundle[:, :1], points_location_first_bundle[:, -1:])
    plot_top_view([cameras_2d_locations, points_2d], title, output_path)


    #### section 3 ###

    # read the corresponding GT positions for the keyframes
    gt_transformations = read_gt_transformations(paths.GT_PATH)
    gt_locations_in_l0_coords = [-1 * Rt[:, :3].T @ Rt[:, 3:] for Rt in gt_transformations]

    # gt locations of key frames is 3D point in the coordinates of the first left camera - describing the real
    # locations of the left camera at each of the keyframes.
    gt_locations_of_key_frames = []
    for key_frames in bundle_manager.get_keyframes_list():
        gt_locations_of_key_frames.append(gt_locations_in_l0_coords[key_frames[0]])
    gt_locations_of_key_frames = np.array(gt_locations_of_key_frames)

    # create the optimised locations of the key-frames by composing the Rt's
    rts_of_optimised_key_frames = []
    first_post = gtsam.Pose3()
    # push the first camera pose
    rts_of_optimised_key_frames.append(first_post)

    for cameras_bundle_idx in range(bundle_manager.get_num_bundles()):
        cur_bundle = bundle_manager.get_bundle(cameras_bundle_idx)
        last_pos_cur_bundle = cur_bundle.get_pos_at_idx(-1)
        rts_of_optimised_key_frames.append(last_pos_cur_bundle.inverse())

    all_optimized_in_global = [rts_of_optimised_key_frames[0]]
    for rt in rts_of_optimised_key_frames[1:]:
        cur_rt = rt.compose(all_optimized_in_global[-1].inverse())
        all_optimized_in_global.append(cur_rt.inverse())

    positions_all_optimized_in_global = np.array([(rt.x(), rt.y(), rt.z()) for rt in all_optimized_in_global])

    # collect the 3d points as well
    all_3d_points = []
    # for i, cloud in enumerate(all_3d_points_by_bundle):
    for bundle_idx in range(bundle_manager.get_num_bundles()):
        cloud = bundle_manager.get_bundle(bundle_idx).get_points()
        for point in cloud:
            # project this point to global. It is received in the coordinates of the first keyframe of the bundle
            point_in_global = all_optimized_in_global[bundle_idx].transformFrom(point)
            all_3d_points.append(point_in_global)
    all_3d_points = np.array(all_3d_points)

    tracks_title = "Top view key-frames location - GT(blue) vs. Optimized(orange)"
    track_fig_output = "{}/{}.png".format(paths.plots_output, "top_view_key_frames_vs_gt")
    plot_relative_position_of_cameras([gt_locations_of_key_frames, positions_all_optimized_in_global],
                                          title=tracks_title,
                                          output_fig_name=track_fig_output)

    tracks_title = "Top view key-frames location - including 3D points"
    track_fig_output = "{}/{}.png".format(paths.plots_output, "top_view_key_frames_including_3d_points")
    plot_relative_position_of_cameras([gt_locations_of_key_frames, positions_all_optimized_in_global,
                                           all_3d_points],
                                          title=tracks_title,
                                          output_fig_name=track_fig_output)

    # last section - compute and plot the keyframe localization error
    pos_for_diff = positions_all_optimized_in_global[:-1, :]
    pos_for_diff = pos_for_diff[:-5:]
    gt_for_diff = gt_locations_of_key_frames.reshape(
        (gt_locations_of_key_frames.shape[0], gt_locations_of_key_frames.shape[1]))
    gt_for_diff = gt_for_diff[:-5, :]
    # plot the projection distance over time
    num_of_key_frames = gt_for_diff.shape[0]
    diff = np.subtract(pos_for_diff, gt_for_diff)
    norm = np.sum(np.abs(diff) ** 2, axis=-1) ** (1. / 2)
    plt.plot(range(num_of_key_frames), norm)
    plt.xlabel('Keyframes')
    plt.ylabel('Distance')
    plt.title('Distance between our poses and gt over the keyframes')
    plt.savefig('{}/dist_pos_keyframes.png'.format(paths.plots_output))
    plt.close()


if __name__ == '__main__':
    # question 1
    q1()

    # Question 2
    q2()

    q2_q3_plots()

