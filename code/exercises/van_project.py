from VAN_project.code.utils import paths
from VAN_project.code.exercises import ex4_analysis
from VAN_project.code.utils import plot_utils
from VAN_project.code.utils import matching_utils
from VAN_project.code.utils import db_utils
from VAN_project.code.utils import rts_utils
import numpy as np
from VAN_project.code.utils.utils import from_gtsam_relative_to_global, read_data_from_pickle, read_cameras, two_transformations_composition, \
    from_relative_to_global_transformtions, from_global_trans_to_gtsam_pose, write_data
import cv2
project_directory = 'project_plots'

if __name__ == '__main__':
    # bundles = read_data_from_pickle(paths.bundle_graphs_output)
    #
    # # # The RELATIVE transformations computed by the Pnp
    transformations_pnp = np.array(read_data_from_pickle(paths.transformations_of_movie_path))
    # # The GLOBAL transformations computed by the Pnp
    # transformations_pnp_global = from_relative_to_global_transformtions(transformations_pnp)
    #
    # # The global locations of the cameras - output of Pnp
    # locations_pnp = rts_utils.get_positions_of_camera(transformations_pnp[1:])
    #
    # # The global rotation matrices of the cameras - output of Pnp
    # rotations_pnp = transformations_pnp_global[:, :, :-1]
    # #
    # #
    pose_graph = read_data_from_pickle(paths.pose_graph)
    # pose_graph_loops = read_data_from_pickle(paths.pose_graph_with_circles)
    # all_tracks = read_data_from_pickle(paths.all_tracks_path)
    # all_frames = read_data_from_pickle(paths.all_frames_path)
    # # inliers_dict = read_data_from_pickle(paths.inliers_path)
    # # matches_per_frame_dict = read_data_from_pickle(paths.num_matches_per_frame_path)
    # # providing statistics
    # print("Total number of tracks is: ", len(all_tracks))
    # print("Total number of frames is: ", len(all_frames))
    # max_track_len = ex4_analysis.calc_track_stats(all_tracks)
    #
    # ex4_analysis.calc_mean_frame_links(all_tracks, all_frames)
    #
    # #  percentage of inliers per frame graph
    # # ex4_analysis.calc_inliers_percentage(len(all_frames.keys()), inliers_dict, project_directory)
    #
    # # Connectivity graph
    # ex4_analysis.create_connectivity_graph(all_frames, project_directory)
    # # Track length histogram todo check log scale
    # ex4_analysis.plot_tracks_length_hist(max_track_len, all_tracks, project_directory)
    #
    # # Matches per frame
    # # plot_utils.plot_matches_per_frames(matches_per_frame_dict, project_directory)
    #
    #
    # #  Median (or any other meaningful statistic) projection error of the different track links as a function of
    # #  distance from the reference frame (1st frame for Bundle, triangulation frame for PnP)
    # plot_utils.pnp_median_projection_error_per_distance_from_reference(transformations_pnp, all_frames, all_tracks, project_directory)
    #
    # plot_utils.bundles_median_projection_error_per_distance_from_reference(bundles, all_frames,
    #                                                                all_tracks, 'Bundle estimation', project_directory)
    #
    # # Absolute PnP estimation location error
    # plot_utils.plot_pnp_absolute_location_error(locations_pnp, project_directory)
    #
    # # Absolute PnP estimation angle error
    # plot_utils.plot_absolute_pnp_angle_estimation_error(rotations_pnp, project_directory)
    #
    # # Absolute Pose Graph (without loop closure) estimation error - location - Looks-good
    # plot_utils.plot_posegraph_absolute_location_error(pose_graph, " without loops", project_directory)
    #
    # # Absolute Pose Graph (without loop closure) estimation error - angle - Fixed!!!!!!
    # plot_utils.plot_absolute_pose_graph_angle_estimation_error(pose_graph, ' without loop closure', project_directory)
    #
    # # Absolute Pose Graph (with loop closure) estimation error - location - Fixed!!!!
    # plot_utils.plot_posegraph_absolute_location_error(pose_graph_loops, ' with loop closure', project_directory)
    #
    # # Absolute Pose Graph (with loop closure) estimation error - angle
    # plot_utils.plot_absolute_pose_graph_angle_estimation_error(pose_graph_loops, ' with loop closure', project_directory)
    # # todo Relative PnP estimation error graph
    # plot_utils.relative_pnp(transformations_pnp, 100, project_directory)
    # plot_utils.relative_pnp(transformations_pnp, 300, project_directory)
    # plot_utils.relative_pnp(transformations_pnp, 800, project_directory)
    # todo Relative Bundle estimation error
    # plot_utils.relative_bundle(pose_graph, 100, project_directory)
    # plot_utils.plot_relative_bundle_estimation_location_error(pose_graph, 100, project_directory)
    # plot_utils.plot_relative_bundle_estimation_location_error(pose_graph, 300, project_directory)
    # plot_utils.plot_relative_bundle_estimation_location_error(pose_graph, 800, project_directory)


    # Relative PnP estimation angle error
    # plot_utils.relative_angle_pnp(transformations_pnp, 100, project_directory)
    # plot_utils.relative_angle_pnp(transformations_pnp, 300, project_directory)
    # plot_utils.relative_angle_pnp(transformations_pnp, 800, project_directory)

    plot_utils.relative_angle_bundle(pose_graph, 100, project_directory)

    # Read the meta data of the circles in the pose graph
    # circles_meta = read_data_from_pickle(paths.circles_meta_data)
    #
    # # Number of matches per successful loop closure frame, and the inliers - output as text file
    # plot_utils.print_matches_inliers_per_loop_closure(circles_meta, paths.matches_inliers_loop_closures_text)
    #
    # # Uncertainty size vs keyframe – pose graph without loop closure - Location Uncertainty
    # plot_utils.plot_location_uncertainty_size_of_pose_graph_no_loops(pose_graph, project_directory)
    #
    # # Uncertainty size vs keyframe – pose graph with loop closure - Location Uncertainty
    # plot_utils.plot_location_uncertainty_size_of_pose_graph_with_loops(pose_graph_loops, project_directory)
