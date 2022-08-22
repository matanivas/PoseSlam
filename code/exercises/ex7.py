from VAN_project.code.utils.utils import read_data_from_pickle
from VAN_project.code.utils import plot_utils
from VAN_project.code.utils import paths

MINIMUM_MATCHES_FOR_PNP = 150
LOOP_THRESHOLD = 50
CONSENSUS_MATCHING_THRESHOLD = 40
MAX_NUM_PATHS_TO_CHECK_PER_POS = 3
# gaussian_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, 1.0]))
# cauchy = gtsam.noiseModel.mEstimator.Cauchy.Create(2)
# cauchy_model = gtsam.noiseModel.Robust.Create(cauchy, gaussian_model)

# # plot detections on images
# def my_plot_accepted_and_rejected(frame_1_idx, frame_2_idx, img1_accepted_coors,
#                                img2_accepted_coors):
#     """
#     plot on images the accepted and rejected coords
#     """
#     img1, _ = read_images(frame_1_idx)
#     img2, _ = read_images(frame_2_idx)
#     figure = plt.figure(figsize=(9, 6))
#     figure.add_subplot(2, 1, 1)
#     img1_accepted_coors = np.array(img1_accepted_coors)
#     img2_accepted_coors = np.array(img2_accepted_coors)
#     plt.title('Frame {} orange: inliers, cyan: outliers'.format(frame_1_idx))
#     plt.imshow(img1, cmap='gray')
#     plt.scatter(img1_accepted_coors[:, 0], img1_accepted_coors[:, 1], s=2, color='orange')
#
#     figure.add_subplot(2, 1, 2)
#     plt.title('Frame {} orange: inliers, cyan: outliers'.format(frame_2_idx))
#     plt.imshow(img2, cmap='gray')
#     plt.scatter(img2_accepted_coors[:, 0], img2_accepted_coors[:, 1], s=2, color='orange')
#     plt.show()

if __name__ == '__main__':
    pose_graph = read_data_from_pickle(paths.pose_graph)

    pose_graph.close_loops()

    pose_graph.save(paths.pose_graph_with_circles)
    plot_utils.plot_pose_graph_poses(pose_graph.result, pose_graph.initialEstimate, pose_graph.poses_symbols,
                                     title="Pose graph with circles",
                                     output_fig_name="{}/pose_graph_with_circles.png".format(paths.plots_output))

