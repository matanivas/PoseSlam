import numpy as np
import gtsam
from VAN_project.code.utils import utils
from VAN_project.code.utils import paths
from VAN_project.code.GraphsOptimization.PoseDir.PoseGraph import PoseGraph, keyframes_pose_cov


ex6_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.00001, 0.00001, 0.00001, 0.00001, 0.0001, 0.0001]))


if __name__ == '__main__':

    bundle_manager = utils.read_data_from_pickle(paths.bundle_graphs_output)
    first_bundle = bundle_manager.get_bundle(0)
    marginal = gtsam.Marginals(first_bundle.graph, first_bundle.result)
    # plot.plot_trajectory(1, first_bundle.result, marginals=marginal, scale=1)

    first_keyframes_conditioned_cov, first_relative_pose = keyframes_pose_cov(first_bundle, marginal)
    print("First keyframes pair relative pose is: {} \n First keyframes pair Covariance matrix is: {}".format
          (first_relative_pose, first_keyframes_conditioned_cov))

    # q 6.2

    # create pose graph using the bundle-graphs
    pose_graph = PoseGraph(bundle_manager)
    # optimize the pose graph
    pose_graph.optimize()
    # Save the pose graph for ex 7
    pose_graph.save(paths.pose_graph)

    graph = pose_graph.graph
    result = pose_graph.result

    pose_graph_marginal = gtsam.Marginals(graph, result)
    fig = "Final - Pose graph top view results"

    # plot.plot_trajectory(1, result, marginals=None, scale=1, title="Rotated 3D Trajectory after optimization- Without covariances", d2_view=True)
    # plt.savefig("Rotated 3D Trajectory after optimization of the poses- Without covariances")
    # plt.close()
    #
    # plot.plot_trajectory(1, result, marginals=None, scale=1, title="Rotated 3D Trajectory of initial poses- Without covariances", d2_view=True)
    # plt.savefig("Rotated 3D Trajectory of initial poses- Without covariances")
    # plt.close()


