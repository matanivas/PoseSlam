import gtsam
import numpy as np
from VAN_project.code.utils.utils import from_relative_to_global_transformtions_gtsam, write_data, read_cameras, from_relative_to_global_transformtions, from_global_trans_to_gtsam_pose, from_gtsam_relative_to_global
from VAN_project.code.GraphsOptimization.BundleDir.BundleGraph import POSE_NOISE
from VAN_project.code.GraphsOptimization.BundleDir.BundleGraph import BundleGraph
from igraph import *
from VAN_project.code.utils.matching_utils import *
from VAN_project.code.utils import paths
from VAN_project.code.FramesDir.Drive import Drive
import copy
# https://igraph.org/python/tutorial/latest/tutorial.html#starting-igraph
MINIMUM_MATCHES_FOR_PNP = 150
LOOP_THRESHOLD = 50
CONSENSUS_MATCHING_THRESHOLD = 40
MAX_NUM_PATHS_TO_CHECK_PER_POS = 3


def loop_closure_consensus_matching(first_frame_idx, second_frame_idx):
    """
    :param first_frame_idx: the index of the first image to match
    :param second_frame_idx: the index of the second image to match
    :return: Drive processed of the two frames
    """
    cur_drive = Drive((first_frame_idx, second_frame_idx), True)
    cur_drive.run()
    return cur_drive


def relative_pose_estimation(cur_drive, K, m1, m2):
    """
    :param cur_drive: the Drive composing of the two to analyse
    :return: relative covariance and transformation between the frames
    """
    all_frames = cur_drive.get_frames_db()
    all_tracks = cur_drive.get_tracks_db()
    relative_transformation = cur_drive.get_transformations()[0]

    trans_of_bundle = from_relative_to_global_transformtions([relative_transformation])
    poses_of_bundle = from_global_trans_to_gtsam_pose(trans_of_bundle)

    small_bundle = BundleGraph((0, 1), poses_of_bundle, K, m1, m2, False)
    small_bundle.optimize(all_frames, all_tracks)

    keyframes_key = gtsam.KeyVector()
    keyframes_key.append(small_bundle.camera_nodes[0])
    keyframes_key.append(small_bundle.camera_nodes[1])
    marginal = gtsam.Marginals(small_bundle.graph, small_bundle.result)
    rel_cov = marginal.jointMarginalCovariance(keyframes_key).at(keyframes_key[-1], keyframes_key[-1])
    return rel_cov, small_bundle.get_pos_at_idx(1)


def keyframes_pose_cov(bundle, marginal):
    """
    :param bundle: the bundle to analyse
    :param marginal: the bundle's marginal
    :return: the covariance and relative pose betweeen the key-frames (first and last in the bundle)
    """
    first_cam_symbol = bundle.camera_nodes[0]
    last_cam_symbol = bundle.camera_nodes[-1]
    #get keyframes covariance
    keyframes_key = gtsam.KeyVector()
    keyframes_key.append(first_cam_symbol)
    keyframes_key.append(last_cam_symbol)
    keyframes_information = marginal.jointMarginalInformation(keyframes_key).at(keyframes_key[-1], keyframes_key[-1])
    keyframes_cov = np.linalg.inv(keyframes_information)

    #get relative pose3
    first_gt_pose = bundle.result.atPose3(first_cam_symbol)
    last_gt_pose = bundle.result.atPose3(last_cam_symbol)
    relative_pose = first_gt_pose.between(last_gt_pose)
    return keyframes_cov, relative_pose


def t2v(pose):
    """
    :param rvecs: pose
    :return: oiler trandformation vector of angles and coordinates
    """
    rot = pose.rotation()
    azim = rot.yaw()
    pitch = rot.pitch()
    roll = rot.roll()
    x = pose.x()
    y = pose.y()
    z = pose.z()
    return np.array([azim, pitch, roll, x, y, z])


def closeness_indication(cov, pose):
    delta_c_vec = t2v(pose)
    return delta_c_vec.T @ cov @ delta_c_vec


class PoseGraph:

    def __init__(self, bundles_manager):
        """
        Given the bundles-graphs data - create a pose graph based on the keyframes
        """

        # The graph
        self.graph = gtsam.NonlinearFactorGraph()
        # A list of the symbol of the poses
        self.poses_symbols = []

        # A list of the global frames in the movie that correspond to the poses
        self.global_frames = []

        # The factors in the graph
        self.factors = []

        # The initial estimate of the graph
        self.initialEstimate = gtsam.Values()

        # The result of the optimization
        self.result = None

        # The state of the graph (before or after optimization)
        self.optimized = False

        # dict for relative covs between poses
        self.relative_covs = dict()

        # The marginals of the graph
        self.marginal = None

        # for stats - save data relevant to loop closures - dictionary - key - frame(i)-frame(j)
        # and value is another dictionary with two keys: number of matches in the loop closure.
        # and the inliers percentage.
        self.loop_closures_meta = {}

        # create neighborness graph (for finding shortest paths.
        self.neighborness_graph = Graph()
        self.symbols_neighborness_nodes_mapping = {}
        self.neighborness_nodes_symbols_mapping = {}

        # create the first pose of the graph - global zero.
        self.add_symbol(bundles_manager.get_bundle(0).first_frame_global)
        first_pose = gtsam.Pose3()
        first_factor = gtsam.PriorFactorPose3(self.poses_symbols[-1], first_pose, POSE_NOISE)

        self.graph.add(first_factor)
        self.factors.append(first_factor)


        for i in range(bundles_manager.get_num_bundles()):
            bundle = bundles_manager.get_bundle(i)
            cur_marginal = gtsam.Marginals(bundle.graph, bundle.result)
            keyframes_conditioned_cov, relative_pose = keyframes_pose_cov(bundle, cur_marginal)

            # key_frames_symbol = gtsam.symbol('p', i + 1)
            # self.poses_symbols.append(key_frames_symbol)
            self.add_symbol(bundle.last_frame_global)
            gt_cov = gtsam.noiseModel.Gaussian.Covariance(keyframes_conditioned_cov)

            cur_factor = gtsam.BetweenFactorPose3(self.poses_symbols[i], self.poses_symbols[i + 1],
                                                  relative_pose, gt_cov)
            self.add_factor_between_poses(cur_factor, self.poses_symbols[i], self.poses_symbols[i+1])

        gtsam_rel_world_to_cam = [bundles_manager.get_bundle(0).get_pos_at_idx(0).inverse()]
        for i in range(bundles_manager.get_num_bundles()):
            bundle = bundles_manager.get_bundle(i)
            last_pos_in_bundle = bundle.result.atPose3(bundle.camera_nodes[-1]).inverse()
            first_pos_in_bundle = bundle.result.atPose3(bundle.camera_nodes[0]).inverse()
            last_in_first_coords = last_pos_in_bundle.compose(first_pos_in_bundle)
            gtsam_rel_world_to_cam.append(last_in_first_coords)

        # transfer the relative  "world to cam" to global  "world to cam"
        gtsam_global_world_to_cam = from_relative_to_global_transformtions_gtsam(gtsam_rel_world_to_cam)

        # for i, x in enumerate(key_frames_poses_symbols):
        for i, x in enumerate(self.poses_symbols):
            self.initialEstimate.insert(x, gtsam_global_world_to_cam[i].inverse())

    def optimize(self):
        """
        optimize the graph
        """
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initialEstimate)
        self.result = optimizer.optimize()
        self.optimized = True
        # Calc the new marginals of the graph
        self.marginal = gtsam.Marginals(self.graph, self.result)

    def close_loops(self):
        self.marginal = gtsam.Marginals(self.graph, self.result)

        # get camera's intrinsic matrices for the relative pose estimation
        K, m1, m2 = read_cameras()
        frames_locations_for_pose_graph_version = []
        all_closeness_scores = {}
        num_circles = 0
        for i, pose in enumerate(self.poses_symbols):  # loop through all poses
            # create a list of the best paths
            all_paths_of_possible_circle = []
            for j, prev_pos in enumerate(self.poses_symbols[:i]):  # for a given pos, loop through all poses up to it.
                # get the shortest path between the two poses
                shortest_path = self.get_shortest_path(pose, prev_pos)
                # get the distance score between the two poses
                if len(shortest_path) < 30:
                    continue
                closeness_score = self._get_closeness(shortest_path)
                if closeness_score <= LOOP_THRESHOLD:
                    all_paths_of_possible_circle.append((closeness_score, shortest_path))

            all_closeness_scores[str(i)] = all_paths_of_possible_circle
            # find the best possible circles
            sorted_paths = sorted(all_paths_of_possible_circle,
                                  key=lambda tup: tup[0])
            best_paths = sorted_paths[:min(MAX_NUM_PATHS_TO_CHECK_PER_POS, len(sorted_paths))]

            plot_consensus_match = False
            # Perform concensus matching over the best possible circles
            print("Analysing keyframe {} with num possible paths: {}".format(i, len(best_paths)))
            for candidate_path in best_paths:
                candidate_path = candidate_path[1]  # get the candidate path itself
                first_frame_idx = self.get_global_frame(candidate_path[0])
                second_frame_idx = self.get_global_frame(candidate_path[-1])

                # Process the frames to find matches
                cur_drive = loop_closure_consensus_matching(first_frame_idx, second_frame_idx)
                # Check if the match was successful
                if not cur_drive.get_success_for_loop_closure():
                    continue

                # if we're here - we found a circle - and want to add it to the graph and optimize
                # get the bundles output - covariance and relative-position
                rel_cov, rel_pos = relative_pose_estimation(cur_drive, K, m1, m2)

                print("***** Found circle between frame: {} and frame: {}".format(
                    self.get_global_frame(candidate_path[-1]),
                    self.get_global_frame(candidate_path[0])))
                num_circles += 1
                # add the new factor to the graph
                gt_cov = gtsam.noiseModel.Gaussian.Covariance(rel_cov)
                cur_factor = gtsam.BetweenFactorPose3(candidate_path[0], candidate_path[-1],
                                                      rel_pos, gt_cov)
                self.add_factor_between_poses(cur_factor, candidate_path[0], candidate_path[-1])

                # optimize the graph given the new factor
                self.optimize()
                frames_locations_for_pose_graph_version.append(first_frame_idx)
                # Update data of loop closure for stats
                self.loop_closures_meta[str(first_frame_idx) + '-' + str(second_frame_idx)] = {
                    "num_matches": cur_drive.get_matches_per_frame_dict(),
                    "inliers_dict": cur_drive.get_inliers_per_frame_dict()
                }
                # if first_frame_idx in frames_locations_for_pose_graph_version:
                #     plot_utils.plot_pose_graph(1, pose_graph.result, first_frame_idx, pose_graph.marginal)
        # write the meta data of the circles found
        utils.write_data(paths.circles_meta_data, self.loop_closures_meta)
        print("Total num of circles {}".format(num_circles))

    def add_factor_between_poses(self, factor, first_pos_symbol, second_pos_symbol):
        """
        Add factor between poses to the graph, and add the connection to the neighborness_graph
        :param factor: the factor between poses to add to the graph
        :param first_pos_symbol: the symbol of the first pose in the factor
        :param second_pos_symbol: the symbol of the second pose in the factor
        """
        self.optimized = False
        # add factor to graph
        self.graph.add(factor)
        # save factor for debugging usage
        self.factors.append(factor)
        # add factor as an edge to the neighborness graph
        self.neighborness_graph.add_edge(self.symbols_neighborness_nodes_mapping[first_pos_symbol],
                                         self.symbols_neighborness_nodes_mapping[second_pos_symbol])

    def add_symbol(self, global_frame_idx):
        """
        add a camera symbol to the graph
        and add it to the neighborness_graph as a node.
        :param symbol: the symbol to add
        """
        num_of_symbols = len(self.poses_symbols)
        # create the symbol
        key_frame_symbol = gtsam.symbol('p', num_of_symbols)
        # add it to list of symbols
        self.poses_symbols.append(key_frame_symbol)
        # add the global frame that matches the symbol
        self.global_frames.append(global_frame_idx)
        # add it to neighborness_graph
        self.symbols_neighborness_nodes_mapping[key_frame_symbol] = num_of_symbols
        self.neighborness_nodes_symbols_mapping[num_of_symbols] = key_frame_symbol
        self.neighborness_graph.add_vertex(self.symbols_neighborness_nodes_mapping[key_frame_symbol])

    def get_shortest_path(self, first_symbol, second_symbol):
        """
        Given two symbols - return the shortest path between them - list of symbols.
        """
        first_node_in_graph = self.symbols_neighborness_nodes_mapping[first_symbol]
        second_node_in_graph = self.symbols_neighborness_nodes_mapping[second_symbol]
        shortest_path_in_nodes = self.neighborness_graph.get_shortest_paths(first_node_in_graph, second_node_in_graph)[0]
        shortest_path_in_symbols = [self.neighborness_nodes_symbols_mapping[node] for node in shortest_path_in_nodes]
        return shortest_path_in_symbols

    def get_global_frame(self, symbol):
        """
        Given pos symbol, return the global frame index that matches it.
        This assumes that the global frames were added in correspondence to the camera poses symbols.
        """
        idx_of_pos = self.poses_symbols.index(symbol)
        return self.global_frames[idx_of_pos]

    def get_global_pos_of_symbol(self, symbol):
        """
        given a pose_symbol, return it's pose
        :param symbol:
        :return: pose
        """
        return self.result.atPose3(symbol)

    def get_all_poses_and_global_frames(self):
        """
        return all the poses in the graph
        :return:
        """
        return [self.get_global_pos_of_symbol(i) for i in self.poses_symbols], [self.get_global_frame(i) for i in self.poses_symbols]

    def save(self, output_path):
        self.marginal = None
        write_data(output_path, self)

    def _get_closeness(self, symbols_path):
        """
        :param symbols_path: path to analyse - list of symbols corresponding to poses
        :return: true is the first and last pose in the path are close enough to each other
        """
        relative_cov = self._get_rel_cov(symbols_path)
        delta_c = self._get_delta_c(symbols_path)
        closeness_score = closeness_indication(relative_cov, delta_c)

        return closeness_score

    def _get_rel_cov(self, symbols_path):
        """
        :param symbols_path: path to analyse - list of symbols corresponding to poses
        :return: the sum-of-covariances between the two poses on the path (first and last)
        """
        covs = []

        for i in range(len(symbols_path) - 1):
            prev = symbols_path[i]
            cur = symbols_path[i + 1]

            if self.optimized and str(prev) + str(cur) in self.relative_covs:
                covs.append(self.relative_covs[str(prev) + str(cur)])
            else:
                keyframes_key = gtsam.KeyVector()
                keyframes_key.append(prev)
                keyframes_key.append(cur)
              # cur_cov = pose_graph.marginal.jointMarginalCovariance(keyframes_key).at(keyframes_key[-1], keyframes_key[-1])
                cur_info = self.marginal.jointMarginalInformation(keyframes_key).at(keyframes_key[-1],
                                                                                     +
                                                                                     keyframes_key[-1])
                cur_cov = np.linalg.inv(cur_info)

                assert np.max(cur_cov) < 1
                self.relative_covs[str(prev) + str(cur)] = cur_cov
                self.relative_covs[str(cur) + str(prev)] = cur_cov
                covs.append(cur_cov)

        covs_sum = copy.deepcopy(covs[0])
        for cov in covs[1:]:
                        covs_sum += cov

        return covs_sum

    def _get_delta_c(self, symbols_path):
        """
        :param symbols_path: a list of symbols
        :return: the relative distance between the last pose in the path and the first
        """
        # here we create a list of the relative transformation between the two cameras in CAM TO WORLD
        rel_trans = [gtsam.Pose3()]
        first_gt_pose = self.result.atPose3(symbols_path[0])

        second_gt_pose = self.result.atPose3(symbols_path[-1])
        relative_transformation = first_gt_pose.between(second_gt_pose)
        rel_trans.append(relative_transformation)

        global_gtsam_pos = from_gtsam_relative_to_global(rel_trans)

        last_pos_in_path = global_gtsam_pos[-1]

        return last_pos_in_path

    def get_loop_closures_meta(self):
        return self.loop_closures_meta
