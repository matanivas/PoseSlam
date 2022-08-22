import gtsam
from gtsam import symbol
import numpy as np
import copy
RADIANS_NOISE = (np.pi / 180) ** 2
POSE_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([RADIANS_NOISE, RADIANS_NOISE, RADIANS_NOISE, 1.0, 1.0, 1.0]))

x_for_camera_symbol = 'x'
l_for_camera_symbol = 'l'


def stereo_to_gs_stereo(stereo):
    """
    get tuple - (x1, x2, y) and return as gtsam stereopoint2 object
    :param stereo:
    :return:
    """
    return gtsam.StereoPoint2(stereo[0], stereo[1], stereo[2])


def gs_triangulation(stereo_point, cur_gs_frame):
    """
    given tuple (x1, x2, y) and gtsam frame - return 3D point (gtsam object)
    :param stereo_point:
    :param stereo_point:
    :param cur_gs_frame:
    :return:
    """
    # create stereo point
    sp2 = gtsam.StereoPoint2(stereo_point[0], stereo_point[1], stereo_point[2])
    # get 3d point in global - corresponding to the stereo point
    p3 = cur_gs_frame.backproject(sp2)
    return p3


def gtsam_K(k, m1, m2):
    return gtsam.Cal3_S2Stereo(k[0, 0], k[1, 1], k[0, 1], k[0, 2], k[1, 2], -m2[0, 3])


class PointGraph:

    def __init__(self, point_symbol, three_d):
        self._symbol = point_symbol
        self._three_d = three_d

    def get_location(self):
        return self._three_d

    def get_symbol(self):
        return self._symbol


class BundleGraph:

    def __init__(self, key_frames_tuple, poses_through_window, k, m1, m2, filter_tracks_based_on_jump=True):
        self.first_frame_global = key_frames_tuple[0]
        self.last_frame_global = key_frames_tuple[1]
        self.initial_poses = poses_through_window
        self.k = k
        self.m1 = m1
        self.m2 = m2

        # create the objects for the optimization
        gtsam_k = gtsam_K(self.k, self.m1, self.m2)
        self.gs_frames_of_bundle = [gtsam.gtsam.StereoCamera(x, gtsam_k) for x in self.initial_poses]
        self.point_nodes = []
        self.camera_nodes = []
        for i, frame in enumerate(range(self.first_frame_global, self.last_frame_global + 1)):
            self.camera_nodes.append(symbol('x', i))

        # create instance of factor graph
        self.graph = gtsam.NonlinearFactorGraph()

        # create initial estimate place holder
        self.initialEstimate = gtsam.Values()

        # create place holder for the result of the optimization
        self.result = None
        # Create place holder for
        self.gs_frames_of_bundle_after_optimization = None

        # add the constraint to start at (0,0,0)
        first_pose = gtsam.Pose3()
        first_factor = gtsam.PriorFactorPose3(self.camera_nodes[0], first_pose, POSE_NOISE)
        self.graph.add(first_factor)

        # create 3d points dictionary - to triangulate only once per track
        self.tracks_3d_points_of_last_frame = {}

        # create flag for not filtering points based on jump in coordinates
        self.filter_tracks_based_on_jump = filter_tracks_based_on_jump

    def optimize(self, all_frames, all_tracks):
        # get gtsam k (not a member because can't be pickled
        gtsam_k = gtsam_K(self.k, self.m1, self.m2)

        # for each frame - for each track in frame - add to graph all constraints this track imposes (on all frames in it)
        for i, frame in enumerate(range(self.first_frame_global, self.last_frame_global + 1)):
            cur_frame_track_ids = all_frames[str(frame)]
            for track_id in cur_frame_track_ids:
                cur_track = all_tracks[str(track_id)]
                if str(track_id) not in self.tracks_3d_points_of_last_frame:
                    if not cur_track.is_good_track() and self.filter_tracks_based_on_jump:
                        # if the track is going to add noise to the bundle - don't add it
                        continue

                    # this is the first time we see this track - find it's last_frame, triangulate
                    # get max frame (length of track, bounded by the window's length) - relative. NOT GLOBAL INDEX
                    cur_p3 = self.triangulate_track_point_in_last_relevant_frame(cur_track)

                    if cur_p3[2] > 100 or cur_p3[2] < 1:  # don't add points that are very far
                        continue

                    cur_l = symbol('l', len(self.point_nodes))
                    self.point_nodes.append(PointGraph(cur_l, cur_p3))
                    # add this point to our tracks-to-3d-points-dict
                    self.tracks_3d_points_of_last_frame[str(track_id)] = (cur_l, cur_p3)

                # here we know we have the 3D point in hand - so we project it and add the constraint to the graph
                cur_coords = cur_track.get_coords_in_frame(frame)  # cur coords expects Sglobal frame id

                noise_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, 1.0]))
                cur_factor = gtsam.GenericStereoFactor3D(stereo_to_gs_stereo(cur_coords), noise_model,
                                                         self.camera_nodes[i],
                                                         self.tracks_3d_points_of_last_frame[str(track_id)][0], gtsam_k)
                # todo clean the unnecessary data bases and objects
                # self.factors.append((cur_factor, self.tracks_3d_points_of_last_frame[str(track_id)][0], cur_track))
                self.graph.add(cur_factor)

        for i, x in enumerate(self.camera_nodes):
            self.initialEstimate.insert(x, self.initial_poses[i])
        for point in self.point_nodes:
            self.initialEstimate.insert(point.get_symbol(), point.get_location())
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initialEstimate)
        self.result = optimizer.optimize()
        # print("total number of factors in graph 2: {} ERROR INIT 2 {} ERROR RESULT 2 {}".format(len(self.factors),
        #                                                                                         self.graph.error(
        #                                                                                             self.initialEstimate),
        #                                                                                         self.graph.error(self.result)))

        # Create gtsam Frames using the camera positions after optimization
        self.gs_frames_of_bundle_after_optimization = [gtsam.gtsam.StereoCamera(x, gtsam_k) for x in self.get_poses()]

    def triangulate_track_point_in_last_relevant_frame(self, cur_track, optimized=False):
        """
        Given track - triangulate in in the last relevant frame (the track's last frame or the bundle's last frame)
        :return: the 3d triangulation
        """
        # get max frame (length of track, bounded by the window's length) - relative. NOT GLOBAL INDEX
        max_frame_for_track_in_global = min(cur_track.last_frame, self.last_frame_global)
        # get coord of track in the last frame (that's inside the bundle window)
        cur_track_coords_last_frame = cur_track.get_coords_in_frame(max_frame_for_track_in_global)
        # triangulate using the last frames coords
        max_frame_for_track_relative_to_bundle = max_frame_for_track_in_global - self.first_frame_global
        cur_p3 = None
        if optimized:
            cur_p3 = gs_triangulation(cur_track_coords_last_frame,
                                      self.gs_frames_of_bundle_after_optimization[
                                          max_frame_for_track_relative_to_bundle])
        else:
            cur_p3 = gs_triangulation(cur_track_coords_last_frame,
                                      self.gs_frames_of_bundle[max_frame_for_track_relative_to_bundle])
        return cur_p3

    def get_pos_at_idx(self, idx):
        camera_symbol = self.camera_nodes[idx]
        return self.result.atPose3(camera_symbol)

    def get_points(self):
        """
        :return: the 3d-points of the bundle in the coordinates of the first frame in the bundle
        """
        cur_graph_3d_points = []
        for point in self.point_nodes:
            point_location = self.result.atPoint3(point.get_symbol())
            cur_graph_3d_points.append(point_location)
        return cur_graph_3d_points

    def get_poses(self):
        """
        :return: the poses of the cameras of the bundle in the coordinates of the first camera in the bundle
        """
        cur_graph_camera_locations = []
        for camera_symbol in self.camera_nodes:
            camera_location = self.result.atPose3(camera_symbol)
            cur_graph_camera_locations.append(camera_location)
        return cur_graph_camera_locations

    def get_reprojection_error_of_track(self, track):
        """
        :return: the reprojection error of the given track -
        a list of distances, each corresponding to the number of frames from the triangulation
        """
        assert self.first_frame_global <= track.first_frame <= self.last_frame_global or \
               self.first_frame_global <= track.last_frame <= self.last_frame_global or \
               (track.first_frame <= self.first_frame_global and self.last_frame_global <= track.last_frame), "Track {},  and bundle first {} bundle last {}".format(track, self.first_frame_global, self.last_frame_global)

        # The 3D point of the track. In coordinates of the first frame of the bundle
        cur_p3 = self.triangulate_track_point_in_last_relevant_frame(track, True)

        max_frame_for_track_in_global = min(track.last_frame, self.last_frame_global)
        max_frame_for_track_relative_to_bundle = max_frame_for_track_in_global - self.first_frame_global

        min_frame_for_track_in_global = max(track.first_frame, self.first_frame_global)
        num_projections = max_frame_for_track_in_global - min_frame_for_track_in_global + 1

        gs_projections = []
        for i in range(num_projections):
            try:
                gs_projections.append(
                    self.gs_frames_of_bundle_after_optimization[max_frame_for_track_relative_to_bundle - i].project(cur_p3))
            except:
                return None

        # re-arange the projections as left/right projections arrays
        gs_left_coors = np.array([(gs_projections[i].uL(), gs_projections[i].v()) for i in range(len(gs_projections))])
        gs_right_coors = np.array([(gs_projections[i].uR(), gs_projections[i].v()) for i in range(len(gs_projections))])

        # get the coordinates in each image of the track
        coords = [track.get_coords_in_frame(cur_frame) for cur_frame in range(min_frame_for_track_in_global, max_frame_for_track_in_global + 1)]
        coords.reverse()
        gs_coords = [stereo_to_gs_stereo(x) for x in coords][:len(gs_projections)]
        # re-arange our coordinates-detection as left/right coordinates arrays
        left_coors = np.array([(gs_coords[i].uL(), gs_coords[i].v()) for i in range(len(gs_coords))])
        right_coors = np.array([(gs_coords[i].uR(), gs_coords[i].v()) for i in range(len(gs_coords))])

        # calc projection dist
        left_proj_dist = np.linalg.norm(gs_left_coors - left_coors, axis=1)
        right_proj_dist = np.linalg.norm(gs_right_coors - right_coors, axis=1)
        dists = (left_proj_dist + right_proj_dist) / 2

        # # Always return the same length of distances
        # zeros_to_append = np.zeros(constant_length_of_distances - dists.shape[0])
        # dists_constant_size = np.concatenate((dists, zeros_to_append))
        return dists
