from VAN_project.code.TracksDir.track import Track
from VAN_project.code.FramesDir.MultiFrame import MultiFrame
from VAN_project.code.FramesDir.StereoFrame import StereoFrame
from VAN_project.code.utils.matching_utils import Uid
from VAN_project.code.utils.rts_utils import project
import numpy as np
from copy import deepcopy

class TrackDb:

    def __init__(self, stereo_frame: StereoFrame):
        # Tracks dict - each key is a track-id and the value is the track object
        self._tracks_dict = {}

        # Frame to tracks mapping
        self._frames_dict = {}

        # Create uid generator
        self._uid_generator = Uid()

        # init the first-stereo data to the tracks dict
        self._track_ids = [self._uid_generator.get_next_uid() for i in range(stereo_frame.get_num_matches())]

        for i, track_id in enumerate(self._track_ids):
            coords_in_first_frame = stereo_frame.get_left_right_point(i)
            cur_track = Track(track_id, stereo_frame.get_idx(), coords_in_first_frame)
            self._tracks_dict[str(track_id)] = cur_track

        # Data of last frame in db - left-kpts, right-kpts and track-ids (correspondingly)
        self._last_frame_track_ids = deepcopy(self._track_ids)

        # Inliers percentage per frame
        self._inliers_dict = {}


    def update(self, multiframe: MultiFrame, cur_frame_id):
        left_projection_rt_using_best_rt = multiframe.get_best_left_projection()
        right_projection_rt_using_best_rt = multiframe.get_best_right_projection()
        matches = multiframe.get_matches()

        track_ids = [-1] * multiframe.get_second_stereo().get_num_matches()
        number_of_matches_rejected_using_rt = 0
        final_good_match_idxs = []  # the index of matches after filtering based on Rt
        for idx in range(len(matches)):
            # get the track id of the point from the previous frame that matched
            cur_track_id = self._last_frame_track_ids[matches[idx].queryIdx]

            # get the coordinates that match the point-track in the current frame. (x_left, x_right, y(avg))
            coords_to_push = multiframe.get_second_stereo().get_left_right_point(matches[idx].trainIdx)

            # move the 3d point from the coordinates of the previous frame to the coordinates of the
            # current frame, project it on the cameras - and then calc the delta between the projection and
            # coords to push - if is large - continue (i.e. don't update the track (bad point).

            p3_in_prev_frame_world = multiframe.get_first_stereo().get_three_d_pts()[matches[idx].queryIdx]
            # move the p3 to the world of the current frame, and project
            proj_cur_left = project(p3_in_prev_frame_world, left_projection_rt_using_best_rt)
            proj_cur_right = project(p3_in_prev_frame_world, right_projection_rt_using_best_rt)
            proj_corrds = np.array(
                [proj_cur_left[0], proj_cur_right[0], (proj_cur_left[1] + proj_cur_right[1]) / 2.0])

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
            coords_to_push = multiframe.get_second_stereo().get_left_right_point(matches[idx].trainIdx)
            self._tracks_dict[str(cur_track_id)].push_point_to_track(coords_to_push)

        # update inliers percentage of the current frame
        num_of_potential_points_in_prev_frame = multiframe.get_first_stereo().get_num_matches()
        self._inliers_dict[str(cur_frame_id)] = (num_of_potential_points_in_prev_frame - number_of_matches_rejected_using_rt) * 100 / num_of_potential_points_in_prev_frame

        for i, id in enumerate(track_ids):
            if id == -1:  # i.e. it's a point that was not matched to some previous point. >> start a new track
                next_uid = self._uid_generator.get_next_uid()

                track_ids[i] = next_uid
                # push the new track to all_tracks_dict
                coords_to_push = multiframe.get_second_stereo().get_left_right_point(i)
                self._tracks_dict[str(next_uid)] = Track(next_uid, cur_frame_id, coords_to_push)

        # Data of last frame in db - left-kpts, right-kpts and track-ids (correspondingly)
        self._last_frame_track_ids = deepcopy(track_ids)

    def clean_and_create_frames_db(self, total_num_frames, specific_frames=None):
        # remove tracks of length 1
        all_tracks_of_length_larger_than_one = {}
        for track_key in self._tracks_dict:
            if self._tracks_dict[track_key].length > 1:
                all_tracks_of_length_larger_than_one[track_key] = self._tracks_dict[track_key]
        print("finished cleaning tracks.")

        # create dictionary of frames
        for i in range(total_num_frames):
            if specific_frames:
                self._frames_dict[str(specific_frames[i])] = set()
            else:
                self._frames_dict[str(i)] = set()
        # print("num of tracks {}".format(len(all_tracks_of_length_larger_than_one.keys())))
        for track_id, track in all_tracks_of_length_larger_than_one.items():
            for i in range(track.length):
                self._frames_dict[str(track.first_frame + i)].add(track_id)

        # update the tracks dict to contain only tracks longer than 1
        self._tracks_dict = all_tracks_of_length_larger_than_one

    def get_tracks_db(self):
        return self._tracks_dict

    def get_frames_db(self):
        return self._frames_dict

    def get_inliers_percentage(self):
        return self._inliers_dict
