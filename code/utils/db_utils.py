import numpy as np
from VAN_project.code.TracksDir.track import Track
from VAN_project.code.utils import utils
import datetime

MONTH = datetime.datetime.today().month
DAY = datetime.datetime.today().day

def read_gt_transformations(path_to_gt):
    """
    returns a list of the GT extrinsic matrices for the drive, with respect to the "world coordinates" (t0)
    """
    gt_transformations = []
    f = open(path_to_gt, "r")
    for line in f:
        cur_mat = np.fromstring(line, dtype=float, sep=' ').reshape((3, 4))
        gt_transformations.append(cur_mat)

    return gt_transformations


def get_track_ids_of_frame(frame_id, frames_to_tracks_dict):
    """
    given frame id return the track-ids that appear in the frame
    :param frame_id:
    :param frames_to_tracks_dict:
    :return: list of track-ids in the frame
    """
    return frames_to_tracks_dict[str(frame_id)]["track_ids"]


def get_frames_of_track_id(track_id, track_ids_to_frames_dict):
    """
    given track-id return a tuple (first-frame, last-frame) it appears in
    :param track_id:
    :param track_ids_to_frames_dict:
    :return:
    """
    return track_ids_to_frames_dict[str(track_id)]


def get_coordinates_of_track_id_in_frame(frame_id, track_id, frames_to_tracks_dict):
    """
    given frame-id and track-id - return coordinates in cameras
    :param frame_id:
    :param track_id:
    :param frames_to_tracks_dict:
    :return:
    """
    track_id = int(track_id)
    track_ids_in_frame = frames_to_tracks_dict[str(frame_id)]["track_ids"]
    for i, id in enumerate(track_ids_in_frame):
        if id == track_id:
            return (frames_to_tracks_dict[str(frame_id)]["left_kpts"][i][0],
                    frames_to_tracks_dict[str(frame_id)]["right_kpts"][i][0],
                    frames_to_tracks_dict[str(frame_id)]["left_kpts"][i][1])


# def convert_old_dicts_to_use_tracks(uncleaned_tracks_to_frames_dict, uncleaned_frames_to_tracks_dict):
#     """
#     use the old dicts to create the new dict while removing tracks of length 1.
#     :param uncleaned_tracks_to_frames_dict:
#     :param uncleaned_frames_to_tracks_dict:
#     :return:
#     """
#     print("started running....")
#     create dictionary of tracks
#     all_tracks = {}
#     for track_id in uncleaned_tracks_to_frames_dict:
#         frames = uncleaned_tracks_to_frames_dict[track_id]
#         coordinates_first_frame = get_coordinates_of_track_id_in_frame(frames[0], track_id,
#                                                                                 uncleaned_frames_to_tracks_dict)
#         new_track = Track(track_id, frames[0], coordinates_first_frame)
#         for frame in range(frames[1] - frames[0]):
#             cur_frame = frame + 1  # already pushed the first element
#             coords_to_push = get_coordinates_of_track_id_in_frame(cur_frame, track_id,
#                                                                            uncleaned_frames_to_tracks_dict)
#             new_track.push_point_to_track(coords_to_push)
#
#         all_tracks[track_id] = new_track
#
#     print("finished creating tracks. Track 916 is {}".format(all_tracks["916"].get_coords_in_frame(1)))
#     # remove tracks of length 1
#     all_tracks_of_length_larger_than_one = {}
#     for track_key in all_tracks:
#         if all_tracks[track_key].length > 1:
#             all_tracks_of_length_larger_than_one[track_key] = all_tracks[track_key]
#
#     print("finished cleaning tracks Track 916 is {}".format(all_tracks_of_length_larger_than_one["916"].get_coords_in_frame(1)))
#
#     # create dictionary of frames
#     all_frames = {}
#     for frame in range(len(uncleaned_frames_to_tracks_dict.keys())):
#         all_frames[str(frame)] = []
#         for track_id in all_tracks_of_length_larger_than_one:
#             if all_tracks_of_length_larger_than_one[track_id].is_in_frame(frame):
#                 all_frames[str(frame)].append(track_id)
#
#     utils.write_data("all_tracks_27_5", all_tracks_of_length_larger_than_one)
#     utils.write_data("all_frames_27_5", all_frames)

def from_tracks_dict_to_frames_dict(track_dict, total_num_frames, write_data=False):
    """
    create a frames dict, using the new struct - track dict
    :param track_dict: keys: track_id, values: Track object
    :return: dictionary - keys: frames, values: list of tracks that appear in that frame
    """
    # remove tracks of length 1
    all_tracks_of_length_larger_than_one = {}
    for track_key in track_dict:
        if track_dict[track_key].length > 1:
            all_tracks_of_length_larger_than_one[track_key] = track_dict[track_key]
    print("finished cleaning tracks.")

    # create dictionary of frames
    all_frames = {}
    for i in range(total_num_frames):
        all_frames[str(i)] = set()
    print("num of tracks {}".format(len(all_tracks_of_length_larger_than_one.keys())))
    j = 0
    for track_id, track in all_tracks_of_length_larger_than_one.items():
        for i in range(track.length):
            all_frames[str(track.first_frame+i)].add(track_id)
        j += 1
        print("passed track number {}".format(j))

    if write_data:
        utils.write_data("./data_{}_{}_filtered/final_cleaned_all_tracks_{}_{}".format(DAY, MONTH, DAY, MONTH), all_tracks_of_length_larger_than_one)
        utils.write_data("./data_{}_{}_filtered/final_cleaned_all_frames_{}_{}".format(DAY, MONTH, DAY, MONTH), all_frames)
    return all_tracks_of_length_larger_than_one, all_frames


# if __name__ == '__main__':
#
#     # get the tracks data
#     frame_to_trackids_dict = utils.read_data_from_pickle("./data_back_up/frame_to_trackids_dict_24_5.pickle")
#     trackid_to_frames_dict = utils.read_data_from_pickle("./data_back_up/trackid_to_frames_dict_24_5.pickle")
#
#     convert_old_dicts_to_use_tracks(trackid_to_frames_dict, frame_to_trackids_dict)