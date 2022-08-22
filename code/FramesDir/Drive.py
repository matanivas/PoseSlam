from VAN_project.code.utils import utils
from VAN_project.code.utils.paths import DATA_PATH, GT_PATH, images_path
from VAN_project.code.utils.matching_utils import Uid
from VAN_project.code.FramesDir.StereoFrame import StereoFrame
from VAN_project.code.FramesDir.MultiFrame import MultiFrame
from VAN_project.code.TracksDir.TrackDb import TrackDb
from VAN_project.code.utils.paths import images_path, transformations_of_movie_path, \
    all_tracks_path, all_frames_path, inliers_path, num_matches_per_frame_path
import os

MINIMUM_MATCHES_FOR_PNP = 150
CONSENSUS_MATCHING_THRESHOLD = 40


class Drive:

    def __init__(self, frames_range=None, loop_closure_consensus_matching=False, detector=None):
        # read num of frames in drive
        self._images_range = None
        if frames_range is None:
            self._images_range = range(len([name for name in os.listdir(images_path)]))
        else:
            self._images_range = frames_range

        self._loop_closure_consensus_matching = loop_closure_consensus_matching
        # Flag - succeeded in match - for loop_closure_consensus_matching
        self._succeeded_consensus_match = False

        # read cameras physical data.
        K, m1, m2 = utils.read_cameras()
        self._K = K
        self._m2 = m2
        self._img1_m = K @ m1
        self._img2_m = K @ m2

        # transformations of drive
        self._transformations_of_drive = []

        # Create Uid generator for tracks
        self._uid_generator = Uid()

        # Create Db to hold all of the tracks.
        self._track_db = None

        # For stats - save the number of matches per stereo frame
        self._matches_per_frame_dict = {}

        # Detector to use
        self._detector = detector


    def run(self):
        # Process first stereo image
        first_stereo = StereoFrame(self._images_range[0], self._img1_m, self._img2_m, detector=self._detector)
        # save num of matches for stats
        self._matches_per_frame_dict[str(self._images_range[0])] = first_stereo.get_num_matches()
        # init the tracks db
        self._track_db = TrackDb(first_stereo)

        for i in self._images_range[1:]:
            second_stereo = StereoFrame(i, self._img1_m, self._img2_m, detector=self._detector)
            # save num of matches for stats
            self._matches_per_frame_dict[str(i)] = second_stereo.get_num_matches()

            # Process the multiframe data to extract the transformation between frames
            multi_frame = MultiFrame(first_stereo, second_stereo, self._K, self._m2)
            multi_frame.match()

            # In case used of loop-closure - stop here if didn't find enough matches
            if self._loop_closure_consensus_matching and len(multi_frame.get_matches()) < 0.75 * MINIMUM_MATCHES_FOR_PNP:
                self._succeeded_consensus_match = False
                return
            # run ransac
            multi_frame.ransac()
            if self._loop_closure_consensus_matching and (multi_frame.get_best_transformation() is None
                                                          or len(
                        multi_frame.get_best_supporters_idx()) < 0.2*MINIMUM_MATCHES_FOR_PNP):
                self._succeeded_consensus_match = False
                return

            # Refine transformation using all supporters
            multi_frame.refine_transformation()

            # In case used of loop-closure - stop here if didn't find enough matches
            if self._loop_closure_consensus_matching and (multi_frame.get_best_transformation() is None
                                                          or len(multi_frame.get_best_supporters_idx()) < CONSENSUS_MATCHING_THRESHOLD):
                self._succeeded_consensus_match = False
                return
            else:
                self._succeeded_consensus_match = True

            # update the tracks db
            self._track_db.update(multi_frame, i)

            self._transformations_of_drive.append(multi_frame.get_best_transformation())

            # update the image at t1 to be the image of t0 of next round
            first_stereo = second_stereo

            print("Finished image number {}".format(i))

        self._track_db.clean_and_create_frames_db(len(self._images_range), specific_frames=self._images_range)

    def get_transformations(self):
        return self._transformations_of_drive

    def get_tracks_db(self):
        return self._track_db.get_tracks_db()

    def get_frames_db(self):
        return self._track_db.get_frames_db()

    def get_success_for_loop_closure(self):
        return self._succeeded_consensus_match

    def get_matches_per_frame_dict(self):
        return self._matches_per_frame_dict

    def get_inliers_per_frame_dict(self):
        return self._track_db.get_inliers_percentage()

    def save(self, output_dir=None):
        cur_transformations_of_movie_path = transformations_of_movie_path
        cur_all_tracks_path = all_tracks_path
        cur_all_frames_path = all_frames_path
        cur_inliers_path = inliers_path
        cur_num_matches_per_frame_path = num_matches_per_frame_path
        if output_dir is not None:
            prefix = "../" + output_dir
            split_by = '..'
            cur_transformations_of_movie_path = prefix + cur_transformations_of_movie_path.split(split_by)[1]
            cur_all_tracks_path = prefix + cur_all_tracks_path.split(split_by)[1]
            cur_all_frames_path = prefix + cur_all_frames_path.split(split_by)[1]
            cur_inliers_path = prefix + inliers_path.split(split_by)[1]
            cur_num_matches_per_frame_path = prefix + inliers_path.split(split_by)[1]

        utils.write_data(cur_transformations_of_movie_path, self.get_transformations())
        utils.write_data(cur_all_tracks_path, self.get_tracks_db())
        utils.write_data(cur_all_frames_path, self.get_frames_db())
        utils.write_data(cur_inliers_path, self._track_db.get_inliers_percentage())
        utils.write_data(cur_num_matches_per_frame_path, self._matches_per_frame_dict)
