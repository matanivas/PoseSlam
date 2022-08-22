from VAN_project.code.GraphsOptimization.BundleDir.BundleGraph import BundleGraph
from VAN_project.code.utils.utils import read_cameras, read_data_from_pickle, from_relative_to_global_transformtions, from_global_trans_to_gtsam_pose, write_data
from VAN_project.code.utils.paths import transformations_of_movie_path, all_tracks_path, all_frames_path
import numpy as np

MIN_GAP_BETWEEN_KEY_FRAMES = 2
MAX_GAP_BETWEEN_KEY_FRAMES = 20
LARGE_DIST = 10  # Kitti is at 10 HZ - i.e. 5-20 frames is 0.5-2 seconds. Then 10 meters dist is between 5 and 20 m/s
SIMILAR_HEADING_THRESH = 0.99
RADIANS_NOISE = 0.001 * (np.pi / 180)**2


def cos(A, B):
    """ compute cos between vectors or matrices
    https://math.stackexchange.com/questions/2113634/comparing-two-rotation-matrices
    """
    Aflat = A.reshape(-1)  # views
    Bflat = B.reshape(-1)
    return np.dot(Aflat, Bflat)/max(np.linalg.norm(Aflat) * np.linalg.norm(Bflat), 1e-10 )


class BundlesManager:

    def __init__(self):
        # get the cameras matrices
        k, m1, m2 = read_cameras()
        self._K = k
        self._m1 = m1
        self._m2 = m2

        # Read relative transformations of drive
        self._relative_transformations = read_data_from_pickle(transformations_of_movie_path)
        # Transform them to global
        self._global_transformations = from_relative_to_global_transformtions(self._relative_transformations)

        # Read track db
        self._all_tracks = read_data_from_pickle(all_tracks_path)
        # Read frames db
        self._all_frames = read_data_from_pickle(all_frames_path)

        # Place holder for keyframes
        self._keyframes_list = []


        # Place holder for all bundles
        self._bundles_output = []

    def process_drive(self):
        self.__split_drive_to_keyframes()

        for bundle_range in self._keyframes_list:
            print("Bundle is {}".format(bundle_range))
            trans_of_bundle = from_relative_to_global_transformtions(
                self._relative_transformations[bundle_range[0]: bundle_range[1]])
            poses_of_bundle = from_global_trans_to_gtsam_pose(trans_of_bundle)

            # create using the new BundleGraph class.
            cur_bundle_window = BundleGraph(bundle_range, poses_of_bundle, self._K, self._m1, self._m2)
            cur_bundle_window.optimize(self._all_frames, self._all_tracks)
            self._bundles_output.append(cur_bundle_window)

    def __split_drive_to_keyframes(self):

        # create keyframes list
        # the criterion for choosing the keyframes would be either large distance or large lateral change,
        # the distance can be computed using our approximations for t's (of the Rt's).
        # the large lateral change can be computed using the R's (of the Rt's).
        cur_first_key_frame = 0

        for i, Rt in enumerate(self._global_transformations):
            is_key_frame = False
            # don't allow too short bundles
            if i < cur_first_key_frame + MIN_GAP_BETWEEN_KEY_FRAMES or (
                    i + MIN_GAP_BETWEEN_KEY_FRAMES >= len(self._global_transformations) - 1 and not i == len(self._global_transformations) - 1):
                continue
            # bound the maximum length of a bundle, or - reached the end of the movie
            elif i > cur_first_key_frame + MAX_GAP_BETWEEN_KEY_FRAMES or i == len(self._global_transformations) - 1:
                is_key_frame = True
            else:
                # if we passed enough distance - create a key frame
                dist = np.linalg.norm([self._global_transformations[cur_first_key_frame][:, 3] - self._global_transformations[i][:, 3]])
                # if we changed heading significantly - create a key frame
                heading_similarity = cos(self._global_transformations[cur_first_key_frame][:, :3], self._global_transformations[i][:, :3])

                if dist > LARGE_DIST or heading_similarity < SIMILAR_HEADING_THRESH:
                    is_key_frame = True

            if is_key_frame:
                self._keyframes_list.append((cur_first_key_frame, i))
                cur_first_key_frame = i

    def get_bundle(self, i):
        return self._bundles_output[i]

    def get_bundle_that_contains_frame(self, frame):
        for i, keyframe_tuple in enumerate(self._keyframes_list):
            if keyframe_tuple[0] <= frame <= keyframe_tuple[1]:
                return self.get_bundle(i)

    def get_num_bundles(self):
        return len(self._bundles_output)

    def get_keyframes_list(self):
        return self._keyframes_list

    def save(self, path):
        # pickle can't handle the gtsam K
        write_data(path, self)