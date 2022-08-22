import numpy as np


class Track:

    def __init__(self, uid, first_frame, coordinates_first_frame):
        self.uid = uid
        self.first_frame = first_frame
        self.last_frame = first_frame
        self.length = 1
        self.coordinates = [coordinates_first_frame]

    def __repr__(self):
        return "uid {} \n first frame {} \n last frame {} \n length {}".format(self.uid, self.first_frame,
                                                                               self.last_frame, self.length)

    def __str__(self):
        return "uid {} \n first frame {} \n last frame {} \n length {}".format(self.uid, self.first_frame,
                                                                               self.last_frame, self.length)

    def push_point_to_track(self, coords_to_push):
        """
        update a track - change the last frame to be the new last frame, and store the left and right points
        of the new last frame in their corresponding places.
        :param cur_last_frame: The new last frame
        :param left_point: the left point - in the last new frame
        :param right_point: the right point in the last new frame
        :return: None
        """
        self.length += 1
        self.last_frame += 1
        self.coordinates.append(coords_to_push)

    def get_coords_in_frame(self, frame):
        """
        return the coordsinates of the track in frame.
        IMPORTANT - frame is global!!
        :param frame:
        :return:
        """
        assert self.is_in_frame(frame)
        return self.coordinates[frame - self.first_frame]

    def is_in_frame(self, query_frame):
        """
        return in the track appers in that frame
        :param query_frame:
        :return:
        """
        return self.first_frame <= query_frame <= self.last_frame

    def is_good_track(self):
        coords_as_np = np.array(self.coordinates)
        diff = np.abs(np.diff(coords_as_np, axis=0))  # compute the delta over time for each of the coordinates
        max_vals = np.max(diff, axis=0)  # compute the max-delta over time for each of the coordinates
        maximum_diff_in_coordinates = np.max(max_vals)  # get the maximum delta over all coordinates
        return maximum_diff_in_coordinates < 70 and self.length > 2
