from VAN_project.code.FramesDir.Drive import Drive
from VAN_project.code.utils import paths, utils
import cv2

def get_weak_frames():
    # todo - return a list of weak frames based on the connectivity or number of matches graphs (of the whole drive)
    inliers_dict = utils.read_data_from_pickle(paths.inliers_path)
    num_of_frames = len(inliers_dict.keys())
    percentages = [(0, 0)] * num_of_frames
    for frame in range(1, num_of_frames):
        percentages[frame] = (inliers_dict[str(frame)], frame)

    percentages.sort(key=lambda x: x[0])
    return percentages[1:5]

if __name__ == '__main__':

    # todo - get list of frames with bad connectivity
    weak_frames = get_weak_frames()

    # for each "weak" frame - test stats with other detectors
    for weak_frame in weak_frames:
        print("Original percentage of inliers: {}".format(weak_frame[0]))

        frame = weak_frame[1]

        frame_before = frame - 1
        frame_after = frame + 1
        frames_to_process = (frame_before, frame, frame_after)

        detectors = [cv2.AKAZE_create(), cv2.BRISK_create(), cv2.ORB_create(), cv2.SIFT_create()]

        for detector in detectors:
            cur_drive = Drive(frames_to_process, loop_closure_consensus_matching=False, detector=detector)
            cur_drive.run()

            matches_per_frame = cur_drive.get_matches_per_frame_dict()
            inliers_per_frame = cur_drive.get_inliers_per_frame_dict()

            # todo - use those dicts to output graphs - and see if some detector performs better then the first (our default - Akaze)
            print("For detector {}".format(detector))
            print(matches_per_frame)
            print(inliers_per_frame)


        a = 5