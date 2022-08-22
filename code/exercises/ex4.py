from VAN_project.code.utils import utils
import time
from VAN_project.code.utils.paths import DATA_PATH, GT_PATH, images_path, transformations_of_movie_path, \
    all_tracks_path, all_frames_path, inliers_path
from VAN_project.code.FramesDir.Drive import Drive

COLORS_LIST = ["blue", "orange", "grey"]

if __name__ == '__main__':
    start_time = time.time()
    cur_drive = Drive()
    cur_drive.run()
    end_time = time.time()
    print("Time of tracking throughout the movie is: {} minutes".format(((end_time - start_time) / 60)))
    cur_drive.save()
