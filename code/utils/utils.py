import numpy as np
import cv2
import pickle
from VAN_project.code.utils import paths
from VAN_project.code.utils.rts_utils import two_transformations_composition, ZEROS_ONE
from VAN_project.code.TracksDir.track import Track
import gtsam
import matplotlib.pyplot as plt
import os

def read_cameras():
    """
    :return: k, m1,m2 (cameras intrinsic data)
    """
    with open(paths.DATA_PATH + 'calib.txt') as f:
        l1 = f.readline().split()[1:]  # skip first token
        l2 = f.readline().split()[1:]  # skip first token
        l1 = [float(i) for i in l1]
        m1 = np.array(l1).reshape(3, 4)
        l2 = [float(i) for i in l2]
        m2 = np.array(l2).reshape(3, 4)
        k = m1[:, :3]
        m1 = np.linalg.inv(k) @ m1
        m2 = np.linalg.inv(k) @ m2
        return k, m1, m2


def read_images(idx):
    """
    :param idx: Images's index in the Kitti dataset
    :return: left and right cameras photos
    """
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(paths.DATA_PATH + 'image_0/' + img_name, 0)
    img2 = cv2.imread(paths.DATA_PATH + 'image_1/' + img_name, 0)
    return img1, img2


def sample_track_longer_than_n(n, all_tracks, rand=True):
    """
    :param n: length of track to sample
    :param trackid_to_frames: the db of tracks
    :return: the chosen track
    """
    if rand:
        tracks_longer_than_n = []
        for track_id, track in all_tracks.items():
            if track.length > n:
                tracks_longer_than_n.append(track)
        return np.random.choice(tracks_longer_than_n)
    for track_id, track in all_tracks.items():
        if track.length > n:
            return track
    return -1


def read_data_from_pickle(pkl_path):
    """
    returns a list of the relative transformations that were written to a pickle
    """
    objects = []
    with (open(pkl_path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    return objects[0]


def from_relative_to_global_transformtions(rts):
    """
    :param rts: a list of relative transformations
    :return: a corresponding list of global transformations
    """
    prev = np.append(np.identity(3), np.zeros((3, 1)), axis=1)
    global_rts = []
    global_rts.append(prev)

    # loop and push al following locations
    for Rt in rts:
        prev = two_transformations_composition(prev, Rt)
        global_rts.append(prev)
    return np.array(global_rts)


def from_relative_to_global_transformtions_gtsam(rts):
    """
    :param rts: a list of gtsam relative transformations in WORLD TO CAM
    :return: a corresponding list of global transformations WORLD TO CAM
    """
    global_rts = []
    global_rts.append(rts[0])
    prev = global_rts[0]
    # loop and push al following locations
    for Rt in rts[1:]:
        prev = Rt.compose(prev)
        global_rts.append(prev)
    return global_rts


def from_gtsam_relative_to_global(rel_poses):
    """
    :param rel_poses: gtsam relative poses in CAM TO WORLD
    :return: gtsam global poses in cam to world  CAM TO WORLD
    """
    rel_world_to_cam = [pos.inverse() for pos in rel_poses]
    gtsam_global_world_to_cam = from_relative_to_global_transformtions_gtsam(rel_world_to_cam)
    gtsam_global_cam_to_world = [pos.inverse() for pos in gtsam_global_world_to_cam]
    return gtsam_global_cam_to_world


def write_data(data_name, data):
    """
    write data into a pickle named by data_name
    """
    # write the dictionaries to pickle
    if "pickle" not in data_name:
        data_name = "{}.pickle".format(data_name)
    # If directory does not exist - create it
    os.makedirs(os.path.dirname(data_name), exist_ok=True)
    with open(data_name, 'wb') as f:
        pickle.dump(data, f)


def from_global_trans_to_gtsam_pose(rts):
    poses = []
    for i, rt in enumerate(rts):
        R = rt[:, :3].T
        t = -1 * R @ rt[:, 3]
        R = gtsam.Rot3(R)
        t = gtsam.Point3(t)
        poses.append(gtsam.Pose3(R, t))
    return poses


def plot_point_on_stereo_images(img1, img2, our_stereo_point, color, title = ""):
    """
    plot on images the accepted and rejected coords
    """
    figure = plt.figure(figsize=(9, 6))
    plt.title(title)
    figure.add_subplot(2, 1, 1)
    plt.title('image 1 (left)')
    plt.imshow(img1, cmap='gray')
    plt.scatter(our_stereo_point[0], our_stereo_point[2], s=4, color=color)
    # plt.scatter(gtsam_stereo_point.uL(), gtsam_stereo_point.v(), s=2, color="blue")

    figure.add_subplot(2, 1, 2)
    plt.title('image 2 (right)')
    plt.imshow(img2, cmap='gray')
    plt.scatter(our_stereo_point[1], our_stereo_point[2], s=4, color=color)
    # plt.scatter(gtsam_stereo_point.uR(), gtsam_stereo_point.v(), s=2, color="blue")
    plt.savefig("temp_yesterday.png")

def compose_list_of_relatives(lst_of_rels):
    prev = np.append(np.identity(3), np.zeros((3, 1)), axis=1)
    for i in range(lst_of_rels.shape[0]-1, -1, -1):
        Rt = lst_of_rels[i]
        prev = two_transformations_composition(prev, Rt)
    return prev

def inverse_rt(rt):
    loc = rt[:, 3]
    rot = rt[:, : 3]
    inv_rot = rot.T
    inv_loc = - rot.T @ loc
    return np.hstack((inv_rot, inv_loc.reshape(3, 1)))