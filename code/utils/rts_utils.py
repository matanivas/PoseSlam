import numpy as np
import cv2

ZEROS_ONE = [np.array([0, 0, 0, 1])]


def two_transformations_composition(prev, cur):
    """
    composition of two Rt's
    """
    prev_as_RT = np.append(prev, ZEROS_ONE, axis=0)
    return cur @ prev_as_RT


def get_positions_of_camera(list_of_Rts, m2=None):
    """
    given list of relative Rts - return the positions of the camera
    """
    prev = np.append(np.identity(3), np.zeros((3,1)), axis=1)
    relative_positions_of_cameras = []

    # push first location
    if m2 is not None:
        relative_positions_of_cameras.append(-m2[:, 3:])
    else:
        relative_positions_of_cameras.append(prev[:, 3:])

    # loop and push al following locations
    for Rt in list_of_Rts:
        prev = two_transformations_composition(prev, Rt)
        if m2 is not None:
            cur_right_camera_trans = two_transformations_composition(prev, m2)
            cur_right_camera_pos = -1 * cur_right_camera_trans[:, :3].T @ cur_right_camera_trans[:, 3:]
            relative_positions_of_cameras.append(cur_right_camera_pos)
        else:
            cur_relative_position = -1 * prev[:, :3].T @ prev[:, 3:]
            relative_positions_of_cameras.append(cur_relative_position)

    locations = np.array(relative_positions_of_cameras)
    locations = locations.reshape(locations.shape[0], locations.shape[1])
    return locations


def rotate_and_traslate(Rt, x):
    """
    :param Rt: rotation and translation matrix 3X4
    :param x: point 3X1
    :return: R@x + t
    """
    rotate = Rt[:, :3]
    translate = Rt[:, -1:].reshape(3, )
    return np.add(rotate @ x, translate)


def rodriguez_to_mat(rvec, tvec):
    rot, _ = cv2.Rodrigues(rvec)
    return np.hstack((rot, tvec))


def project(p3d_pts, cam_proj_mat):
    """
    :param p3d_pts: point in 3D
    :param cam_proj_mat: camera's projection matrix
    :return: the pixels of the point on the pictue - homogenous point (x,y,1) (? is it x or y first?)
    """
    hom_proj = (p3d_pts @ cam_proj_mat[:, :3].T + cam_proj_mat[:, 3].T).flatten()
    # return hom_proj[:, :2] / hom_proj[:, [-1]]
    return hom_proj/hom_proj[2] # todo this is a fix used for ex_4. Make sure it didn't break ex4_analysis
