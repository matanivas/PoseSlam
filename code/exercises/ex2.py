from VAN_project.code.utils import utils
import cv2
import matplotlib.pyplot as plt
import numpy as np
from VAN_project.code.utils.matching_utils import get_y_distances


def print_dists_percentage(dists):
    gt = 0
    for dist in dists:
        if dist > 2:
            gt += 1
    print(f'percentage of more than 2 pixels deviation is {(100 * gt) / len(dists)}')


def discard_matches_by_stereo_patten(matches, kpts1, kpts2):
    """
    given stereo-matches and keypoins - reject kpts using y-distance
    """
    dists = get_y_distances(matches, kpts1, kpts2)
    img1_accepted_coors = []
    img1_rejected_coors = []
    img2_accepted_coors = []
    img2_rejected_coors = []
    for i in range(len(dists)):
        if dists[i] < 1:
            img1_accepted_coors.append(kpts1[matches[i].queryIdx].pt)
            img2_accepted_coors.append(kpts2[matches[i].trainIdx].pt)
        else:
            img1_rejected_coors.append(kpts1[matches[i].queryIdx].pt)
            img2_rejected_coors.append(kpts2[matches[i].trainIdx].pt)
    return np.array(img1_accepted_coors), np.array(img1_rejected_coors), np.array(img2_accepted_coors),\
           np.array(img2_rejected_coors)


def plot_accepted_and_rejected(img1, img2, img1_accepted_coors, img1_rejected_coors,
                               img2_accepted_coors, img2_rejected_coors):
    """
    plot on images the accepted and rejected coords
    """
    figure = plt.figure(figsize=(9, 6))
    figure.add_subplot(2, 1, 1)
    plt.title('image 1 (left)')
    plt.imshow(img1, cmap='gray')
    plt.scatter(img1_accepted_coors[:, 0], img1_accepted_coors[:, 1], s=2, color='orange')
    plt.scatter(img1_rejected_coors[:, 0], img1_rejected_coors[:, 1], s=2, color='cyan')

    figure.add_subplot(2, 1, 2)
    plt.title('image 2 (left)')
    plt.imshow(img2, cmap='gray')
    plt.scatter(img2_accepted_coors[:, 0], img2_accepted_coors[:, 1], s=2, color='orange')
    plt.scatter(img2_rejected_coors[:, 0], img2_rejected_coors[:, 1], s=2, color='cyan')
    plt.show()


def linear_least_squares(img1_m, img2_m, kpts1_coords, kpts2_coords):
    """
    perform svd for triangulation
    """
    rc1 = kpts1_coords[0] * img1_m[2] - img1_m[0]
    rc2 = kpts1_coords[1] * img1_m[2] - img1_m[1]
    rc3 = kpts2_coords[0] * img2_m[2] - img2_m[0]
    rc4 = kpts2_coords[1] * img2_m[2] - img2_m[1]
    mat = np.array([rc1, rc2, rc3, rc4])
    _1, _2, VT = np.linalg.svd(mat, compute_uv=True)
    return VT[-1]


def triangulate(img1_m, img2_m, kpts1_coords, kpts2_coords):
    ans = []
    for i in range(len(kpts1_coords)):
        lls = linear_least_squares(img1_m, img2_m, kpts1_coords[i], kpts2_coords[i])
        ans.append(lls[:3] / lls[3])
    return np.array(ans)


def plot_triangulations(ourT, cv2T):
    """
    Draws the 3d points triangulations (Our and open-cv)
    """
    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_title("Our Triangulation")
    ax.scatter3D(ourT[:, 0], ourT[:, 1], ourT[:, 2])
    ax.set_xlim3d(-20, 20)
    ax.set_ylim3d(-20, 20)
    ax.set_zlim3d(-200, 200)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_title("Open-cv's Triangulation")
    ax.scatter3D(cv2T[:, 0], cv2T[:, 1], cv2T[:, 2])
    ax.set_xlim3d(-20, 20)
    ax.set_ylim3d(-20, 20)
    ax.set_zlim3d(-200, 200)
    plt.show()


def triangulate_several_imgs(n):
    k, m1, m2 = utils.read_cameras()
    img1_m = k @ m1
    img2_m = k @ m2
    for i in range(n):
        img1, img2 = utils.read_images(i)
        alg = cv2.SIFT_create()
        img1_kpts, img1_dscs = alg.detectAndCompute(img1, None)
        img2_kpts, img2_dscs = alg.detectAndCompute(img2, None)
        brute_force = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = brute_force.match(img1_dscs, img2_dscs)
        img1_accepted_coors, img1_rejected_coors, img2_accepted_coors, img2_rejected_coors = \
            discard_matches_by_stereo_patten(matches, img1_kpts, img2_kpts)
        ourT = triangulate(img1_m, img2_m, img1_accepted_coors, img2_accepted_coors)
        cv2T = cv2.triangulatePoints(img1_m, img2_m, img1_accepted_coors.T, img2_accepted_coors.T).T
        cv2T = cv2.convertPointsFromHomogeneous(cv2T).reshape(img1_accepted_coors.shape[0], 3)
        plot_triangulations(ourT, cv2T)


if __name__ == '__main__':
    # q1
    img1, img2 = utils.read_images(0)
    alg = cv2.SIFT_create()
    img1_kpts, img1_dscs = alg.detectAndCompute(img1, None)
    img2_kpts, img2_dscs = alg.detectAndCompute(img2, None)

    brute_force = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = brute_force.match(img1_dscs, img2_dscs)
    dists = get_y_distances(matches, img1_kpts, img2_kpts)
    print_dists_percentage(dists)
    n, bins, patches = plt.hist(dists)
    plt.show()

    # q2
    img1_accepted_coors, img1_rejected_coors, img2_accepted_coors, img2_rejected_coors = \
        discard_matches_by_stereo_patten(matches, img1_kpts, img2_kpts)
    print(f'of {len(img1_accepted_coors) + len(img1_rejected_coors)} matches, the stereo pattern had been able to '
          f'reject {len(img1_rejected_coors)}')
    plot_accepted_and_rejected(img1, img2, img1_accepted_coors, img1_rejected_coors,
                               img2_accepted_coors, img2_rejected_coors)

    # q3
    k, m1, m2 = utils.read_cameras()
    img1_m = k @ m1
    img2_m = k @ m2
    ourT = triangulate(img1_m, img2_m, img1_accepted_coors, img2_accepted_coors)
    cv2T = cv2.triangulatePoints(img1_m, img2_m, img1_accepted_coors.T, img2_accepted_coors.T).T
    cv2T = cv2.convertPointsFromHomogeneous(cv2T).reshape(img1_accepted_coors.shape[0], 3)
    plot_triangulations(ourT, cv2T)

    print(f'the norm of absolute value of differences is: {np.linalg.norm(np.abs(ourT - cv2T))}')



