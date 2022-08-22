from VAN_project.code.utils import utils
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import pickle
from VAN_project.code.utils.paths import images_path, transformations_of_movie_path
from VAN_project.code.utils.rts_utils import get_positions_of_camera, rotate_and_traslate
from VAN_project.code.utils.plot_utils import plot_relative_position_of_cameras
from VAN_project.code.FramesDir.StereoFrame import StereoFrame
from VAN_project.code.FramesDir.MultiFrame import MultiFrame

COLORS_LIST = ["blue", "orange", "grey"]


def repeat_for_the_whole_movie():
    """
    :return: all transformation computed for the movie. No track db created (ex4 and forward)
    """
    num_images = len([name for name in os.listdir(images_path)])

    # get images matrices
    K, m1, m2 = utils.read_cameras()
    img1_m = K @ m1
    img2_m = K @ m2

    transformations_of_movie = []

    start_time = time.time()
    first_stereo = StereoFrame(0, img1_m, img2_m)

    for i in range(1, num_images):
        second_stereo = StereoFrame(i, img1_m, img2_m)

        # Process the multiframe data to extract the transformation between frames
        multi_frame = MultiFrame(first_stereo, second_stereo, K, m2)
        multi_frame.match_and_filter_using_significance()
        multi_frame.ransac()
        multi_frame.refine_transformation()

        transformations_of_movie.append(multi_frame.get_best_transformation())

        # update the image at t1 to be the image of t0 of next round
        first_stereo = second_stereo

        print("Finished image number {}".format(i))

    end_time = time.time()
    print("Time of tracking throughout the movie is: {}".format(end_time - start_time))

    # write the transformations to a pickle to process later
    with open(transformations_of_movie_path, 'wb') as f:
        pickle.dump(transformations_of_movie, f)

    return transformations_of_movie


def plot_3D_point_clouds(multiframe, output_fig_name="./temp.png"):
    """
    :param images_over_time_data:
    :param transformation:
    :return: None
    """
    points_3d_t1 = np.array(multiframe.get_second_three_d())
    x_3d_t1 = points_3d_t1[:, :1]
    z_3d_t1 = points_3d_t1[:, -1:]
    points_3d_t0_transformed_to_t1 = np.array(
        [rotate_and_traslate(multiframe.get_best_transformation(), pt_3d) for pt_3d in multiframe.get_first_three_d()])
    x_3d_t0 = points_3d_t0_transformed_to_t1[:, :1]
    z_3d_t0 = points_3d_t0_transformed_to_t1[:, -1:]

    fig, ax = plt.subplots()
    plt.scatter(x_3d_t1, z_3d_t1, color="grey")
    plt.scatter(x_3d_t0, z_3d_t0, color="blue")
    plt.xlabel("X axis")
    plt.ylabel("Z axis")
    plt.xlim(-50, 50)
    plt.ylim(0, 50)
    plt.title("Top view - first pair (blue) after transformation VS. second pair")
    plt.savefig(output_fig_name)
    plt.close()


def plot_matches_and_supporters(img1, img2, images_over_time_data, supporters_l0, supporters_l1, supporters_or_inliers):
    # l0_matches = np.array([kpnt for kpnt in images_over_time_data["left_t0_kpts"]])
    # l1_matches = np.array([kpnt for kpnt in images_over_time_data["left_t1_kpts"]])

    nice_l0_matches = np.array(images_over_time_data["left_t0_kpts"])
    nice_l1_matches = np.array(images_over_time_data["left_t1_kpts"])


    figure = plt.figure(figsize=(9, 6))
    figure.add_subplot(2, 1, 1)
    plt.title('Image 1 (left at t0) - {} in cyan'.format(supporters_or_inliers))
    plt.imshow(img1, cmap='gray')
    plt.scatter(nice_l0_matches[:, 0], nice_l0_matches[:, 1], s=2, color='orange')
    plt.scatter(supporters_l0[:, 0], supporters_l0[:, 1], s=2, color='cyan')

    figure.add_subplot(2, 1, 2)
    plt.title('Image 2 (left at t1) - {} in cyan'.format(supporters_or_inliers))
    plt.imshow(img2, cmap='gray')
    plt.scatter(nice_l1_matches[:, 0], nice_l1_matches[:, 1], s=2, color='orange')
    plt.scatter(supporters_l1[:, 0], supporters_l1[:, 1], s=2, color='cyan')
    plt.savefig("./ex3_figs/{}.png".format(supporters_or_inliers))
    plt.close()


if __name__ == '__main__':
    # get images matrices
    K, m1, m2 = utils.read_cameras()
    img1_m = K @ m1
    img2_m = K @ m2

    # Q 3.1
    # Analyse first stereo pair
    first_stereo = StereoFrame(0, img1_m, img2_m)
    # Analyse second stereo pair
    second_stereo = StereoFrame(1, img1_m, img2_m)

    # Q 3.2
    # find matches between two lefts and get the data for points that matched both across time (t0-t1) and between left and right
    multi_frame = MultiFrame(first_stereo, second_stereo, K, m2)
    multi_frame.match_and_filter_using_significance()

    # Q 3.3
    # sample randomly 4 points, and solve pnp
    multi_frame.solve_pnp(4)
    # Plot locations of four cameras
    plot_title = "Camera positions - top view"
    all_positions_left = get_positions_of_camera([multi_frame.get_current_transformation()], m2=None)
    all_positions_right = get_positions_of_camera([multi_frame.get_current_transformation()], m2=m2)
    plot_relative_position_of_cameras([all_positions_left, all_positions_right], title=plot_title, output_fig_name="test_locations_4_cameras")

    # Q 3.4
    multi_frame.find_supporters(multi_frame.get_current_transformation())

    # Plot on images left0 and left1 the matches, with supporters in different color.
    # plot_matches_and_supporters(img_left_t0, img_left_t1, images_over_time_data, supporters_l0, supporters_l1, "supporters")

    # Q 3.5 - ransac
    multi_frame.ransac()

    # refine the transformation using all supporters
    multi_frame.refine_transformation()
    print(multi_frame.get_best_transformation())

    # use refined T to plot the two 3D clouds from above, using coordinates of left-t1
    plot_3D_point_clouds(multi_frame)

    # plot inliers and outliers on left0 and left1 images
    multi_frame.find_supporters(multi_frame.get_best_transformation())
    # todo - fix this plot function
    # plot_matches_and_supporters(img_left_t0, img_left_t1, images_over_time_data, supporters_l0, supporters_l1, "inliers")


    # Q 3.6
    all_transformations = repeat_for_the_whole_movie()
    # plot the trajectory implied

    all_transformations = utils.read_data_from_pickle(transformations_of_movie_path)
    output_fig_name = "car_route"
    title = "Route of the car - left camera locations"
    all_positions = get_positions_of_camera(all_transformations)
    plot_relative_position_of_cameras([all_positions], title, output_fig_name)

    # plot the trajectory implied together wit GT

    # gt_path = r'..\dataset\poses\00.txt'
    # gt_transformations = db_utils.read_gt_transformations(gt_path)
    # gt_locations_in_l0_coords = np.array([-1 * Rt[:, :3].T @ Rt[:, 3:] for Rt in gt_transformations])

    output_fig_name = "car_route_and_gt"
    title = "Route of the car and GT - left camera locations"
    plot_relative_position_of_cameras([all_positions], title, output_fig_name)