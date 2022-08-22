import os
import ex2
import matplotlib.pyplot as plt
import numpy as np
import cv2
from VAN_project.code.utils import utils
from VAN_project.code.utils.paths import GT_PATH, images_path, all_tracks_path, all_frames_path, inliers_path

from VAN_project.code.utils.rts_utils import project
from VAN_project.code.utils.db_utils import read_gt_transformations, get_coordinates_of_track_id_in_frame, from_tracks_dict_to_frames_dict

PXL_AROUND_FEATURE = 50


def plot_track_on_images(all_tracks, output_path):
    # find a track with length >= 10
    track = utils.sample_track_longer_than_n(10, all_tracks)

    # plot points on the relevant image
    for frame in range(track.first_frame, track.last_frame + 1):
        img1, img2 = utils.read_images(frame)
        max_height = img1.shape[0]
        max_width = img1.shape[1]

        cur_coordinates = track.get_coords_in_frame(frame)

        x1 = int(cur_coordinates[0])
        x2 = int(cur_coordinates[1])
        y = int(cur_coordinates[2])

        sub_img_1 = img1[max(y - PXL_AROUND_FEATURE, 0): min(y + PXL_AROUND_FEATURE, max_height),
                         max(x1 - PXL_AROUND_FEATURE, 0): min(x1 + PXL_AROUND_FEATURE, max_width)]

        sub_img_2 = img2[max(y - PXL_AROUND_FEATURE, 0): min(y + PXL_AROUND_FEATURE, max_height),
                         max(x2 - PXL_AROUND_FEATURE, 0): min(x2 + PXL_AROUND_FEATURE, max_width)]

        final_image1 = cv2.circle(sub_img_1, (PXL_AROUND_FEATURE, PXL_AROUND_FEATURE), radius=1, color=(0, 0, 255), thickness=-1)
        final_image2 = cv2.circle(sub_img_2, (PXL_AROUND_FEATURE, PXL_AROUND_FEATURE), radius=1, color=(0, 0, 255), thickness=-1)

        # draw the images side by side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(final_image1)
        axarr[1].imshow(final_image2)
        f.suptitle("Frame {}".format(frame))
        plt.savefig('{}_{}.png'.format(output_path, frame))
        plt.close()


def calc_mean_frame_links(all_tracks_dict, all_frames_dict):
    total_number_of_times_a_track_passes_through_any_frame = 0
    for key in all_frames_dict:
        number_of_tracks_in_frame = 0
        for track_id in all_frames_dict[key]:
            # track_id_frames = trackid_to_frames[str(track_id)]
            track = all_tracks_dict[str(track_id)]
            if track.length > 1:
                number_of_tracks_in_frame += 1
        total_number_of_times_a_track_passes_through_any_frame += number_of_tracks_in_frame
    print("Mean number of frames links: {}".format(total_number_of_times_a_track_passes_through_any_frame/len(all_frames_dict.keys())))


def calc_track_stats(track_id_to_frames):
    min_track_length = 9999
    max_track_length = 0
    sum_tracks_lengths = 0
    for key in track_id_to_frames:
        track_len = track_id_to_frames[key].length
        min_track_length = min(min_track_length, track_len)
        max_track_length = max(max_track_length, track_len)
        sum_tracks_lengths += track_len
    mean_track_len = sum_tracks_lengths / len(track_id_to_frames.keys())
    print("Mean track length: {} Max track length: {} Min track length: {}".format(mean_track_len, max_track_length, min_track_length))
    return max_track_length


def create_connectivity_graph(all_frames, output_path):
    num_of_frames = len(all_frames.keys())
    frames_conectivity = [0] * num_of_frames
    tracks_on_prev_frame = all_frames["0"]
    for i in range(1, num_of_frames):
        tracks_on_cur_frame = all_frames[str(i)]
        count = 0
        for track in tracks_on_prev_frame:
            if track in tracks_on_cur_frame:
                count += 1
        frames_conectivity[i-1] = count
        tracks_on_prev_frame = tracks_on_cur_frame
    plt.plot(range(num_of_frames), frames_conectivity)
    plt.xlabel('Frame')
    plt.ylabel('Outgoing tracks')
    plt.title('Connectivity')
    plt.savefig('{}/Connectivity.png'.format(output_path))
    plt.close()


def remove_element_from_frame_to_trackids(frame_to_trackid, frame, trackid):
    """
    given trackid and frame - remove corresponding elements from the frame_to_trackid datastructure.
    todo - this is expensive. maybe run once, and save.
    :param frame_to_trackid:
    :param frame:
    :param trackid:
    :return:
    """
    # find index corresponding to the trackid that should be removed (in the frame given)
    index_of_the_track_id = np.where(frame_to_trackid[str(frame)]["track_ids"] == int(trackid))

    # remove the track_id and the corresponding left and right coordinates
    frame_to_trackid[str(frame)]["left_kpts"] = np.delete(frame_to_trackid[str(frame)]["left_kpts"], index_of_the_track_id, 0)
    frame_to_trackid[str(frame)]["right_kpts"] = np.delete(frame_to_trackid[str(frame)]["right_kpts"], index_of_the_track_id, 0)
    frame_to_trackid[str(frame)]["track_ids"] = np.delete(frame_to_trackid[str(frame)]["track_ids"], index_of_the_track_id, 0)


def remove_tracks_of_length_one(trackid_to_frames, frame_to_trackids):
    # turn the list of "track_ids" per frame (in frame_to_trackids) into ndarray
    for frame in frame_to_trackids:
        frame_to_trackids[frame]["track_ids"] = np.asarray(frame_to_trackids[frame]["track_ids"])

    # remove "tracks" of length "1" from the trackid_to_frames_dist
    print("Num track id's before removal:{}".format(len(trackid_to_frames.keys())))
    track_ids_to_remove = []
    for track_id, frames in trackid_to_frames.items():
        if frames[1] == frames[0]:
            track_ids_to_remove.append((track_id, frames[0]))
    print("num track ids to remove {}".format(len(track_ids_to_remove)))

    # track_ids_to_remove = []
    # for frame in frame_to_trackids_dict:
    #     for track_id in frame_to_trackids_dict[frame]["track_ids"]:
    #         if track_id not in trackid_to_frames_dict:
    #             track_ids_to_remove.append((track_id, frame))
    #

    for track_to_remove, frame in track_ids_to_remove:
        del trackid_to_frames[track_to_remove]
        remove_element_from_frame_to_trackids(frame_to_trackids, frame, track_to_remove)
    print("Num track id's after removal:{}".format(len(trackid_to_frames.keys())))


def calc_inliers_percentage(num_of_frames, inliers_dictionary, output_path):
    """
    calc and plot the inliers percentage per frame
    :param num_of_frames:
    :param frame_to_trackids:
    :return:
    """
    percentages = [0] * num_of_frames
    for frame in range(1, num_of_frames):
        percentages[frame] = inliers_dictionary[str(frame)]
    plt.plot(range(num_of_frames), percentages)
    plt.xlabel('Frame')
    plt.ylabel('Inliers percentage')
    plt.title('Inliers percentage vs frame')
    plt.savefig('{}/Inliers_percentage_vs_frame.png'.format(output_path))
    plt.close()


def plot_tracks_length_hist(max_length_track, all_tracks, output_path):
    """
    plot the histogram
    :param max_length_track:
    :param trackid_to_frames:
    :return:
    """
    hist = [0] * (max_length_track + 1)
    for track_id, track in all_tracks.items():
        hist[track.length] += 1

    plt.plot(range(max_length_track + 1), hist)
    plt.xlabel('Track length')
    plt.ylabel('Tracks #')
    plt.title('Tracks length histogram')
    plt.savefig('{}/Tracks_length_histogram.png'.format(output_path))
    plt.close()

    # Plot the same plot with focus
    plt.plot(range(max_length_track + 1), hist)
    plt.ylim((0, 200))
    plt.xlim((10, 100))
    plt.xlabel('Track length')
    plt.ylabel('Tracks #')
    plt.title('Tracks length histogram - Zoomed')
    plt.savefig('{}/Tracks_length_histogram_zoom_right.png'.format(output_path))
    plt.close()

    # Plot the same plot with focus
    plt.plot(range(max_length_track + 1), hist)
    # plt.ylim((0, 200))
    plt.xlim((3, 10))
    plt.xlabel('Track length')
    plt.ylabel('Tracks #')
    plt.title('Tracks length histogram - Zoomed')
    plt.savefig('{}/Tracks_length_histogram_zoom_left.png'.format(output_path))
    plt.close()


def get_projection_dist_of_track(track, rts, K, m1, m2, max_frame_id):
    """
    Given track and the relative Rts that were computed by the pnp - return the projection distance of a track
    as a function of the number of frames from the frame of the triangulation
    """
    # the frames corresponding to the track
    first_frame_on_track_id = track.first_frame
    last_frame_on_track_id = min(track.last_frame, max_frame_id - 1)

    track_frames_ids = [i for i in range(first_frame_on_track_id, last_frame_on_track_id + 1)]
    trans = utils.from_relative_to_global_transformtions(rts[first_frame_on_track_id + 1: last_frame_on_track_id + 1])

    # get the coordinates in each image of the track
    coords = [track.get_coords_in_frame(i) for i in track_frames_ids]
    left_coors = [(coords[i][0], coords[i][2]) for i in range(len(coords))]
    right_coors = [(coords[i][1], coords[i][2]) for i in range(len(coords))]

    # get the last coordinate
    first_left_coor = left_coors[0]
    first_right_coor = right_coors[0]
    first_left_transformation = trans[0]

    last_left_proj_mat = K @ first_left_transformation
    last_right_proj_mat = K @ utils.two_transformations_composition(first_left_transformation, m2)


    global_point_in_3d_homogen = ex2.triangulate(last_left_proj_mat, last_right_proj_mat, [first_left_coor], [first_right_coor])

    left_image_pxls = []
    right_image_pxls = []

    for trans in trans:
        left_proj_matrix = K @ trans
        right_proj_matrix = K @ utils.two_transformations_composition(trans, m2)

        left_proj = project(global_point_in_3d_homogen, left_proj_matrix)[:2]
        right_proj = project(global_point_in_3d_homogen, right_proj_matrix)[:2]

        left_image_pxls.append(left_proj)
        right_image_pxls.append(right_proj)

    left_projections = np.array(left_image_pxls)
    right_projections = np.array(right_image_pxls)

    # calc projection dist
    left_proj_dist = np.linalg.norm(left_projections - left_coors, axis=1)
    right_proj_dist = np.linalg.norm(right_projections - right_coors, axis=1)
    dists = (left_proj_dist + right_proj_dist) / 2

    return dists


def plot_proj_dists(all_tracks, output_path):

    K, m1, m2 = utils.read_cameras()

    track = utils.sample_track_longer_than_n(10, all_tracks, rand=True)
    # the frames corresponding to the track
    first_frame_on_track_id = track.first_frame
    last_frame_on_track_id = track.last_frame

    track_frames_ids = [i for i in range(first_frame_on_track_id, last_frame_on_track_id + 1)]
    ground_truth_trans = read_gt_transformations(GT_PATH)[first_frame_on_track_id: last_frame_on_track_id + 1]

    # get the coordinates in each image of the track
    coords = [track.get_coords_in_frame(i) for i in track_frames_ids]
    left_coors = [(coords[i][0], coords[i][2]) for i in range(len(coords))]
    right_coors = [(coords[i][1], coords[i][2]) for i in range(len(coords))]

    # get the last coordinate
    last_left_coor = left_coors[-1]
    last_right_coor = right_coors[-1]
    last_left_transformation = ground_truth_trans[-1]

    last_left_proj_mat = K @ last_left_transformation
    last_right_proj_mat = K @ utils.two_transformations_composition(last_left_transformation, m2)


    global_point_in_3d_homogen = ex2.triangulate(last_left_proj_mat, last_right_proj_mat, [last_left_coor], [last_right_coor])

    left_image_pxls = []
    right_image_pxls = []

    for trans in ground_truth_trans:
        left_proj_matrix = K @ trans
        right_proj_matrix = K @ utils.two_transformations_composition(trans, m2)

        left_proj = project(global_point_in_3d_homogen, left_proj_matrix)[0]
        right_proj = project(global_point_in_3d_homogen, right_proj_matrix)[0]

        left_image_pxls.append(left_proj)
        right_image_pxls.append(right_proj)

    left_projections = np.array(left_image_pxls)
    right_projections = np.array(right_image_pxls)

    # calc projection dist
    left_proj_dist = np.linalg.norm(left_projections - left_coors, axis=1)
    right_proj_dist = np.linalg.norm(right_projections - right_coors, axis=1)
    dists = (left_proj_dist + right_proj_dist) / 2

    # plot the projection distance over time
    plt.plot(track_frames_ids, dists)
    plt.xlabel('frame id')
    plt.ylabel('projection distance')
    plt.title('projection distances')
    plt.savefig('{}/projection_distances.png'.format(output_path))
    plt.close()


if __name__ == "__main__":
    output_path = "./temp/"

    ### Question 2 ###
    num_images = len([name for name in os.listdir(images_path)])
    cleaned_all_tracks = utils.read_data_from_pickle(all_tracks_path)
    cleaned_frames = utils.read_data_from_pickle(all_frames_path)


    # total number of tracks:
    number_of_tracks = len(cleaned_all_tracks.keys())
    print("The total number of tracks is: {}".format(number_of_tracks))
    number_of_frames = len(cleaned_frames.keys())
    print("The total number of frames is: {}".format(number_of_frames))

    # Calc track lengths statistics
    max_length_of_track = calc_track_stats(cleaned_all_tracks)

    # Calc mean number of frame links (number of tracks on an average image)
    calc_mean_frame_links(cleaned_all_tracks, cleaned_frames)

    ### Question 3 ###
    plot_track_on_images(cleaned_all_tracks, output_path)

    ### Question 4 ###
    #  connectivity graph
    create_connectivity_graph(cleaned_frames, output_path)


    ###  Question 5 Inliers percentage ###
    # NOTICE THIS ASSUMES THE PREV DATA STRUCTURE OF FRAMES
    inliers_dict = utils.read_data_from_pickle(inliers_path)
    calc_inliers_percentage(number_of_frames, inliers_dict, output_path)

    ###  Question 6 Tracks length histogram ###
    plot_tracks_length_hist(max_length_of_track, cleaned_all_tracks, output_path)

    ### Question 7 - calc projection distances over the track ###
    # plot_proj_dists(cleaned_all_tracks, output_path)

