
def my_plot_track_on_images(all_tracks, track_id):
    # find a track with length >= 10
    # plot points on the relevant image
    track = all_tracks[str(track_id)]
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
        plt.savefig('./temp/track_{}_frame_{}.png'.format(str(track_id), str(frame)))
        plt.close()


#for debugging:
# bad_factors = []
# for factor in factors:
#     if factor[0].error(initialEstimate) > 50:
#         bad_factors.append(factor)

