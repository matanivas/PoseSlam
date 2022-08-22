# get info for paths
for path in best_paths:
    print("Dist {} first frame {} last frame {}".format(path[0], pose_graph.get_global_frame(path[1][-1]), pose_graph.get_global_frame(path[1][0])))

# plot detections on images
def my_plot_accepted_and_rejected(frame_1_idx, frame_2_idx, img1_accepted_coors,
                               img2_accepted_coors):
    """
    plot on images the accepted and rejected coords
    """
    img1, _ = read_images(frame_1_idx)
    img2, _ = read_images(frame_2_idx)
    figure = plt.figure(figsize=(9, 6))
    figure.add_subplot(2, 1, 1)
    img1_accepted_coors = np.array(img1_accepted_coors)
    img2_accepted_coors = np.array(img2_accepted_coors)
    plt.title('Frame {} orange: inliers, cyan: outliers'.format(frame_1_idx))
    plt.imshow(img1, cmap='gray')
    plt.scatter(img1_accepted_coors[:, 0], img1_accepted_coors[:, 1], s=2, color='orange')

    figure.add_subplot(2, 1, 2)
    plt.title('Frame {} orange: inliers, cyan: outliers'.format(frame_2_idx))
    plt.imshow(img2, cmap='gray')
    plt.scatter(img2_accepted_coors[:, 0], img2_accepted_coors[:, 1], s=2, color='orange')
    plt.show()


my_plot_accepted_and_rejected(1575, 155, images_over_time_data['left_t0_kpts'], images_over_time_data['left_t1_kpts'])


for i in range(10, 30):
    pos_start_loop = pose_graph.result.atPose3(pose_graph.poses_symbols[i])
    translation_start_loop = np.array([pos_start_loop.x(), pos_start_loop.y(), pos_start_loop.z()])
    for j, pos in enumerate(pose_graph.poses_symbols[100:]):
        pos = pose_graph.result.atPose3(pos)
        translation = np.array([pos.x(), pos.y(), pos.z()])
        delta = np.linalg.norm([translation_start_loop - translation])
        if delta < 10:
            print("i {} j {} delta {} pos i {} pos j {}".format(i, j+100, delta, translation_start_loop, translation))




scores = []
for pos in all_closeness_scores.items():
    if len(pos[1]) > 0:
        scores.append(pos[1][0][0])



 #        Code for finding smallest manhalobis distances

all_closeness_scores = {}
for i, pose in enumerate(pose_graph.poses_symbols): # loop through all poses
    # create a list of the best paths
    print("Anlysing pose i {}".format(i))
    all_paths_of_possible_circle = []
    for j, prev_pos in enumerate(pose_graph.poses_symbols[:i]): # for a given pos, loop through all poses up to it.
        # get the shortest path between the two poses
        shortest_path = pose_graph.get_shortest_path(pose, prev_pos)
        # get the distance score between the two poses
        closeness_score = get_closeness(shortest_path, pose_graph)
        if closeness_score < LOOP_THRESHOLD:  # todo check if high score is close or the opposite.
            all_paths_of_possible_circle.append((closeness_score, shortest_path))

    all_closeness_scores[str(i)] = all_paths_of_possible_circle
    # find the best possible circles
    sorted_paths = sorted(all_paths_of_possible_circle, key=lambda tup: tup[0]) # todo if prev todo is changes - this should also change
    best_paths = sorted_paths[:min(MAX_NUM_PATHS_TO_CHECK_PER_POS, len(sorted_paths))]

    # Perform concensus matching over the best possible circles
    plot_consensus_match = False