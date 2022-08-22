import cv2
import matplotlib.pyplot as plt
import random
from VAN_project.code.utils import utils
RATIO = 0.4


def match_and_remove_outliers_using_significance(first_dscs, second_dscs, k, ratio):
    passed = []
    failed = []
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(first_dscs, second_dscs, k)
    for m1, m2 in matches:
        if m1.distance < ratio * m2.distance:
            passed.append([m1])
        else:
            failed.append([m1])
    return passed, failed


if __name__ == '__main__':
    # parts 1 & 2
    img1, img2 = utils.read_images(0)

    alg = cv2.AKAZE_create()

    kpts1, dscs1 = alg.detectAndCompute(img1, None)
    kpts2, dscs2 = alg.detectAndCompute(img2, None)
    print("Image 1 first descriptor:\n", dscs1[0])
    print("Image 2 first descriptor:\n", dscs2[0])

    img1_with_kpts = cv2.drawKeypoints(img1, kpts1, None, color=(0, 0, 255), flags=0)
    img2_with_kpts = cv2.drawKeypoints(img2, kpts2, None, color=(0, 0, 255), flags=0)
    plt.title('img1 keypoints')
    plt.imshow(img1_with_kpts)
    plt.savefig('img1 keypoints')
    plt.show()


    plt.title('img2 keypoints')
    plt.imshow(img2_with_kpts)
    plt.savefig('img2 keypoints')
    plt.show()





    # part 3
    f = lambda x: x.distance
    matches = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True).match(dscs1, dscs2)
    matches = sorted(matches, key=f)

    selected_matches = random.choices(matches, k=20)

    matches_drawn = cv2.drawMatches(img1, kpts1, img2, kpts2, selected_matches, None, flags=2)
    plt.title('matches')
    plt.imshow(matches_drawn)
    plt.savefig('matches')
    plt.show()

    # part 4
    passed, failed = match_and_remove_outliers_using_significance(dscs1, dscs2, k=2, ratio=RATIO)

    total = len(passed) + len(failed)
    selected_from_passed_lst = random.choices(passed, k=20)
    selected_from_failed_lst = random.choices(failed, k=20)

    passed_matches_drawn = cv2.drawMatchesKnn(img1, kpts1, img2, kpts2, selected_from_passed_lst, None, flags=2)
    failed_matches_drawn = cv2.drawMatchesKnn(img1, kpts1, img2, kpts2, selected_from_failed_lst, None, flags=2)

    plt.title(f'Matches that Passed the significance test\n RATIO = 0.4\n {len(passed)} of {total} passed')
    plt.imshow(passed_matches_drawn)
    plt.savefig('passed matches')

    plt.show()

    plt.title(f'Matches that Failed the significance test\n RATIO = 0.4\n {len(failed)} of {total} failed')
    plt.imshow(failed_matches_drawn)
    plt.savefig('failed matches')
    plt.show()



