import cv2
import numpy as np
class Visualizer:

    def visulize_orig(defects_dataloader_train):
        # the type of dataloader is [0] for original [i] for ith image(depending on batch size) 0 for channel
        # the last 0 is negligible, heer we need to merge the 3 channel original pic with the single channel label
        for i, img in enumerate(defects_dataloader_train):
            img_batch = img
            break
        n = 1
        for i in range(n):
    # raw image is  img_batch[0][i].permute(1, 2, 0) and mask is img_batch[1][i][0]
    #im1 = cv2.cvtColor(np.array(img_batch[1][i][0]), cv2.COLOR_RGB2BGR)
            im1 = img_batch[0][i][0]
            im2 = img_batch[1][i][0]
            hmerge = np.hstack((im1, im2))  # 水平拼接
            cv2.imshow('name',hmerge)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
