## custom class for dataset #
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy
class custom_dataset(Dataset):
    """custom dataset"""

    def __init__(self, file_path, mask_dir):
        self.file_path = file_path
        self.mask_dir = mask_dir
        self.count = 0
        # alll pic show be this size
        self.x_in_size = 256
        self.y_in_size = 256

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, idx):
        # print(self.count)
        sample_out = [None] * 2

        # get file name with ext
        file_base_name = os.path.basename(self.file_path[self.count])

        # split file name and ext
        split_file_name = os.path.splitext(file_base_name)

        # get file name
        file_name = split_file_name[0]

        # mask path
        img_path_mask = self.mask_dir + file_name + '.png'

        img_raw_tem = cv2.imread(self.file_path[self.count])
        img_raw_tem = cv2.cvtColor(img_raw_tem, cv2.COLOR_BGR2RGB)
        img_raw_tem = cv2.resize(img_raw_tem, (self.x_in_size, self.y_in_size));
        img_raw_tem = numpy.rollaxis(img_raw_tem, 2, 0)

        img_mask_tem = cv2.imread(img_path_mask)
        img_mask_tem = cv2.resize(img_mask_tem, (self.x_in_size, self.y_in_size));
        img_mask_tem = cv2.cvtColor(img_mask_tem, cv2.COLOR_BGR2GRAY)
        img_mask_tem = numpy.expand_dims(img_mask_tem, axis=0)

        sample_out[0] = img_raw_tem
        sample_out[1] = img_mask_tem

        self.count += 1

        if self.count == len(self.file_path):
            self.count = 0

        return sample_out