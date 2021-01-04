## predict some sample #
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import torch
import  glob
from unet import UNet
# start time
# test data path
def predict_result(test_image):
    start_time = time.time()
    # transform size
    # trochvision.transforms.Compose([
    #                transforms.Resize(48),
    #                transforms.ToTensor(),
    #                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
    x_in_size = 256
    y_in_size = 256
    # open test image
    # change bgr to rgb (cv2 read from)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    x_or = len(test_image)
    y_or = len(test_image[0])
    test_image = cv2.resize(test_image, (x_in_size, y_in_size))
    #(256, 256, 3)
    test_image = np.rollaxis(test_image, 2, 0)
    #(3, 256, 256)
    test_image = np.expand_dims(test_image, axis=0) # expend to 0
    #(1, 3, 256, 256)
    test_image_tensor = torch.from_numpy(test_image)

    # convert to float
    device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")
    test_image_tensor = test_image_tensor.to(device, dtype=torch.float)
## setup model


#model = UNet(in_channels=3, out_channels=1, init_features=32)

    pth_path = "./model_best_loss_blow_hole.pt"
    model = torch.load(pth_path)
    model = model.cuda()
    #checkpoint = torch.load(pth_path)
    #model.load_state_dict(checkpoint)


    # no grad for test
    with torch.no_grad():
    # eval model (evaluation mode)
        model.eval()

    # predict
    img_out = model(test_image_tensor)
    #torch.Size([1, 1, 256, 256])
    # get output in numpy
    img_out_numpy = img_out.cpu().detach().numpy()

    # get image in numpy
    img_out_ori = test_image_tensor[0].permute(1, 2, 0)
    img_out_ori_plot = np.uint8(img_out_ori.cpu().detach().numpy())

    elapsed_time = time.time() - start_time
    print('prediction time for 1 image = ' + str(elapsed_time) + ' seconds')
    org1 = cv2.resize(img_out_ori_plot,(y_or,x_or))
    org2 = cv2.resize(img_out_numpy[0][0],(y_or,x_or))
    #print(np.max(org2)), max org2 is 0.88
    plt.subplot(1,3,1)
    # show original image
    plt.imshow(org1)
    plt.title('input image')
    plt.subplot(1,3,2)
    # show detect area
    org22 = cv2.cvtColor(org2, cv2.COLOR_BGR2RGB)
    plt.imshow(org22)
    plt.title('predicted defect')
    plt.subplot(1, 3, 3)
    # show detect area
    plt.imshow(ground_image)
    plt.title('Ground Truth')
    plt.show()


test_image_dir = './datasets/blow_hole/img_test/'


for image_name in glob.glob(test_image_dir+"*.jpg"):
    test_image = cv2.imread(image_name)

    ground_image= cv2.imread(image_name.replace('img_test', 'img_mask').replace('jpg','png'))
    predict_result(test_image)
