
import numpy as np
import torch


def imgs_dif(img1, img2):
    # 此处进行简单的处理，img2此时是torch.size([1,3,H,W]=[1,3,552,600]
    # 将图像从tensor变为ndarray数据类型，然后降维成灰度图像（单通道图像）
    # 传入的img1和img2都是float类型
    img1 = img1.squeeze(0)
    img1 = img1.cpu().numpy()
    img1 = np.mean(img1, axis=0)
    img2 = img2.squeeze(0)
    img2 = img2.cpu().numpy()
    img2 = np.mean(img2, axis=0)
    # img2 = (255 * img2).astype(np.int32)
    # img2 = img2.astype(np.int32)
    # 测试img2（np.int32） - img1(np.int32)
    # img1 = img1.astype(np.int32)
    # img2 = img2.astype(np.int32)
    img_dif = abs(img2 - img1)
    img_dif = img_dif.astype(np.int32)  # 此处需要转换成int32类型，否则在调用cv2.imshow时，会先除以256，再映射到[0-255]

    return img_dif