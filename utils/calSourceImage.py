import numpy as np


def calsource(img1, img2):

    # 输入的res_img需要是二维单通道图像，数据类型是np.array
    # 返回残差均值res_mean和残差方差res_variance
    img1 = img1.squeeze(0)
    img1 = img1.cpu().numpy()
    img1 = np.mean(img1, axis=0)

    img2 = img2.squeeze(0)
    img2 = img2.cpu().numpy()
    img2 = np.mean(img2, axis=0)
    res_img = abs(img1 - img2).astype(np.int32)
    # Solar0513
    if res_img.shape[-2] == 552:
        res_img = res_img[6:546, 5:595]
    # Scale5Original0513
    if (res_img.shape[-2] == 424):
        res_img = res_img[7:417, 5:745]

    if (res_img.shape[-2] == 544):
        res_img = res_img[7:537, 6:586]

    if (res_img.shape[-2] == 448):
        res_img = res_img[5:443, 5:1019]
    # print("res_img.shape = ", res_img.shape)
    # 残差均值--此处comp_dif中已经将残差矩阵绝对值化了
    res_mean = res_img.mean()
    # 残差方差
    res_variance = res_img.var()
    return res_mean, res_variance