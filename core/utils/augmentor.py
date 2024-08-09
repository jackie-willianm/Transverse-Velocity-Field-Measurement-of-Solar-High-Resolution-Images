import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2
from torchvision.transforms import ColorJitter

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class FlowAugmentorMSY_A:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.2, contrast=0.4, saturation=0.4)  # 0.4 0.4 0.4
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)
            # img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint16)
            # img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint16)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            # image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint16)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def irregular_eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation with irregular shape """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)

            # 随机生成不规则区域的多边形顶点
            num_vertices = np.random.randint(3, 8)  # 生成 3 到 7 个顶点
            vertices = np.random.randint(0, min(ht, wd), size=(num_vertices, 2))

            # 创建掩码，表示多边形区域
            mask = np.zeros((ht, wd), dtype=np.uint8)
            cv2.fillPoly(mask, [vertices], 1)

            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)

                # 将多边形区域遮挡为均值颜色
                img2[y0:y0 + ht, x0:x0 + wd, :][mask[:, :, np.newaxis] > 0] = mean_color

        return img1, img2

    def irregular_expand_transform(self, img1, img2):
        ht, wd = img2.shape[:2]

        # 计算大圆形区域的半径（设置为图像的1/5）
        large_circle_radius = min(ht, wd) // 5

        # 随机选择大圆形区域的中心坐标，确保不会超出图像边界
        large_center_x = np.random.randint(large_circle_radius, wd - large_circle_radius)
        large_center_y = np.random.randint(large_circle_radius, ht - large_circle_radius)

        # 创建大圆形掩码
        large_mask = np.zeros((ht, wd), dtype=np.uint8)
        cv2.circle(large_mask, (large_center_x, large_center_y), large_circle_radius, 1, thickness=-1)

        # 计算小圆形区域的半径（设置为图像的1/6）
        small_circle_radius = min(ht, wd) // 6

        # 随机选择小圆形区域的中心坐标，确保在大圆形区域内
        small_center_x = np.random.randint(large_center_x - large_circle_radius + small_circle_radius,
                                           large_center_x + large_circle_radius - small_circle_radius)
        small_center_y = np.random.randint(large_center_y - large_circle_radius + small_circle_radius,
                                           large_center_y + large_circle_radius - small_circle_radius)

        # 创建小圆形掩码
        small_mask = np.zeros((ht, wd), dtype=np.uint8)
        cv2.circle(small_mask, (small_center_x, small_center_y), small_circle_radius, 1, thickness=-1)

        # 创建大圆形区域外部的掩码，用于选择大圆形区域中不包含小圆形区域的部分
        large_outer_mask = large_mask.copy()
        large_outer_mask[small_mask == 1] = 0

        # 复制小圆形区域特征到大圆形区域中不包含小圆形区域的部分
        img2_expanded = img2.copy()
        img2_expanded[large_outer_mask == 1] = img2[small_mask == 1]

        return img1, img2_expanded

    def spatial_transform(self, img1, img2, flow):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        # if np.random.rand() < self.stretch_prob:
        #     scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        #     scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # # 添加扩张增强
            # img1, img2 = self.irregular_expand_transform(img1, img2)
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.v_flip_prob:  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        # if(img1.shape[0] <= self.crop_size[0] or img1.shape[1] <= self.crop_size[1]):
        #     print("img1.shape[0]=", img1.shape[0], "\tself.crop_size[0]=", self.crop_size[0],
        #           "\nimg1.shape[1]=", img1.shape[1], "\tself.crop_size[1]=", self.crop_size[1])
        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        return img1, img2, flow

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return img1, img2

    def __call__(self, img1, img2, flow):
        img1, img2 = self.irregular_expand_transform(img1, img2)
        img1, img2, flow = self.spatial_transform(img1, img2, flow)
        img1_org, img2_org = img1, img2
        img1, img2 = self.color_transform(img1, img2)
        # img1, img2, img1_org, img2_org = self.eraser_transform(img1, img2)
        # img1, img2, flow, = self.spatial_transform(img1, img2, flow, img1_org, img2_org)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        img1_org = np.ascontiguousarray(img1_org)
        img2_org = np.ascontiguousarray(img2_org)

        return img1, img2, flow, img1_org, img2_org


class FlowAugmentorMSY_S:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.2, contrast=0.4, saturation=0.4)  # 0.4 0.4 0.4
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)
            # img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint16)
            # img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint16)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            # image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint16)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def irregular_eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation with irregular shape """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)

            # 随机生成不规则区域的多边形顶点
            num_vertices = np.random.randint(3, 8)  # 生成 3 到 7 个顶点
            vertices = np.random.randint(0, min(ht, wd), size=(num_vertices, 2))

            # 创建掩码，表示多边形区域
            mask = np.zeros((ht, wd), dtype=np.uint8)
            cv2.fillPoly(mask, [vertices], 1)

            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)

                # 将多边形区域遮挡为均值颜色
                img2[y0:y0 + ht, x0:x0 + wd, :][mask[:, :, np.newaxis] > 0] = mean_color

        return img1, img2

    def irregular_expand_transform(self, img1, img2, expansion_factor=1.5):
        ht, wd = img1.shape[:2]

        # 计算正方形区域的大小
        square_size = min(ht, wd) // 10
        expand_size = min(ht, wd) // 8

        # 随机选择正方形区域的左上角坐标
        x0 = np.random.randint(0, wd - square_size)
        y0 = np.random.randint(0, ht - square_size)

        # 创建掩码，表示正方形区域
        mask = np.zeros((ht, wd), dtype=np.uint8)
        mask[y0:y0 + square_size, x0:x0 + square_size] = 1

        # 计算扩张区域的大小
        expand_height = min(y0 + square_size + expand_size, ht) - y0
        expand_width = min(x0 + square_size + expand_size, wd) - x0

        # 对选择的区域进行缩放
        img1_expanded = img1.copy()
        # img1_expanded[y0:y0 + square_size, x0:x0 + square_size] = cv2.resize(
        #     img1[y0:y0 + square_size, x0:x0 + square_size], (expand_width, expand_height)
        # )

        img2_expanded = img2.copy()
        img2_expanded[y0:y0 + square_size, x0:x0 + square_size] = cv2.resize(
            img2[y0:y0 + square_size, x0:x0 + square_size], (square_size, square_size))

        return img1_expanded, img2_expanded

    def spatial_transform(self, img1, img2, flow):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        # if np.random.rand() < self.stretch_prob:
        #     scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        #     scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # # 添加扩张增强
            # img1, img2 = self.irregular_expand_transform(img1, img2)
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.v_flip_prob:  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        # if(img1.shape[0] <= self.crop_size[0] or img1.shape[1] <= self.crop_size[1]):
        #     print("img1.shape[0]=", img1.shape[0], "\tself.crop_size[0]=", self.crop_size[0],
        #           "\nimg1.shape[1]=", img1.shape[1], "\tself.crop_size[1]=", self.crop_size[1])
        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        return img1, img2, flow

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return img1, img2

    def __call__(self, img1, img2, flow):
        img1, img2 = self.irregular_expand_transform(img1, img2)
        img1, img2, flow = self.spatial_transform(img1, img2, flow)
        img1_org, img2_org = img1, img2
        img1, img2 = self.color_transform(img1, img2)
        # img1, img2, img1_org, img2_org = self.eraser_transform(img1, img2)
        # img1, img2, flow, = self.spatial_transform(img1, img2, flow, img1_org, img2_org)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        img1_org = np.ascontiguousarray(img1_org)
        img2_org = np.ascontiguousarray(img2_org)

        return img1, img2, flow, img1_org, img2_org


class FlowAugmentorCL:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.2, contrast=0.4, saturation=0.4)  # 0.4 0.4 0.4
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)
            # img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint16)
            # img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint16)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            # image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint16)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return img1, img2

    def crop_transform(self, crop_size, img1, img2):
        # y0 = np.random.randint(0, img1.shape[0] - crop_size[0])
        # x0 = np.random.randint(0, img1.shape[1] - crop_size[1])

        img1 = img1[0:0 + crop_size[0], 0:0 + crop_size[1]]
        img2 = img2[0:0 + crop_size[0], 0:0 + crop_size[1]]

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)

        return img1, img2

    def spatial_transform(self, img1, img2, flow):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        # if np.random.rand() < self.stretch_prob:
        #     scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        #     scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.v_flip_prob:  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        # if(img1.shape[0] <= self.crop_size[0] or img1.shape[1] <= self.crop_size[1]):
        #     print("img1.shape[0]=", img1.shape[0], "\tself.crop_size[0]=", self.crop_size[0],
        #           "\nimg1.shape[1]=", img1.shape[1], "\tself.crop_size[1]=", self.crop_size[1])
        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        return img1, img2, flow

    def __call__(self, img1, img2, flow):
        img1_org, img2_org = img1, img2
        image1_org_ar, image2_org_ar = self.crop_transform([384, 448], img1_org, img2_org)
        # img1, img2 = self.eraser_transform(img1, img2)
        # img1, img2, flow = self.spatial_transform(img1, img2, flow)
        # # img1_org, img2_org = img1, img2
        # img1, img2 = self.color_transform(img1, img2)
        # # img1, img2, img1_org, img2_org = self.eraser_transform(img1, img2)
        # # img1, img2, flow, = self.spatial_transform(img1, img2, flow, img1_org, img2_org)
        #
        # img1 = np.ascontiguousarray(img1)
        # img2 = np.ascontiguousarray(img2)
        # flow = np.ascontiguousarray(flow)

        image1_org_ar = np.ascontiguousarray(image1_org_ar)
        image2_org_ar = np.ascontiguousarray(image2_org_ar)

        return img1, img2, flow, image1_org_ar, image2_org_ar


class FlowAugmentorMSY:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.2, contrast=0.4, saturation=0.4)  # 0.4 0.4 0.4
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)
            # img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint16)
            # img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint16)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            # image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint16)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        # if np.random.rand() < self.stretch_prob:
        #     scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        #     scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.v_flip_prob:  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        # if(img1.shape[0] <= self.crop_size[0] or img1.shape[1] <= self.crop_size[1]):
        #     print("img1.shape[0]=", img1.shape[0], "\tself.crop_size[0]=", self.crop_size[0],
        #           "\nimg1.shape[1]=", img1.shape[1], "\tself.crop_size[1]=", self.crop_size[1])
        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        return img1, img2, flow

    def __call__(self, img1, img2, flow):
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow = self.spatial_transform(img1, img2, flow)
        img1_org, img2_org = img1, img2
        img1, img2 = self.color_transform(img1, img2)
        # img1, img2, img1_org, img2_org = self.eraser_transform(img1, img2)
        # img1, img2, flow, = self.spatial_transform(img1, img2, flow, img1_org, img2_org)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        img1_org = np.ascontiguousarray(img1_org)
        img2_org = np.ascontiguousarray(img2_org)

        return img1, img2, flow, img1_org, img2_org


class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.2, contrast=0.4, saturation=0.4)  # 0.4 0.4 0.4
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)
            # img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint16)
            # img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint16)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            # image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint16)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        # if np.random.rand() < self.stretch_prob:
        #     scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        #     scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.v_flip_prob:  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        # if(img1.shape[0] <= self.crop_size[0] or img1.shape[1] <= self.crop_size[1]):
        #     print("img1.shape[0]=", img1.shape[0], "\tself.crop_size[0]=", self.crop_size[0],
        #           "\nimg1.shape[1]=", img1.shape[1], "\tself.crop_size[1]=", self.crop_size[1])
        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        return img1, img2, flow

    def spatial_transform_no_flow(self, img1, img2):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        # if np.random.rand() < self.stretch_prob:
        #     scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        #     scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]

            if np.random.rand() < self.v_flip_prob:  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]

        # if(img1.shape[0] <= self.crop_size[0] or img1.shape[1] <= self.crop_size[1]):
        #     print("img1.shape[0]=", img1.shape[0], "\tself.crop_size[0]=", self.crop_size[0],
        #           "\nimg1.shape[1]=", img1.shape[1], "\tself.crop_size[1]=", self.crop_size[1])
        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]

        return img1, img2

    def __call__(self, img1, img2, flow):
        img1_org, img2_org = self.eraser_transform(img1, img2)
        img1_org, img2_org = self.spatial_transform_no_flow(img1_org, img2_org)
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow = self.spatial_transform(img1, img2, flow)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        img1_org = np.ascontiguousarray(img1_org)
        img2_org = np.ascontiguousarray(img2_org)

        return img1, img2, flow, img1_org, img2_org


class SparseFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3 / 3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0 + dy, x0:x0 + dx, :] = mean_color

        return img1, img2

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid >= 1]
        flow0 = flow[valid >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, img1, img2, flow, valid):
        # randomly sample scale

        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)

        if self.do_flip:
            if np.random.rand() < 0.5:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                valid = valid[:, ::-1]

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        img2 = img2[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        flow = flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        valid = valid[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]]
        return img1, img2, flow, valid

    def __call__(self, img1, img2, flow, valid):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow, valid = self.spatial_transform(img1, img2, flow, valid)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return img1, img2, flow, valid
