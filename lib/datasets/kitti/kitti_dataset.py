import os
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageFile, ImageEnhance
import random
from skimage import io
import skimage.transform
import cv2
import torch.nn.functional as F
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True

import tqdm
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.dirname(ROOT_DIR)
ROOT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(ROOT_DIR)

from lib.datasets.kitti.pd import PhotometricDistort

from lib.datasets.utils import angle2class
from lib.datasets.utils import gaussian_radius
from lib.datasets.utils import draw_umich_gaussian
from lib.datasets.kitti.kitti_utils import get_objects_from_label
from lib.datasets.kitti.kitti_utils import Calibration
from lib.datasets.kitti.kitti_utils import get_affine_transform
from lib.datasets.kitti.kitti_utils import affine_transform
from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result
from lib.datasets.kitti.kitti_eval_python.eval import get_distance_eval_result
import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti
import copy

# from .pd import PhotometricDistort


class KITTI_Dataset(data.Dataset):
    def __init__(self, split, cfg, data_augmentation=True):

        # basic configuration
        self.root_dir = cfg.get("root_dir")
        self.split = split
        self.num_classes = 3
        self.max_objs = 50
        self.class_name = ["Pedestrian", "Car", "Cyclist"]
        self.cls2id = {"Pedestrian": 0, "Car": 1, "Cyclist": 2}
        self.resolution = np.array(cfg.get("resolution", [1280, 384]))  # W * H

        self.use_3d_center = cfg.get("use_3d_center", True)
        self.writelist = cfg.get("writelist", ["Car"])
        # anno: use src annotations as GT, proj: use projected 2d bboxes as GT
        self.bbox2d_type = cfg.get("bbox2d_type", "anno")
        assert self.bbox2d_type in ["anno", "proj"]
        self.meanshape = cfg.get("meanshape", False)
        self.class_merging = cfg.get("class_merging", False)
        self.use_dontcare = cfg.get("use_dontcare", False)

        self.maintain_image_ratio = cfg.get("maintain_image_ratio", True)
        self.target_focal_list = cfg.get("target_focal_list", [748.8391264865, 900, 1100, 1300])
        self.test_focal = cfg.get("test_focal", 748.8391264865)
        self.use_consistency_loss = cfg.get("use_consistency_loss", False)

        if self.class_merging:
            self.writelist.extend(["Van", "Truck"])
        if self.use_dontcare:
            self.writelist.extend(["DontCare"])

        # data split loading
        assert self.split in ["train", "val", "trainval", "test"]
        self.split_file = os.path.join(self.root_dir, "ImageSets", self.split + ".txt")

        self.idx_list = [x.strip() for x in open(self.split_file).readlines()]
        if "train" in self.split and cfg.get("use_all_samples", False):
            self.idx_list = self.idx_list * len(self.target_focal_list)
        # path configuration
        self.data_dir = os.path.join(self.root_dir, "testing" if split == "test" else "training")
        self.image_dir = os.path.join(self.data_dir, "image_2")
        # self.depth_dir = os.path.join(self.data_dir, 'depth_2')
        self.calib_dir = os.path.join(self.data_dir, "calib")
        self.label_dir = os.path.join(self.data_dir, "label_2")
        if cfg.get("use_filtered_labels_for_target_focal", False):
            self.label_dir = os.path.join(self.data_dir, f"label_2_{self.test_focal}")

        # data augmentation configuration
        # self.data_augmentation = True if split in ["train", "trainval"] else False
        self.data_augmentation = data_augmentation

        self.aug_pd = cfg.get("aug_pd", False)
        self.aug_crop = cfg.get("aug_crop", False)
        self.aug_calib = cfg.get("aug_calib", False)

        self.random_mixup3d = cfg.get("random_mixup3d", 0.5)
        self.random_flip = cfg.get("random_flip", 0.5)
        self.random_crop = cfg.get("random_crop", 0.5)
        self.scale = cfg.get("scale", 0.4)
        self.shift = cfg.get("shift", 0.1)

        self.depth_scale = cfg.get("depth_scale", "normal")

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.cls_mean_size = np.array(
            [
                [1.76255119, 0.66068622, 0.84422524],
                [1.52563191462, 1.62856739989, 3.88311640418],
                [1.73698127, 0.59706367, 1.76282397],
            ]
        )
        if not self.meanshape:
            self.cls_mean_size = np.zeros_like(self.cls_mean_size, dtype=np.float32)

        # others
        self.downsample = 32
        self.depth_downsample_factor = 16
        self.pd = PhotometricDistort()
        self.clip_2d = cfg.get("clip_2d", False)

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, "%06d.png" % idx)
        assert os.path.exists(img_file)
        return Image.open(img_file)  # (H, W, 3) RGB mode

    def get_depth_map(self, idx):
        """
        Loads depth map for a sample
        Args:
            idx [str]: Index of the sample
        Returns:
            depth [np.ndarray(H, W)]: Depth map
        """
        depth_file = os.path.join(self.depth_dir, "%06d.png" % idx)
        assert os.path.exists(depth_file)
        depth = io.imread(depth_file)
        depth = depth.astype(np.float32)
        depth /= 256.0
        # depth = Image.open(depth_file)
        return depth

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, "%06d.txt" % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, "%06d.txt" % idx)
        assert os.path.exists(calib_file)
        return Calibration(calib_file)

    def eval(self, results_dir, logger):
        logger.info("==> Loading detections and GTs...")
        img_ids = [int(id) for id in self.idx_list]
        dt_annos = kitti.get_label_annos(results_dir)
        gt_annos = kitti.get_label_annos(self.label_dir, img_ids)

        test_id = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}

        logger.info("==> Evaluating (official) ...")
        car_moderate = 0
        for category in self.writelist:
            results_str, results_dict, mAP3d_R40 = get_official_eval_result(gt_annos, dt_annos, test_id[category])
            if category == "Car":
                car_moderate = mAP3d_R40
            logger.info(results_str)
        return car_moderate

    def get_image_with_padding(self, img, target_size, pad_color=(0, 0, 0)):
        original_width, original_height = img.size
        target_width, target_height = target_size
        pad_right = target_width - original_width
        pad_bottom = target_height - original_height
        if pad_right < 0 or pad_bottom < 0:
            raise ValueError("目标尺寸必须大于等于原图尺寸")
        # 新建一张背景图，尺寸为 target_size，填充 pad_color
        new_img = Image.new(img.mode, (target_width, target_height), pad_color)
        # 将原图粘贴到新图的左上角，剩余区域即为 padding
        new_img.paste(img, (0, 0))

        return new_img

    def calculate_new_image_size(self, img_size, resolution):
        if resolution[0] / img_size[0] < resolution[1] / img_size[1]:
            return (
                np.array(
                    [
                        int(resolution[0] / img_size[0] * img_size[0]),
                        int(resolution[0] / img_size[0] * img_size[1]),
                    ]
                ),
                resolution[0] / img_size[0],
            )
        else:
            return (
                np.array(
                    [
                        int(resolution[1] / img_size[1] * img_size[0]),
                        int(resolution[1] / img_size[1] * img_size[1]),
                    ]
                ),
                resolution[1] / img_size[1],
            )

    def __len__(self):
        return self.idx_list.__len__()

    def _apply_focal_transform(self, img_pil, calib, objects, center, crop_scale,
                                img_size, new_img_size, scale_ratio,
                                random_flip_flag, random_mix_flag, random_index,
                                object_num, target_focal, index):
        """Applies focal-dependent affine transform to image and encodes labels."""
        calib = copy.deepcopy(calib)

        # compute affine transform for this focal
        if self.resolution[1] / img_size[1] > self.resolution[0] / img_size[0]:
            crop_ratio = calib.P2[0, 0] * self.resolution[0] / target_focal / img_size[0]
        else:
            crop_ratio = calib.P2[0, 0] * self.resolution[1] / target_focal / img_size[1]
        new_crop_scale = crop_scale - 1 + crop_ratio
        trans, trans_inv = get_affine_transform(center, new_crop_scale * img_size, 0, new_img_size, inv=1)

        # transform image
        img = img_pil.transform(
            tuple(new_img_size.tolist()),
            method=Image.AFFINE,
            data=tuple(trans_inv.reshape(-1).tolist()),
            resample=Image.BILINEAR,
        )
        img = self.get_image_with_padding(img, self.resolution)
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W

        # label encoding arrays
        calibs_arr = np.zeros((self.max_objs, 3, 4), dtype=np.float32)
        indices = np.zeros((self.max_objs), dtype=np.int64)
        mask_2d = np.zeros((self.max_objs), dtype=bool)
        labels = np.zeros((self.max_objs), dtype=np.int8)
        depth = np.zeros((self.max_objs, 1), dtype=np.float32)
        heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
        heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
        size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
        size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        boxes = np.zeros((self.max_objs, 4), dtype=np.float32)
        boxes_3d = np.zeros((self.max_objs, 6), dtype=np.float32)
        obj_region = np.zeros((img.shape[1], img.shape[2]), dtype=bool)

        def _encode_object(obj, slot_idx):
            """Encodes a single object into label arrays at slot_idx. Returns False if skipped."""
            bbox_2d = obj.box2d.copy()
            bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
            bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)

            center_2d = np.array(
                [(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2],
                dtype=np.float32,
            )

            ymin = int(max(bbox_2d[1], 0))
            ymax = int(min(bbox_2d[3], img.shape[1]))
            xmin = int(max(bbox_2d[0], 0))
            xmax = int(min(bbox_2d[2], img.shape[2]))
            obj_region[ymin:ymax, xmin:xmax] = 1

            corner_2d = bbox_2d.copy()

            center_3d = obj.pos + [0, -obj.h / 2, 0]
            center_3d = center_3d.reshape(-1, 3)
            center_3d, _ = calib.rect_to_img(center_3d)
            center_3d = center_3d[0]

            if random_flip_flag and not self.aug_calib:
                center_3d[0] = img_size[0] - center_3d[0]
            center_3d = affine_transform(center_3d.reshape(-1), trans)

            if center_3d[0] < 0 or center_3d[0] >= self.resolution[0]:
                return False
            if center_3d[1] < 0 or center_3d[1] >= self.resolution[1]:
                return False

            cls_id = self.cls2id[obj.cls_type]
            labels[slot_idx] = cls_id

            w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
            size_2d[slot_idx] = 1.0 * w, 1.0 * h

            center_2d_norm = center_2d / self.resolution
            size_2d_norm = size_2d[slot_idx] / self.resolution

            corner_2d_norm = corner_2d.copy()
            corner_2d_norm[0:2] = corner_2d[0:2] / self.resolution
            corner_2d_norm[2:4] = corner_2d[2:4] / self.resolution
            center_3d_norm = center_3d / self.resolution

            l = center_3d_norm[0] - corner_2d_norm[0]
            r = corner_2d_norm[2] - center_3d_norm[0]
            t = center_3d_norm[1] - corner_2d_norm[1]
            b = corner_2d_norm[3] - center_3d_norm[1]

            if l < 0 or r < 0 or t < 0 or b < 0:
                if self.clip_2d:
                    l = np.clip(l, 0, 1)
                    r = np.clip(r, 0, 1)
                    t = np.clip(t, 0, 1)
                    b = np.clip(b, 0, 1)
                else:
                    return False

            boxes[slot_idx] = (center_2d_norm[0], center_2d_norm[1], size_2d_norm[0], size_2d_norm[1])
            boxes_3d[slot_idx] = center_3d_norm[0], center_3d_norm[1], l, r, t, b

            if self.depth_scale == "normal":
                depth[slot_idx] = obj.pos[-1] * crop_scale
            elif self.depth_scale == "inverse":
                depth[slot_idx] = obj.pos[-1] / crop_scale
            elif self.depth_scale == "none":
                depth[slot_idx] = obj.pos[-1]

            heading_angle = calib.ry2alpha(obj.ry, (obj.box2d[0] + obj.box2d[2]) / 2)
            if heading_angle > np.pi:
                heading_angle -= 2 * np.pi
            if heading_angle < -np.pi:
                heading_angle += 2 * np.pi
            heading_bin[slot_idx], heading_res[slot_idx] = angle2class(heading_angle)

            src_size_3d[slot_idx] = np.array([obj.h, obj.w, obj.l], dtype=np.float32)
            mean_size = self.cls_mean_size[self.cls2id[obj.cls_type]]
            size_3d[slot_idx] = src_size_3d[slot_idx] - mean_size

            if obj.trucation <= 0.5 and obj.occlusion <= 2:
                mask_2d[slot_idx] = 1

            calibs_arr[slot_idx] = calib.P2
            return True

        # encode primary objects (flip already applied to objects before calling this method)
        for i in range(object_num):
            if objects[i].cls_type not in self.writelist:
                continue
            if objects[i].level_str == "UnKnown" or objects[i].pos[-1] < 2:
                continue
            if objects[i].pos[-1] > 65:
                continue
            _encode_object(objects[i], i)

        # encode mix objects
        if random_mix_flag:
            mix_objects = self.get_label(random_index)
            if random_flip_flag:
                for obj in mix_objects:
                    [x1, _, x2, _] = obj.box2d
                    obj.box2d[0], obj.box2d[2] = img_size[0] - x2, img_size[0] - x1
                    obj.ry = np.pi - obj.ry
                    obj.pos[0] *= -1
                    if obj.ry > np.pi:
                        obj.ry -= 2 * np.pi
                    if obj.ry < -np.pi:
                        obj.ry += 2 * np.pi

            object_num_temp = (
                len(mix_objects) if len(mix_objects) < (self.max_objs - object_num)
                else (self.max_objs - object_num)
            )
            for i in range(object_num_temp):
                if mix_objects[i].cls_type not in self.writelist:
                    continue
                if mix_objects[i].level_str == "UnKnown" or mix_objects[i].pos[-1] < 2:
                    continue
                _encode_object(mix_objects[i], i + object_num)

        # finalize calib and img_size
        if self.maintain_image_ratio:
            img_size_out = self.resolution
            calib.P2[0, 0] = calib.P2[0, 0] * scale_ratio
            if target_focal is not None:
                calib.P2[0, 0] = target_focal
        else:
            img_size_out = img_size

        targets = {
            "calibs": calibs_arr,
            "indices": indices,
            "img_size": img_size_out,
            "labels": labels,
            "boxes": boxes,
            "boxes_3d": boxes_3d,
            "depth": depth,
            "size_2d": size_2d,
            "size_3d": size_3d,
            "src_size_3d": src_size_3d,
            "heading_bin": heading_bin,
            "heading_res": heading_res,
            "mask_2d": mask_2d,
            "obj_region": obj_region,
            "target_focals": np.array([target_focal], dtype=np.float32),
        }

        info = {
            "img_id": index,
            "img_size": img_size_out,
        }
        if self.maintain_image_ratio:
            info["trans_inv"] = trans_inv
            info["img_size"] = self.resolution

        return img, calib.P2, targets, info

    def __getitem__(self, item):
        #  ============================   get inputs (focal-independent)   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        img = self.get_image(index)
        img_size = np.array(img.size)
        features_size = self.resolution // self.downsample  # W * H

        if self.split != "test":
            dst_W, dst_H = img_size

        center = np.array(img_size) / 2
        crop_size, crop_scale = img_size, 1
        random_flip_flag, random_crop_flag = False, False
        random_mix_flag = False
        calib = self.get_calib(index)

        if self.data_augmentation:

            if np.random.random() < self.random_mixup3d:
                random_mix_flag = True

            if self.aug_pd:
                img = np.array(img).astype(np.float32)
                img = self.pd(img).astype(np.uint8)
                img = Image.fromarray(img)

            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            if self.aug_crop:
                if np.random.random() < self.random_crop:
                    random_crop_flag = True
                    crop_scale = np.clip(
                        np.random.randn() * self.scale + 1,
                        1 - self.scale,
                        1 + self.scale,
                    )
                    crop_size = img_size * crop_scale
                    center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                    center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)

        random_index = None
        if random_mix_flag == True:
            count_num = 0
            random_mix_flag = False
            while count_num < 50:
                count_num += 1
                candidate_index = int(np.random.choice(self.idx_list))
                calib_temp = self.get_calib(candidate_index)

                if calib_temp.cu == calib.cu and calib_temp.cv == calib.cv and calib_temp.fu == calib.fu and calib_temp.fv == calib.fv:
                    img_temp = self.get_image(candidate_index)
                    img_size_temp = np.array(img_temp.size)
                    dst_W_temp, dst_H_temp = img_size_temp
                    if dst_W_temp == dst_W and dst_H_temp == dst_H:
                        objects_1 = self.get_label(index)
                        objects_2 = self.get_label(candidate_index)
                        if len(objects_1) + len(objects_2) < self.max_objs:
                            random_mix_flag = True
                            random_index = candidate_index
                            if random_flip_flag == True:
                                img_temp = img_temp.transpose(Image.FLIP_LEFT_RIGHT)
                            img_blend = Image.blend(img, img_temp, alpha=0.5)
                            img = img_blend
                            break

        new_img_size, scale_ratio = self.calculate_new_image_size(img_size, self.resolution)

        # test split: single focal, no label encoding
        if self.split == "test":
            self.target_focal = self.test_focal
            if self.resolution[1] / img_size[1] > self.resolution[0] / img_size[0]:
                crop_ratio = calib.P2[0, 0] * self.resolution[0] / self.test_focal / img_size[0]
            else:
                crop_ratio = calib.P2[0, 0] * self.resolution[1] / self.test_focal / img_size[1]
            new_crop_scale = crop_scale - 1 + crop_ratio
            trans, trans_inv = get_affine_transform(center, new_crop_scale * img_size, 0, new_img_size, inv=1)
            img = img.transform(
                tuple(new_img_size.tolist()),
                method=Image.AFFINE,
                data=tuple(trans_inv.reshape(-1).tolist()),
                resample=Image.BILINEAR,
            )
            img = self.get_image_with_padding(img, self.resolution)
            img = np.array(img).astype(np.float32) / 255.0
            img = (img - self.mean) / self.std
            img = img.transpose(2, 0, 1)
            calib_test = self.get_calib(index)
            calib_test.P2[0, 0] = self.test_focal
            targets = {"target_focals": np.array([self.test_focal], dtype=np.float32)}
            info = {"img_id": index, "img_size": self.resolution, "trans_inv": trans_inv}
            return img, calib_test.P2, targets, info

        #  ============================   get labels (focal-independent)   ==============================
        objects = self.get_label(index)
        calib = self.get_calib(index)
        object_num = len(objects) if len(objects) < self.max_objs else self.max_objs

        if random_flip_flag:
            if self.aug_calib:
                calib.flip(img_size)
            for object in objects:
                [x1, _, x2, _] = object.box2d
                object.box2d[0], object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                object.alpha = np.pi - object.alpha
                object.ry = np.pi - object.ry
                if self.aug_calib:
                    object.pos[0] *= -1
                if object.alpha > np.pi:
                    object.alpha -= 2 * np.pi
                if object.alpha < -np.pi:
                    object.alpha += 2 * np.pi
                if object.ry > np.pi:
                    object.ry -= 2 * np.pi
                if object.ry < -np.pi:
                    object.ry += 2 * np.pi

        #  ============================   focal-dependent transform   ==============================
        if self.use_consistency_loss and "train" in self.split:
            # consistency training: two different focal versions of the same image
            focal_A = random.choice(self.target_focal_list)
            remaining = [f for f in self.target_focal_list if f != focal_A]
            focal_B = random.choice(remaining)
            sample_A = self._apply_focal_transform(
                img, calib, objects, center, crop_scale, img_size,
                new_img_size, scale_ratio, random_flip_flag,
                random_mix_flag, random_index, object_num, focal_A, index,
            )
            sample_B = self._apply_focal_transform(
                img, calib, objects, center, crop_scale, img_size,
                new_img_size, scale_ratio, random_flip_flag,
                random_mix_flag, random_index, object_num, focal_B, index,
            )
            return sample_A, sample_B
        else:
            # single-focal path (val/test always, train when not use_consistency_loss)
            if "train" in self.split:
                target_focal = random.choice(self.target_focal_list)
            else:
                target_focal = self.test_focal
            self.target_focal = target_focal
            return self._apply_focal_transform(
                img, calib, objects, center, crop_scale, img_size,
                new_img_size, scale_ratio, random_flip_flag,
                random_mix_flag, random_index, object_num, target_focal, index,
            )
