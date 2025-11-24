import copy
import logging
import numpy as np
import torch
import cv2
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from sparseinst import SparseInstDatasetMapper

def build_gabor_kernels():
    filters = []
    ksize = 31
    # 4 orientations
    for theta in np.arange(0, np.pi, np.pi / 4):
        # sigma=4.0, lambda=10.0, gamma=0.5, psi=0
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters

def apply_gabor_filter(img, filters):
    # img is HxWxC (BGR) or HxW (Gray)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    accum = np.zeros_like(gray, dtype=np.float32)
    for kern in filters:
        fimg = cv2.filter2D(gray, cv2.CV_32F, kern)
        np.maximum(accum, fimg, accum)
        
    accum = np.clip(accum, 0, 255).astype(np.uint8)
    
    if len(img.shape) == 3:
        gabor_bgr = cv2.cvtColor(accum, cv2.COLOR_GRAY2BGR)
    else:
        gabor_bgr = accum
        
    # Weighted add: 0.8 * Original + 0.2 * Gabor
    enhanced = cv2.addWeighted(img, 0.8, gabor_bgr, 0.2, 0)
    return enhanced

class GaborDatasetMapper(SparseInstDatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
        self.gabor_filters = build_gabor_kernels()
        logger = logging.getLogger(__name__)
        logger.info("GaborDatasetMapper initialized with Gabor filtering enabled.")

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # Apply Gabor Filter
        image = apply_gabor_filter(image, self.gabor_filters)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)

        if self.crop_aug is None:
            transforms = self.default_aug(aug_input)
        else:
            if np.random.rand() > 0.5:
                transforms = self.crop_aug(aug_input)
            else:
                transforms = self.default_aug(aug_input)
        # transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                anno.pop("keypoints", None)
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict


