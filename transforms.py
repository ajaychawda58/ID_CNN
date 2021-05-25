import random
from PIL import Image
from collections.abc import Sequence
import torch
from torchvision.transforms import functional as F
import numpy as np
import cv2
def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data
def _get_corners(bboxes):
    width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
    height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)
    
    x1 = bboxes[:,0].reshape(-1,1)
    y1 = bboxes[:,1].reshape(-1,1)

    x2 = x1 + width
    y2 = y1 

    x3 = x1
    y3 = y1 + height

    x4 = bboxes[:,2].reshape(-1,1)
    y4 = bboxes[:,3].reshape(-1,1)

    corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))
    
    return corners

def _rotate_box(corners,angle,  cx, cy, h, w):
    
    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M,corners.T).T
 
    calculated = calculated.reshape(-1,8)
    
    return calculated

def _get_enclosing_box(corners):
  
    x_ = corners[:,[0,2,4,6]]
    y_ = corners[:,[1,3,5,7]]
    
    xmin = np.min(x_,1).reshape(-1,1)
    ymin = np.min(y_,1).reshape(-1,1)
    xmax = np.max(x_,1).reshape(-1,1)
    ymax = np.max(y_,1).reshape(-1,1)
    
    final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))
    
    return final
    

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[1:]
            image = F.hflip(image)
            bbox = target["boxes"]
            for i in range(len(bbox)):
                x_min = width - bbox[i][2]
                y_min = bbox[i][1]
                x_max = (width - bbox[i][0]) 
                y_max = bbox[i][3]
                bbox[i] = torch.tensor([x_min, y_min, x_max, y_max])
            target["boxes"] = bbox
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[1:]
            image = F.vflip(image)
            bbox = target["boxes"]
            for i in range(len(bbox)):
                x_min = bbox[i][0]
                y_min = height - bbox[i][3] 
                x_max = bbox[i][2] 
                y_max = height - bbox[i][1] 
                bbox[i] = torch.tensor([x_min, y_min, x_max, y_max])
            target["boxes"] = bbox
        return image, target

class Rotation(object):
    def __init__(self, prob):
        self.prob = prob
    
    
    def __call__(self, image, target):
        height, width = image.shape[1:]
        cx, cy = width//2, height//2
        if random.random() < self.prob:
            angle = random.randint(-45, 45)
            image = F.rotate(image, angle)
            bbox = target["boxes"]
            
            corners = _get_corners(bbox)
            corners = np.hstack((corners, bbox[:,4:]))
            corners[:,:8] = _rotate_box(corners[:,:8], angle, cx, cy, height, width)
            new_bbox = _get_enclosing_box(corners)
            target["boxes"] = torch.tensor(new_bbox)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class testTensor(object):
    def __call__(self, image):
        image = F.to_tensor(image)
        return image
    
class Resize(object):
    """Resize the input image to the given size.
    The image can be a PIL Image or a torch Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size).
            In torchscript mode padding as single int is not supported, use a tuple or
            list of length 1: ``[size, ]``.
        interpolation (int, optional): Desired interpolation enum defined by `filters`_.
            Default is ``PIL.Image.BILINEAR``. If input is Tensor, only ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR``
            and ``PIL.Image.BICUBIC`` are supported.
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
 
        image = F.resize(img, self.size, self.interpolation)
        bbox = target['boxes']
        w, h = img.size
        w1, h1 = self.size
        width_ratio = w1/h
        height_ratio = h1/w
        for i in range(len(bbox)):
            bbox[i][0] = bbox[i][0] * height_ratio
            bbox[i][1] = bbox[i][1] * width_ratio
            bbox[i][2] = bbox[i][2] * height_ratio
            bbox[i][3] = bbox[i][3] * width_ratio
            #print(bbox)
        target['boxes'] = bbox
        return image, target


    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)
