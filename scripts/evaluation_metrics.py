from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from skimage.filters import threshold_otsu

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


#We define the anomaly map as the absolute difference between the original image and the generated healthy reconstruction. We sum up over all channels (4 different MR sequences):

difftot=abs(original-healthyreconstruction).sum(dim=0)

#We compute the Otsu threshold for the anomaly map:

diff = np.array(difftot)
thresh = threshold_otsu(diff)
mask = th.where(th.tensor(diff) > thresh, 1, 0)  #this is our predicted binary segmentation

#We load the ground truth segmetation mask and put all the different tumor labels to 1:

Labelmask_GT = th.where(groundtruth_segmentation > 0, 1, 0)

pixel_wise_cls = visualize(np.array(th.tensor(diff).view(1, -1))[0, :])
pixel_wise_gt = visualize(np.array(th.tensor(Labelmask_GT).view(1, -1))[0, :])

#Then we compute the Dice and AUROC scores

DSC=dice_score(mask.cpu(), Labelmask_GT.cpu()) #predicted Dice score
auc = roc_auc_score(pixel_wise_gt, pixel_wise_cls)
