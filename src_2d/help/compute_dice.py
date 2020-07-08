"""
Compute Dice between test ground truth and predictions from groupwise registration.


"""
import os
import nibabel as nib
import glob
import numpy as np
from core import utils_2d
from core.metrics_2d import OverlapMetrics


def one_hot_label(label, label_intensity):
    gt = np.around(label)
    n_class = len(label_intensity)
    label = np.zeros((np.hstack((gt.shape, n_class))), dtype=np.float32)

    for k in range(1, n_class):
        label[..., k] = (gt == label_intensity[k])

    label[..., 0] = np.logical_not(np.sum(label[..., 1:], axis=-1))

    return label


def load_nifty(name):
    img = nib.load(name)
    return np.asarray(img.get_fdata(), np.float32)


if __name__ == '__main__':
    gt_path = '../../../../../../dataset/C0T2LGE/label_center_data/test/*label.nii.gz'
    pred_path = '../../../../../../results/MSCMR/test_predictions_1.5mm_group3_fusion15/*label.nii.gz'

    pred_names = utils_2d.strsort(glob.glob(pred_path))
    gt_names = utils_2d.strsort([name for name in glob.glob(gt_path) if os.path.basename(name).split('_')[1] == 'DE'])
    pred_gt_names = dict(zip(pred_names, gt_names))
    print(pred_gt_names)

    average_dice = []
    myo_dice = []
    LV_dice = []
    RV_dice = []
    for name in pred_names:

        pred_label = load_nifty(name)
        one_hot_pred = one_hot_label(pred_label, (0, 200, 500, 600))
        gt_label = load_nifty(pred_gt_names[name])
        gt_label = np.concatenate([gt for gt in np.dsplit(gt_label, gt_label.shape[-1])
                                   if np.all([np.sum(gt==i) > 0 for i in [200, 500, 600]])], axis=-1)
        one_hot_gt = one_hot_label(gt_label, (0, 200, 500, 600))

        Dice = OverlapMetrics(n_class=4, mode='np')

        dice = Dice.averaged_foreground_dice(one_hot_gt, one_hot_pred)
        m_dice = Dice.class_specific_dice(one_hot_gt, one_hot_pred, i=1)
        l_dice = Dice.class_specific_dice(one_hot_gt, one_hot_pred, i=2)
        r_dice = Dice.class_specific_dice(one_hot_gt, one_hot_pred, i=3)
        average_dice.append(dice)
        myo_dice.append(m_dice)
        LV_dice.append(l_dice)
        RV_dice.append(r_dice)
        print("Average foreground Dice for %s: %.4f" % (os.path.basename(name), dice))

        print("Myocardium Dice for %s: %.4f" % (os.path.basename(name), m_dice))

        print("LV Dice for %s: %.4f" % (os.path.basename(name), l_dice))

        print("RV Dice for %s: %.4f" % (os.path.basename(name), r_dice))

    print("Average prediction Dice: %.4f" % np.mean(average_dice))
    print("Average myocardium Dice: %.4f" % np.mean(myo_dice))
    print("Average LV Dice: %.4f" % np.mean(LV_dice))
    print("Average RV Dice: %.4f" % np.mean(RV_dice))


