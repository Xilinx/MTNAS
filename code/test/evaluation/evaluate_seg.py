# Copyright 2021 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import json
import os
from collections import defaultdict

import os.path as osp

import numpy as np
from PIL import Image


def parse_args():
    """Use argparse to get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['seg', 'det', 'drivable'])
    parser.add_argument('gt', help='path to ground truth')
    parser.add_argument('result', help='path to results to be evaluated')
    args = parser.parse_args()

    return args


def fast_hist(gt, prediction, n):
    k = (gt >= 0) & (gt < n)
    return np.bincount(
        n * gt[k].astype(int) + prediction[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    ious = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    ious[np.isnan(ious)] = 0
    return ious


def find_all_png(folder):
    paths = []
    for root, dirs, files in os.walk(folder, topdown=True):
        paths.extend([osp.join(root, f)
                      for f in files if osp.splitext(f)[1] == '.png'])
    return paths


def evaluate_segmentation(gt_dir, result_dir, num_classes):
    gt_dict = dict([(osp.split(p)[1][:], p)
                    for p in find_all_png(gt_dir)])
    result_dict = dict([(osp.split(p)[1][:], p)
                        for p in find_all_png(result_dir)])
    result_gt_keys = set(gt_dict.keys()) & set(result_dict.keys())
    if len(result_gt_keys) != len(gt_dict):
        raise ValueError('Result folder only has {} of {} ground truth files.'
                         .format(len(result_gt_keys), len(gt_dict)))
    print('Found', len(result_dict), 'results')
    print('Evaluating', len(gt_dict), 'results')
    hist = np.zeros((num_classes, num_classes))
    i = 0
    gt_id_set = set()
    for key in sorted(gt_dict.keys()):
        gt_path = gt_dict[key]
        result_path = result_dict[key]
        gt = np.asarray(Image.open(gt_path, 'r'))
        gt_id_set.update(np.unique(gt).tolist())
        prediction = np.asanyarray(Image.open(result_path, 'r'))
        hist += fast_hist(gt.flatten(), prediction.flatten(), num_classes)
        i += 1
        if i % 100 == 0:
            print('Finished', i, per_class_iu(hist) * 100)
    gt_id_set.remove(255)
    print('GT id set', gt_id_set)
    ious = per_class_iu(hist) * 100
    miou = np.mean(ious[list(gt_id_set)])
    return miou, list(ious)


def evaluate_drivable(gt_dir, result_dir):
    return evaluate_segmentation(gt_dir, result_dir, 3, 17)


def get_ap(recalls, precisions):
    # correct AP calculation
    # first append sentinel values at the end
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))

    # compute the precision envelope
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(recalls[1:] != recalls[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
    return ap


def group_by_key(detections, key):
    groups = defaultdict(list)
    for d in detections:
        groups[d[key]].append(d)
    return groups


def cat_pc(gt, predictions, thresholds):
    """
    Implementation refers to https://github.com/rbgirshick/py-faster-rcnn
    """
    num_gts = len(gt)
    image_gts = group_by_key(gt, 'name')
    image_gt_boxes = {k: np.array([[float(z) for z in b['bbox']]
                                   for b in boxes])
                      for k, boxes in image_gts.items()}
    image_gt_checked = {k: np.zeros((len(boxes), len(thresholds)))
                        for k, boxes in image_gts.items()}
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)

    # go down dets and mark TPs and FPs
    nd = len(predictions)
    tp = np.zeros((nd, len(thresholds)))
    fp = np.zeros((nd, len(thresholds)))
    for i, p in enumerate(predictions):
        box = p['bbox']
        ovmax = -np.inf
        jmax = -1
        try:
            gt_boxes = image_gt_boxes[p['name']]
            gt_checked = image_gt_checked[p['name']]
        except KeyError:
            gt_boxes = []
            gt_checked = None

        if len(gt_boxes) > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(gt_boxes[:, 0], box[0])
            iymin = np.maximum(gt_boxes[:, 1], box[1])
            ixmax = np.minimum(gt_boxes[:, 2], box[2])
            iymax = np.minimum(gt_boxes[:, 3], box[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((box[2] - box[0] + 1.) * (box[3] - box[1] + 1.) +
                   (gt_boxes[:, 2] - gt_boxes[:, 0] + 1.) *
                   (gt_boxes[:, 3] - gt_boxes[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        for t, threshold in enumerate(thresholds):
            if ovmax > threshold:
                if gt_checked[jmax, t] == 0:
                    tp[i, t] = 1.
                    gt_checked[jmax, t] = 1
                else:
                    fp[i, t] = 1.
            else:
                fp[i, t] = 1.

    # compute precision recall
    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)
    recalls = tp / float(num_gts)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = np.zeros(len(thresholds))
    for t in range(len(thresholds)):
        ap[t] = get_ap(recalls[:, t], precisions[:, t])

    return recalls, precisions, ap


def evaluate_detection(gt_path, result_path):
    gt = json.load(open(gt_path, 'r'))
    pred = json.load(open(result_path, 'r'))
    cat_gt = group_by_key(gt, 'category')
    cat_pred = group_by_key(pred, 'category')
    cat_list = sorted(cat_gt.keys())
    thresholds = [0.5]
    aps = np.zeros((len(thresholds), len(cat_list)))
    for i, cat in enumerate(cat_list):
        if cat in cat_pred:
            r, p, ap = cat_pc(cat_gt[cat], cat_pred[cat], thresholds)
            aps[:, i] = ap
    aps *= 100
    mAP = np.mean(aps)
    return mAP, aps.flatten().tolist()


def main():
    args = parse_args()

    if args.task == 'drivable':
        mean, breakdown = evaluate_drivable(args.gt, args.result)
    elif args.task == 'seg':
        mean, breakdown = evaluate_segmentation(args.gt, args.result, 16)
    elif args.task == 'det':
        mean, breakdown = evaluate_detection(args.gt, args.result)

    print('{:.2f}'.format(mean),
          ', '.join(['{:.2f}'.format(n) for n in breakdown]))


if __name__ == '__main__':
    main()
