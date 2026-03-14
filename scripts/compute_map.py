import os
import sys
import argparse
import pandas as pd
import numpy as np

def compute_iou(box1, list_of_boxes):
    """
    box1: [x1, y1, x2, y2]
    list_of_boxes: Nx4 array of boxes [[x1, y1, x2, y2], ...]
    Returns Nx1 array of IoU scores
    """
    x1 = np.maximum(box1[0], list_of_boxes[:, 0])
    y1 = np.maximum(box1[1], list_of_boxes[:, 1])
    x2 = np.minimum(box1[2], list_of_boxes[:, 2])
    y2 = np.minimum(box1[3], list_of_boxes[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (list_of_boxes[:, 2] - list_of_boxes[:, 0]) * (list_of_boxes[:, 3] - list_of_boxes[:, 1])
    
    union = area1 + area2 - intersection
    return intersection / (union + 1e-6)

def voc_ap(rec, prec):
    """
    Calculate AP given precision and recall
    """
    # correct AP calculation
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    
    # compute precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
    # calculate area under PR curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def get_map(gt_path, pred_path, iou_thresh=0.5):
    # Load GT (Frame, ID, X, Y, W, H)
    gt = pd.read_csv(gt_path, header=None, names=['FrameId', 'Id', 'X', 'Y', 'W', 'H'])
    gt['X2'] = gt['X'] + gt['W']
    gt['Y2'] = gt['Y'] + gt['H']
    
    # Load Pred (Frame, ID, X, Y, W, H, Conf, _, _, _)
    pred = pd.read_csv(pred_path, header=None, names=['FrameId', 'Id', 'X', 'Y', 'W', 'H', 'Conf', 'a', 'b', 'c'])
    pred['X2'] = pred['X'] + pred['W']
    pred['Y2'] = pred['Y'] + pred['H']
    
    # Align the 1-indexed prediction frames to the global ground truth frames
    gt_start = gt['FrameId'].min()
    if gt_start > 1000:
        pred['FrameId'] = pred['FrameId'] + gt_start - 1
        
    # Removed artificial GT filtering to correctly penalize False Negatives
        
    # Prepare GT
    gt_boxes_by_frame = {}
    gt_frames = gt.groupby('FrameId')
    npos = 0
    for frame, group in gt_frames:
        boxes = group[['X', 'Y', 'X2', 'Y2']].values
        gt_boxes_by_frame[frame] = {'boxes': boxes, 'det': np.zeros(len(boxes))}
        npos += len(boxes)
        
    # Prepare Pred (Sort by confidence)
    pred = pred.sort_values(by='Conf', ascending=False).reset_index(drop=True)
    
    nd = len(pred)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    
    for d in range(nd):
        frame = pred.loc[d, 'FrameId']
        bb = pred.loc[d, ['X', 'Y', 'X2', 'Y2']].values.astype(float)
        
        if frame in gt_boxes_by_frame:
            R = gt_boxes_by_frame[frame]
            bbgt = R['boxes']
            
            if len(bbgt) > 0:
                overlaps = compute_iou(bb, bbgt)
                jmax = np.argmax(overlaps)
                ovmax = overlaps[jmax]
                
                if ovmax > iou_thresh:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.
            
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos) if npos > 0 else np.zeros_like(tp)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)
    
    return ap

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, required=True)
    parser.add_argument('--pred', type=str, required=True)
    args = parser.parse_args()
    ap = get_map(args.gt, args.pred)
    print(f"mAP@0.5: {ap * 100:.2f}%")
