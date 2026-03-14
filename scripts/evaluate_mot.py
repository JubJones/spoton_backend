import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import asyncio
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import motmetrics as mm

from ultralytics import YOLO
from boxmot.trackers.bytetrack.bytetrack import ByteTrack

def load_custom_gt(filepath):
    df = pd.read_csv(filepath, header=None, names=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height'])
    df['Confidence'] = 1
    df['ClassId'] = 1
    df['Visibility'] = 1
    df.set_index(['FrameId', 'Id'], inplace=True)
    return df

def fix_yolo_serialization():
    import sys
    from ultralytics.nn.modules import block
    mapping = {
        'PatchedC3k2': 'C3k2',
        'PatchedSPPF': 'SPPF',
        'PatchedConv': 'Conv',
        'PatchedBottleneck': 'Bottleneck',
    }
    for mod_name in ['__main__', 'app.main']:
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
            for patched_name, real_name in mapping.items():
                if not hasattr(mod, patched_name) and hasattr(block, real_name):
                    setattr(mod, patched_name, getattr(block, real_name))

def process_video(video_path, model_path, output_path, max_frames=None):
    print(f"Loading YOLO ({model_path}) and Tracker...")
    fix_yolo_serialization()
    detector = YOLO(model_path)
    
    tracker = ByteTrack()
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    print(f"Processing {total_frames} frames from {video_path}...")
    
    with open(output_path, 'w') as f:
        for frame_idx in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect
            results = detector(frame, verbose=False)
            
            np_dets = np.empty((0, 6), dtype=np.float32)
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                # Format for tracker: [x1, y1, x2, y2, conf, cls]
                np_dets = np.hstack([boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()[:, None], boxes.cls.cpu().numpy()[:, None]]).astype(np.float32)
                
                # Filter only persons (class 0)
                np_dets = np_dets[np_dets[:, 5] == 0]
            
            if len(np_dets) == 0:
                np_dets = np.empty((0, 6), dtype=np.float32)
                
            # Track
            tracked_objects = tracker.update(np_dets, frame)
            
            # MTMMC GT frames might be global, but usually per-camera files start at some frame.
            # We'll just write 1-indexed frames. We may need to align them later if the GT is global.
            # Actually, MTMMC GT for factory starts at global frame e.g. 3682. We need to check GT for the start frame!
            mot_frame = frame_idx + 1
            for obj in tracked_objects:
                if len(obj) >= 5:
                    x1, y1, x2, y2, track_id = obj[:5]
                    w = x2 - x1
                    h = y2 - y1
                    f.write(f"{mot_frame},{int(track_id)},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,1,1,1\n")

    cap.release()
    print(f"Saved MOT predictions to {output_path}")

def evaluate_mot(gt_path, ts_path, align_frames=True):
    print(f"Evaluating {ts_path} against {gt_path}...")
    gt = load_custom_gt(gt_path)
    ts = mm.io.loadtxt(ts_path, fmt='mot15-2D', min_confidence=0.5)
    
    # If video frames were 1-indexed but GT frames start arbitrarily (e.g. 3682), align them.
    if align_frames and not gt.empty:
        gt_start_frame = gt.index.get_level_values('FrameId').min()
        print(f"GT starts at frame offset {gt_start_frame}. Aligning TS frames (assuming TS starts at 1)...")
        # Reset index to modify FrameId
        ts.reset_index(inplace=True)
        ts['FrameId'] = ts['FrameId'] + gt_start_frame - 1
        ts.set_index(['FrameId', 'Id'], inplace=True)

    acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
    
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['mota', 'motp', 'num_switches', 'idf1', 'idp', 'idr', 'precision', 'recall'], name='acc')
    res_str = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
    print(res_str)
    with open('evaluation_results.md', 'a') as f:
        f.write(f'\n## {ts_path}\n```\n{res_str}\n```\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--gt', type=str, required=True, help='Path to ground truth txt file')
    parser.add_argument('--model', type=str, default='weights/yolo26m_factory.engine', help='Path to YOLO weights')
    parser.add_argument('--output', type=str, default='pred.txt', help='Output predictions file')
    parser.add_argument('--max-frames', type=int, default=None, help='Max frames to process (for debugging)')
    args = parser.ArgumentParser().parse_args() if False else parser.parse_args()
    
    process_video(args.video, args.model, args.output, args.max_frames)
    evaluate_mot(args.gt, args.output)
