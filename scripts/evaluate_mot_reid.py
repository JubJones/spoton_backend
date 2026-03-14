import os
import sys
import argparse
import asyncio
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import motmetrics as mm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultralytics import YOLO
from boxmot.trackers.bytetrack.bytetrack import ByteTrack

def fix_yolo_serialization():
    from ultralytics.nn.modules import block
    mapping = {'PatchedC3k2': 'C3k2', 'PatchedSPPF': 'SPPF', 'PatchedConv': 'Conv', 'PatchedBottleneck': 'Bottleneck'}
    for mod_name in ['__main__', 'app.main']:
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
            for patched_name, real_name in mapping.items():
                if not hasattr(mod, patched_name) and hasattr(block, real_name):
                    setattr(mod, patched_name, getattr(block, real_name))

def load_custom_gt(filepath):
    df = pd.read_csv(filepath, header=None, names=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height'])
    df['Confidence'] = 1
    df['ClassId'] = 1
    df['Visibility'] = 1
    df.set_index(['FrameId', 'Id'], inplace=True)
    return df

async def process_video_reid(video_path, model_path, camera_id, output_path, max_frames=None):
    fix_yolo_serialization()
    print(f"Loading YOLO + Tracker + OSNet for {camera_id}...")
    
    detector = YOLO(model_path)
    tracker = ByteTrack()
    
    # Init ReID locally
    from app.services.feature_extraction_service import FeatureExtractionService
    from app.services.handoff_manager import HandoffManager
    fe_svc = FeatureExtractionService()
    handoff_mgr = HandoffManager()
    
    cap = cv2.VideoCapture(video_path)
    fh, fw = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames: total_frames = min(total_frames, max_frames)
    
    active_gids = {} # track_id -> global_id
    global_id_counter = 1000
    
    with open(output_path, 'w') as f:
        for frame_idx in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret: break
                
            results = detector(frame, verbose=False)
            np_dets = np.empty((0, 6), dtype=np.float32)
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                np_dets = np.hstack([boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()[:, None], boxes.cls.cpu().numpy()[:, None]]).astype(np.float32)
                np_dets = np_dets[np_dets[:, 5] == 0]
            if len(np_dets) == 0: np_dets = np.empty((0, 6), dtype=np.float32)
                
            tracked_objects = tracker.update(np_dets, frame)
            
            frame_number = frame_idx + 1
            for obj in tracked_objects:
                if len(obj) >= 5:
                    x1, y1, x2, y2, tid = obj[:5]
                    tid = int(tid)
                    
                    # ReID Logic: Assign Global ID on first sight
                    if tid not in active_gids:
                        x1c, y1c, x2c, y2c = max(0, int(x1)), max(0, int(y1)), min(fw, int(x2)), min(fh, int(y2))
                        if x2c > x1c + 10 and y2c > y1c + 10:
                            patch = frame[y1c:y2c, x1c:x2c]
                            embedding = await fe_svc.extract_async(patch)
                            if embedding is not None:
                                match_gid, score = handoff_mgr.find_match(embedding, camera_id)
                                if match_gid and score > 0.8:
                                    active_gids[tid] = match_gid
                                else:
                                    active_gids[tid] = f"{camera_id}_{global_id_counter}"
                                    global_id_counter += 1
                                    # Register to allow others to match it later
                                    handoff_mgr.register_exit(active_gids[tid], embedding, camera_id)
                        
                    gid = active_gids.get(tid, tid)
                    try:
                        out_id = int(gid)
                    except ValueError:
                        out_id = abs(hash(gid)) % 1000000
                        
                    w, h = x2 - x1, y2 - y1
                    f.write(f"{frame_number},{out_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,1,1,1\n")

    print(f"Saved Re-ID MOT predictions to {output_path}")

def evaluate_mot(gt_path, ts_path, align_frames=True):
    print(f"Evaluating {ts_path} against {gt_path}...")
    gt = load_custom_gt(gt_path)
    ts = mm.io.loadtxt(ts_path, fmt='mot15-2D', min_confidence=0.5)
    
    if align_frames and not gt.empty:
        gt_start_frame = gt.index.get_level_values('FrameId').min()
        ts.reset_index(inplace=True)
        ts['FrameId'] = ts['FrameId'] + gt_start_frame - 1
        ts.set_index(['FrameId', 'Id'], inplace=True)

    acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['mota', 'motp', 'num_switches', 'idf1', 'idp', 'idr', 'precision', 'recall'], name='acc')
    res_str = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
    print(res_str)
    
    with open('evaluation_results_reid.md', 'a') as f:
        f.write(f'\n## {ts_path}\n```\n{res_str}\n```\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--camera', type=str, required=True)
    parser.add_argument('--model', type=str, default='weights/yolo26m_factory.pt')
    parser.add_argument('--gt', type=str, required=True)
    parser.add_argument('--output', type=str, default='pred_reid.txt')
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()
    
    asyncio.run(process_video_reid(args.video, args.model, args.camera, args.output, args.max_frames))
    evaluate_mot(args.gt, args.output)
