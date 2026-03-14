import os
import sys
import argparse
import asyncio
import pandas as pd
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from evaluation script
from scripts.evaluate_mot import process_video, evaluate_mot, load_custom_gt
import motmetrics as mm

# Import newly created script logic
from scripts.compute_map import get_map

def evaluate_all(cameras, env, use_reid=False):
    print(f"\n{'='*50}")
    print(f"Starting Consolidated Evaluation for: {cameras}")
    print(f"Environment: {env}")
    print(f"Mode: {'Tracking + Re-ID' if use_reid else 'Tracking Only'}")
    print(f"{'='*50}\n")
    
    mh = mm.metrics.create()
    all_accs = []
    
    # Process each camera
    for cam in cameras:
        print(f"\n--- Processing Camera: {cam} ---")
        if env == 'factory':
            video_path = f"videos/{cam}.mp4"
        else:
            video_path = f"videos/campus/{cam}/sub_video_01.mp4"
            
        gt_path = f"videos/gt/{env}/gt_{cam}.txt"
        
        # Determine the correct model based on environment (Strictly TensorRT Engine)
        if env == 'factory':
            model_path = "weights/yolo26m_factory.engine"
        else:
            model_path = "weights/yolo26m_campus.engine"
            
        out_file = f"pred_eval_{cam}.txt"
        
        # Check if GT exists
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth not found at {gt_path}. Skipping.")
            continue
            
        print(f"Generating predictions for {cam}...")
        start_time = time.time()
        
        if use_reid:
            # Note: For full Reid script we'd call process_video_reid.
            # Using standard evaluation here per original request.
            # You can adapt similarly for evaluate_mot_reid if needed.
            from scripts.evaluate_mot_reid import process_video_reid
            asyncio.run(process_video_reid(video_path, model_path, cam, out_file, max_frames=None))
        else:
            # We call the basic process method
            process_video(video_path, model_path, out_file, max_frames=None)
            
        print(f"Detection/Tracking finished in {time.time() - start_time:.1f}s")
        
        print(f"Evaluating metrics for {cam}...")
        try:
            gt = load_custom_gt(gt_path)
            ts = mm.io.loadtxt(out_file, fmt='mot15-2D', min_confidence=0.5)
            
            # Align frames if necessary
            if not gt.empty:
                gt_start_frame = gt.index.get_level_values('FrameId').min()
                ts.reset_index(inplace=True)
                ts['FrameId'] = ts['FrameId'] + gt_start_frame - 1
                ts.set_index(['FrameId', 'Id'], inplace=True)
                
            # Removed artificial GT filtering to correctly penalize False Negatives
            
            acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
            
            # Custom mAP calculation
            ap = get_map(gt_path, out_file, iou_thresh=0.5)
            
            # Give it a name corresponding to the camera
            all_accs.append((cam, acc, ap))
        except Exception as e:
            print(f"Error evaluating {cam}: {e}")

    # Final Output Summary
    print(f"\n{'='*50}")
    print("FINAL CONSOLIDATED METRICS")
    print(f"{'='*50}")
    
    if not all_accs:
        print("No metrics generated.")
        return

    summary = mh.compute_many(
        [acc for name, acc, ap in all_accs], 
        metrics=['mota', 'idf1', 'num_switches', 'recall', 'precision'],
        names=[name for name, acc, ap in all_accs],
        generate_overall=False
    )
    
    # We don't need pandas to break on us. We'll do it by hand.
    headers = ["Camera", "MOTA", "IDF1", "ID Switches", "Recall", "Precision", "mAP@50"]
    
    rows = []
    for i, cam_name in enumerate(summary.index):
        row_series = summary.iloc[i]
        
        mota = row_series.get('mota', 0)
        idf1 = row_series.get('idf1', 0)
        sw = row_series.get('num_switches', 0)
        rec = row_series.get('recall', 0)
        prec = row_series.get('precision', 0)
        ap = all_accs[i][2] # Get exactly matching ap
        
        def sfmt(v, pct):
            if pd.isna(v): return "NaN"
            return f"{v*100:.1f}%" if pct else str(int(v))
            
        rows.append([
            f"{cam_name}.mp4",
            sfmt(mota, True),
            sfmt(idf1, True),
            sfmt(sw, False),
            sfmt(rec, True),
            sfmt(prec, True),
            sfmt(ap, True)
        ])
    
    final_df = pd.DataFrame(rows, columns=headers)
    print("\n" + final_df.to_string(index=False))
    
    # Save to file
    with open("final_metrics_table.md", "w") as f:
        f.write("# Consolidated Evaluation Metrics\n\n")
        f.write(final_df.to_markdown(index=False))
        
    print("Table saved to final_metrics_table.md")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate multiple MTMMC cameras and generate a single table.")
    parser.add_argument('--env', type=str, default='factory', choices=['factory', 'campus'], help='Environment to test (e.g. factory)')
    parser.add_argument('--cameras', nargs='+', default=['c09', 'c12', 'c13', 'c16'], help='Camera IDs to evaluate')
    parser.add_argument('--reid', action='store_true', help='Use Tracking + Re-ID (Ablation)')
    
    args = parser.parse_args()
    
    evaluate_all(args.cameras, args.env, args.reid)
