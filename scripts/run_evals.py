import sys
import subprocess
import time
import os

sequences = ['c09', 'c12', 'c13', 'c16']
for seq in sequences:
    print(f"================ Evaluating {seq} ================")
    cmd = [sys.executable, 'scripts/evaluate_mot.py', 
           '--video', f'videos/{seq}.mp4', 
           '--gt', f'videos/gt/factory/gt_{seq}.txt', 
           '--model', 'weights/yolo26m_factory.pt', 
           '--output', f'pred_{seq}.txt']
    start = time.time()
    # Also ensures we overwrite any existing pred file
    if os.path.exists(f'pred_{seq}.txt'):
        os.remove(f'pred_{seq}.txt')
    subprocess.run(cmd)
    print(f"Time for {seq}: {time.time() - start:.1f}s\n")
