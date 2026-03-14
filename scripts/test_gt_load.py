import motmetrics as mm
import pandas as pd

def load_custom_gt(filepath):
    df = pd.read_csv(filepath, header=None, names=['FrameId', 'Id', 'X', 'Y', 'Width', 'Height'])
    df['Confidence'] = 1
    df['ClassId'] = 1
    df['Visibility'] = 1
    df.set_index(['FrameId', 'Id'], inplace=True)
    return df

try:
    # Load ground truth
    print("Loading GT...")
    gt = load_custom_gt('gt_half-val.txt')
    print(gt.head())
except Exception as e:
    print(f"Failed to load: {e}")
