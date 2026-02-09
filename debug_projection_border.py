import json
import numpy as np

# Load factoryConfig
with open('homography_data/factory.json', 'r') as f:
    data = json.load(f)

def get_homography(camera_id):
    for cam in data['cameras']:
        if cam['id'] == camera_id:
            return np.array(cam['homography'])
    return None

def project_point(h_matrix, x, y):
    point = np.array([x, y, 1.0])
    projected = h_matrix @ point
    w = projected[2]
    if w == 0: return None
    return (projected[0] / w, projected[1] / w)

h_c09 = get_homography('c09')
h_c16 = get_homography('c16')

# Estimated points from new images
# c09: Far right edge (Person at green table)
p_c09 = (1850, 480) 

# c16: Background/Mid (Person at desk indicated by arrow)
p_c16 = (960, 320)

print(f"--- Projection Test ---")
proj_c09 = project_point(h_c09, *p_c09)
proj_c16 = project_point(h_c16, *p_c16)

print(f"c09 {p_c09} -> Map: {proj_c09}")
print(f"c16 {p_c16} -> Map: {proj_c16}")

# Bounds Check
X_MIN, X_MAX = -2000, 2000
Y_MIN, Y_MAX = -2000, 2000

def check_bounds(p, name):
    if not p: return
    x, y = p
    in_bounds = (X_MIN <= x <= X_MAX) and (Y_MIN <= y <= Y_MAX)
    status = "OK" if in_bounds else "OUT OF BOUNDS"
    print(f"Bounds Check [{name}]: ({x:.2f}, {y:.2f}) -> {status}")

check_bounds(proj_c09, "c09")
check_bounds(proj_c16, "c16")

if proj_c09 and proj_c16:
    dist = np.sqrt((proj_c09[0] - proj_c16[0])**2 + (proj_c09[1] - proj_c16[1])**2)
    print(f"Distance: {dist:.2f} pixels")
