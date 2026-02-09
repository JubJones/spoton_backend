import json
import math

# Load factory.json
with open('homography_data/factory.json', 'r') as f:
    data = json.load(f)

def get_homography(camera_id):
    for cam in data['cameras']:
        if cam['id'] == camera_id:
            return cam['homography']
    return None

def mat_mul_vec(matrix, vec):
    # matrix is 3x3 list of lists, vec is list of 3
    res = [0.0, 0.0, 0.0]
    for i in range(3):
        res[i] = (matrix[i][0] * vec[0] + 
                  matrix[i][1] * vec[1] + 
                  matrix[i][2] * vec[2])
    return res

def project_point(h_matrix, x, y):
    if not h_matrix: return None
    point = [x, y, 1.0]
    projected = mat_mul_vec(h_matrix, point)
    w = projected[2]
    if w == 0: return None
    return (projected[0] / w, projected[1] / w)

h_c09 = get_homography('c09')
h_c16 = get_homography('c16')

# Test Points
# c09: Far right edge (Person walking out)
p_c09_edge = (1850, 450)
p_c09_extreme = (1919, 500) # Extreme edge

# c16: Person on right side (P54/P63 in screenshot)
p_c16_right = (1600, 350)

print(f"--- Edge Projection Test (Pure Python) ---")

# Project c09
proj_c09 = project_point(h_c09, *p_c09_edge)
proj_c09_ext = project_point(h_c09, *p_c09_extreme)
if proj_c09: print(f"c09 {p_c09_edge} -> Map: ({proj_c09[0]:.2f}, {proj_c09[1]:.2f})")
if proj_c09_ext: print(f"c09 {p_c09_extreme} -> Map: ({proj_c09_ext[0]:.2f}, {proj_c09_ext[1]:.2f})")

# Project c16
proj_c16 = project_point(h_c16, *p_c16_right)
if proj_c16: print(f"c16 {p_c16_right} -> Map: ({proj_c16[0]:.2f}, {proj_c16[1]:.2f})")

# Distances
if proj_c09 and proj_c16:
    dist = math.sqrt((proj_c09[0] - proj_c16[0])**2 + (proj_c09[1] - proj_c16[1])**2)
    print(f"Distance (Edge vs Center): {dist:.2f} pixels")

if proj_c09_ext and proj_c16:
    dist_ext = math.sqrt((proj_c09_ext[0] - proj_c16[0])**2 + (proj_c09_ext[1] - proj_c16[1])**2)
    print(f"Distance (Extreme vs Center): {dist_ext:.2f} pixels")
