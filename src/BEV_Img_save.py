import cv2
import numpy as np
from pathlib import Path
import os
from matplotlib import pyplot as plt

# 📁 Configuration
root = Path("/content/drive/MyDrive/Desseration_proj/BEV_Adas/Proj_dataset/images/").resolve()
output_dir = Path("/content/drive/MyDrive/Desseration_proj/BEV_Adas/src/bev_images")
output_dir.mkdir(parents=True, exist_ok=True)

# 🎯 Camera intrinsics
K = np.array([
    [659.9565405462982, 0, 634.6329612029243],
    [0, 625.1032520893773, 544.7433055928482],
    [0, 0, 1.0]
])
D = np.array([-0.2900269437421997, 0.11089496468175668, -0.0003222479159157141, 0.0029110573007121382])

def load_and_undistort(cam_name, frame_id):
    img_path = root / cam_name / f" ({frame_id}).jpg"
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    h, w = img.shape[:2]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0.5)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LANCZOS4)

def create_composite(frame_id):
    front = load_and_undistort("Front", frame_id)
    rear = load_and_undistort("Rear", frame_id)
    left = load_and_undistort("Left", frame_id)
    right = load_and_undistort("Right", frame_id)
    
    comp_width, comp_height = 1000, 1000
    composite = np.zeros((comp_height, comp_width, 3), dtype=np.uint8)
    car_w, car_h = 200, 400
    padding = 20

    def place_centered(img, y_pos=None, x_pos=None):
        h, w = img.shape[:2]
        if y_pos is not None:
            x = comp_width//2 - w//2
            composite[y_pos:y_pos+h, x:x+w] = img
        else:
            y = comp_height//2 - h//2
            composite[y:y+h, x_pos:x_pos+w] = img

    def get_scaled_size(img, max_w, max_h):
        h, w = img.shape[:2]
        scale = min(max_w/w, max_h/h)
        return (int(w*scale), int(h*scale))

    front_size = get_scaled_size(front, comp_width-2*padding, int(0.3*comp_height))
    front = cv2.resize(front, front_size, interpolation=cv2.INTER_LANCZOS4)
    place_centered(front, y_pos=padding)

    rear_size = get_scaled_size(rear, comp_width-2*padding, int(0.3*comp_height))
    rear = cv2.resize(rear, rear_size, interpolation=cv2.INTER_LANCZOS4)
    place_centered(rear, y_pos=comp_height-padding-rear_size[1])

    left_size = get_scaled_size(left, int(0.3*comp_width), comp_height-2*padding)
    left = cv2.resize(left, left_size, interpolation=cv2.INTER_LANCZOS4)
    place_centered(left, x_pos=padding)

    right_size = get_scaled_size(right, int(0.3*comp_width), comp_height-2*padding)
    right = cv2.resize(right, right_size, interpolation=cv2.INTER_LANCZOS4)
    place_centered(right, x_pos=comp_width-padding-right_size[0])

    car_x, car_y = comp_width//2-car_w//2, comp_height//2-car_h//2
    cv2.rectangle(composite, (car_x, car_y), (car_x+car_w, car_y+car_h), (0,255,255), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(composite, "FRONT", (comp_width//2-50, padding+30), font, 0.9, (255,255,255), 2)
    cv2.putText(composite, "REAR", (comp_width//2-40, comp_height-padding-10), font, 0.9, (255,255,255), 2)
    cv2.putText(composite, "LEFT", (padding+10, comp_height//2), font, 0.9, (255,255,255), 2)
    cv2.putText(composite, "RIGHT", (comp_width-padding-80, comp_height//2), font, 0.9, (255,255,255), 2)
    cv2.putText(composite, "CAR", (car_x+60, car_y+car_h//2+10), font, 1.2, (0,0,0), 3)

    return composite

# 🔁 Process frames 0 to 5
for frame_id in range(13):
    try:
        composite = create_composite(str(frame_id))
        output_path = output_dir / f"surround_view_{frame_id}.png"
        cv2.imwrite(str(output_path), composite, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Error processing frame {frame_id}: {e}")
