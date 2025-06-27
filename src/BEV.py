import cv2
import numpy as np
from pathlib import Path

# Configuration
root = Path(r"D:/Mtech/Disseration/BEV_Adas/Dataset/images0").resolve()
rgb_dir = root / "train/rgb"
frame_id = "0"

# Camera intrinsics (unchanged for quality)
K = np.array([
    [659.9565405462982, 0, 634.6329612029243],
    [0, 625.1032520893773, 544.7433055928482],
    [0, 0, 1.0]
])
D = np.array([-0.2900269437421997, 0.11089496468175668, -0.0003222479159157141, 0.0029110573007121382])

def load_and_undistort(cam_name):
    """Load with maximum quality processing"""
    img_path = rgb_dir / cam_name / f"{frame_id}.png"
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    h, w = img.shape[:2]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0.5)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LANCZOS4)  # Best quality interpolation

def create_composite():
    # Load all images with full quality
    front = load_and_undistort("front")
    rear = load_and_undistort("rear")
    left = load_and_undistort("left")
    right = load_and_undistort("right")
    
    # Original full-quality composite dimensions
    comp_width = 1000
    comp_height = 1000
    composite = np.zeros((comp_height, comp_width, 3), dtype=np.uint8)
    
    # Placement parameters (unchanged from original quality version)
    car_w, car_h = 200, 400
    padding = 20
    
    # Placement functions with Lanczos resizing
    def place_centered(img, y_pos=None, x_pos=None):
        h, w = img.shape[:2]
        if y_pos is not None:  # Horizontal center
            x = comp_width//2 - w//2
            composite[y_pos:y_pos+h, x:x+w] = img
        else:  # Vertical center
            y = comp_height//2 - h//2
            composite[y:y+h, x_pos:x_pos+w] = img
    
    # Calculate scaled sizes maintaining aspect ratio
    def get_scaled_size(img, max_w, max_h):
        h, w = img.shape[:2]
        scale = min(max_w/w, max_h/h)
        return (int(w*scale), int(h*scale))
    
    # Front (top)
    front_size = get_scaled_size(front, comp_width-2*padding, int(0.3*comp_height))
    front = cv2.resize(front, front_size, interpolation=cv2.INTER_LANCZOS4)
    place_centered(front, y_pos=padding)
    
    # Rear (bottom)
    rear_size = get_scaled_size(rear, comp_width-2*padding, int(0.3*comp_height))
    rear = cv2.resize(rear, rear_size, interpolation=cv2.INTER_LANCZOS4)
    place_centered(rear, y_pos=comp_height-padding-rear_size[1])
    
    # Left (left side)
    left_size = get_scaled_size(left, int(0.3*comp_width), comp_height-2*padding)
    left = cv2.resize(left, left_size, interpolation=cv2.INTER_LANCZOS4)
    place_centered(left, x_pos=padding)
    
    # Right (right side)
    right_size = get_scaled_size(right, int(0.3*comp_width), comp_height-2*padding)
    right = cv2.resize(right, right_size, interpolation=cv2.INTER_LANCZOS4)
    place_centered(right, x_pos=comp_width-padding-right_size[0])
    
    # Draw car (center)
    car_x, car_y = comp_width//2-car_w//2, comp_height//2-car_h//2
    cv2.rectangle(composite, (car_x, car_y), (car_x+car_w, car_y+car_h), (0,255,255), -1)
    
    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(composite, "FRONT", (comp_width//2-50, padding+30), font, 0.9, (255,255,255), 2)
    cv2.putText(composite, "REAR", (comp_width//2-40, comp_height-padding-10), font, 0.9, (255,255,255), 2)
    cv2.putText(composite, "LEFT", (padding+10, comp_height//2), font, 0.9, (255,255,255), 2)
    cv2.putText(composite, "RIGHT", (comp_width-padding-80, comp_height//2), font, 0.9, (255,255,255), 2)
    cv2.putText(composite, "CAR", (car_x+60, car_y+car_h//2+10), font, 1.2, (0,0,0), 3)
    
    return composite

# Main processing
try:
    # Create full-quality composite
    composite = create_composite()
    
    # Save full-quality output first
    output_path = str(root / "surround_view_full_quality.png")
    cv2.imwrite(output_path, composite, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"Full quality composite saved as {output_path}")
    
    # Create smaller display window (600x600) without resizing the image data
    cv2.namedWindow("Surround View (600x600)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Surround View (600x600)", 600, 600)
    
    # Show the original quality image in the smaller window
    # (OpenCV will handle the display scaling without affecting the source image)
    cv2.imshow("Surround View (600x600)", composite)
    cv2.waitKey(0)
    
except Exception as e:
    print(f"Error: {e}")
finally:
    cv2.destroyAllWindows()