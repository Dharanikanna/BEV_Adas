import cv2
import numpy as np
from pathlib import Path

# Configuration
root = Path(r"D:/Mtech/Disseration/BEV_Adas/Dataset/images0").resolve()
rgb_dir = root / "train/rgb"
frame_id = "0"

# Camera intrinsics
K = np.array([
    [659.9565405462982, 0, 634.6329612029243],
    [0, 625.1032520893773, 544.7433055928482],
    [0, 0, 1.0]
])
D = np.array([-0.2900269437421997, 0.11089496468175668, -0.0003222479159157141, 0.0029110573007121382])

# Bird's eye view parameters
BEV_WIDTH = 800
BEV_HEIGHT = 800

def load_and_undistort(cam_name):
    """Load with maximum quality processing"""
    img_path = rgb_dir / cam_name / f"{frame_id}.png"
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    h, w = img.shape[:2]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0.5)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LANCZOS4)

def debug_camera_views():
    """Debug function to visualize original camera views"""
    cameras = ["front", "rear", "left", "right"]
    
    for cam_name in cameras:
        try:
            img = load_and_undistort(cam_name)
            print(f"{cam_name} camera - Shape: {img.shape}")
            
            # Save individual views for inspection
            debug_path = str(root / f"debug_{cam_name}.png")
            cv2.imwrite(debug_path, img)
            print(f"Saved debug image: {debug_path}")
            
            # Show the image
            cv2.namedWindow(f"{cam_name} Camera", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"{cam_name} Camera", 600, 400)
            cv2.imshow(f"{cam_name} Camera", img)
            
        except Exception as e:
            print(f"Error loading {cam_name}: {e}")
    
    print("Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_adaptive_homography_matrix(img, camera_position):
    """
    Get homography matrix with adaptive sizing based on actual image dimensions
    """
    h, w = img.shape[:2]
    
    if camera_position == "front":
        # For front camera, assume road/ground plane is in lower portion
        src_pts = np.float32([
            [w * 0.15, h * 0.85],   # Bottom left
            [w * 0.85, h * 0.85],   # Bottom right  
            [w * 0.75, h * 0.45],   # Top right
            [w * 0.25, h * 0.45]    # Top left
        ])
        # Map to front area of BEV
        dst_pts = np.float32([
            [BEV_WIDTH//2 - 100, BEV_HEIGHT//2 - 50],   # Bottom left
            [BEV_WIDTH//2 + 100, BEV_HEIGHT//2 - 50],   # Bottom right
            [BEV_WIDTH//2 + 80, BEV_HEIGHT//2 - 200],   # Top right
            [BEV_WIDTH//2 - 80, BEV_HEIGHT//2 - 200]    # Top left
        ])
        
    elif camera_position == "rear":
        src_pts = np.float32([
            [w * 0.25, h * 0.45],   # Top left
            [w * 0.75, h * 0.45],   # Top right
            [w * 0.85, h * 0.85],   # Bottom right
            [w * 0.15, h * 0.85]    # Bottom left
        ])
        dst_pts = np.float32([
            [BEV_WIDTH//2 - 80, BEV_HEIGHT//2 + 200],   # Top left
            [BEV_WIDTH//2 + 80, BEV_HEIGHT//2 + 200],   # Top right
            [BEV_WIDTH//2 + 100, BEV_HEIGHT//2 + 50],   # Bottom right
            [BEV_WIDTH//2 - 100, BEV_HEIGHT//2 + 50]    # Bottom left
        ])
        
    elif camera_position == "left":
        src_pts = np.float32([
            [w * 0.15, h * 0.85],   # Bottom left
            [w * 0.25, h * 0.45],   # Top left
            [w * 0.75, h * 0.45],   # Top right
            [w * 0.85, h * 0.85]    # Bottom right
        ])
        dst_pts = np.float32([
            [BEV_WIDTH//2 - 200, BEV_HEIGHT//2 + 80],   # Bottom left
            [BEV_WIDTH//2 - 200, BEV_HEIGHT//2 - 80],   # Top left
            [BEV_WIDTH//2 - 50, BEV_HEIGHT//2 - 80],    # Top right
            [BEV_WIDTH//2 - 50, BEV_HEIGHT//2 + 80]     # Bottom right
        ])
        
    elif camera_position == "right":
        src_pts = np.float32([
            [w * 0.85, h * 0.85],   # Bottom right
            [w * 0.75, h * 0.45],   # Top right
            [w * 0.25, h * 0.45],   # Top left
            [w * 0.15, h * 0.85]    # Bottom left
        ])
        dst_pts = np.float32([
            [BEV_WIDTH//2 + 50, BEV_HEIGHT//2 + 80],    # Bottom right
            [BEV_WIDTH//2 + 50, BEV_HEIGHT//2 - 80],    # Top right
            [BEV_WIDTH//2 + 200, BEV_HEIGHT//2 - 80],   # Top left
            [BEV_WIDTH//2 + 200, BEV_HEIGHT//2 + 80]    # Bottom left
        ])
    
    return cv2.getPerspectiveTransform(src_pts, dst_pts), src_pts, dst_pts

def create_debug_bev():
    """Create BEV with debug information"""
    
    # Initialize BEV canvas
    bev_canvas = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)
    
    # Load and process each camera view
    cameras = ["front", "rear", "left", "right"]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Debug colors
    
    for i, cam_name in enumerate(cameras):
        try:
            print(f"Processing {cam_name} camera...")
            
            # Load and undistort image
            img = load_and_undistort(cam_name)
            print(f"{cam_name} image shape: {img.shape}")
            
            # Get homography matrix
            H, src_pts, dst_pts = get_adaptive_homography_matrix(img, cam_name)
            
            # Apply perspective transformation
            warped = cv2.warpPerspective(img, H, (BEV_WIDTH, BEV_HEIGHT), 
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(0, 0, 0))
            
            # Create mask for this camera's region
            mask = np.zeros((BEV_HEIGHT, BEV_WIDTH), dtype=np.uint8)
            
            # Fill the destination region
            dst_pts_int = dst_pts.astype(np.int32)
            cv2.fillPoly(mask, [dst_pts_int], 255)
            
            # Apply mask to warped image
            warped_masked = cv2.bitwise_and(warped, warped, mask=mask)
            
            # Add to canvas with some transparency
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            alpha = 0.3
            bev_canvas = cv2.addWeighted(bev_canvas, 1.0, warped_masked, alpha, 0)
            
            # Draw region boundary for debugging
            cv2.polylines(bev_canvas, [dst_pts_int], True, colors[i], 2)
            
            # Add label
            center_x = int(np.mean(dst_pts[:, 0]))
            center_y = int(np.mean(dst_pts[:, 1]))
            cv2.putText(bev_canvas, cam_name.upper(), (center_x-30, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
            
            print(f"Successfully processed {cam_name}")
            
        except Exception as e:
            print(f"Error processing {cam_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return bev_canvas

def create_simple_bev():
    """Create a simple BEV without complex blending"""
    
    # Initialize BEV canvas
    bev_canvas = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)
    
    cameras = ["front", "rear", "left", "right"]
    
    for cam_name in cameras:
        try:
            print(f"Processing {cam_name} camera...")
            
            # Load and undistort image
            img = load_and_undistort(cam_name)
            
            # Get homography matrix
            H, src_pts, dst_pts = get_adaptive_homography_matrix(img, cam_name)
            
            # Apply perspective transformation
            warped = cv2.warpPerspective(img, H, (BEV_WIDTH, BEV_HEIGHT), 
                                       flags=cv2.INTER_LINEAR)
            
            # Simple region-based placement
            if cam_name == "front":
                # Place in upper region
                region_mask = np.zeros((BEV_HEIGHT, BEV_WIDTH), dtype=np.uint8)
                region_mask[0:BEV_HEIGHT//2, :] = 255
                
            elif cam_name == "rear":
                # Place in lower region  
                region_mask = np.zeros((BEV_HEIGHT, BEV_WIDTH), dtype=np.uint8)
                region_mask[BEV_HEIGHT//2:, :] = 255
                
            elif cam_name == "left":
                # Place in left region
                region_mask = np.zeros((BEV_HEIGHT, BEV_WIDTH), dtype=np.uint8)
                region_mask[:, 0:BEV_WIDTH//2] = 255
                
            elif cam_name == "right":
                # Place in right region
                region_mask = np.zeros((BEV_HEIGHT, BEV_WIDTH), dtype=np.uint8)
                region_mask[:, BEV_WIDTH//2:] = 255
            
            # Apply mask and add to canvas
            warped_masked = cv2.bitwise_and(warped, warped, mask=region_mask)
            
            # Simple overlay
            valid_pixels = region_mask > 0
            bev_canvas[valid_pixels] = warped_masked[valid_pixels]
            
        except Exception as e:
            print(f"Error processing {cam_name}: {e}")
    
    # Add vehicle and reference elements
    add_vehicle_and_grid(bev_canvas)
    
    return bev_canvas

def add_vehicle_and_grid(bev_image):
    """Add vehicle representation and grid to BEV image"""
    
    # Draw vehicle
    car_center = (BEV_WIDTH // 2, BEV_HEIGHT // 2)
    car_size = (40, 80)  # width, height in pixels
    
    # Draw car rectangle
    cv2.rectangle(bev_image, 
                 (car_center[0] - car_size[0]//2, car_center[1] - car_size[1]//2),
                 (car_center[0] + car_size[0]//2, car_center[1] + car_size[1]//2),
                 (0, 255, 255), -1)
    
    # Draw orientation arrow (pointing forward/up)
    arrow_start = (car_center[0], car_center[1] - car_size[1]//2)
    arrow_end = (car_center[0], car_center[1] - car_size[1]//2 - 20)
    cv2.arrowedLine(bev_image, arrow_start, arrow_end, (255, 0, 0), 3, tipLength=0.3)
    
    # Add grid lines
    grid_spacing = 50
    grid_color = (100, 100, 100)
    
    # Vertical lines
    for i in range(0, BEV_WIDTH, grid_spacing):
        cv2.line(bev_image, (i, 0), (i, BEV_HEIGHT), grid_color, 1)
    
    # Horizontal lines
    for i in range(0, BEV_HEIGHT, grid_spacing):
        cv2.line(bev_image, (0, i), (BEV_WIDTH, i), grid_color, 1)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bev_image, "FRONT", (BEV_WIDTH//2 - 30, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(bev_image, "REAR", (BEV_WIDTH//2 - 25, BEV_HEIGHT - 10), font, 0.7, (255, 255, 255), 2)
    cv2.putText(bev_image, "LEFT", (10, BEV_HEIGHT//2), font, 0.7, (255, 255, 255), 2)
    cv2.putText(bev_image, "RIGHT", (BEV_WIDTH - 80, BEV_HEIGHT//2), font, 0.7, (255, 255, 255), 2)

def interactive_calibration():
    """Interactive tool for calibrating homography points"""
    
    print("Interactive Calibration Mode")
    print("This will help you manually select ground plane points for better calibration")
    
    cameras = ["front", "rear", "left", "right"]
    
    for cam_name in cameras:
        try:
            img = load_and_undistort(cam_name)
            
            print(f"\nCalibrating {cam_name} camera")
            print("Click on 4 points that form a rectangle on the ground plane")
            print("Order: bottom-left, bottom-right, top-right, top-left")
            
            # Create a copy for point selection
            img_copy = img.copy()
            
            # You can implement point selection here
            # For now, just show the image
            cv2.namedWindow(f"Calibrate {cam_name}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"Calibrate {cam_name}", 800, 600)
            cv2.imshow(f"Calibrate {cam_name}", img_copy)
            
            print(f"Press any key to continue to next camera...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error in calibration for {cam_name}: {e}")

# Main processing
if __name__ == "__main__":
    try:
        print("=== Bird's Eye View Generator ===")
        print("Choose an option:")
        print("1. Debug camera views (recommended first)")
        print("2. Create simple BEV")
        print("3. Create debug BEV (with region boundaries)")
        print("4. Interactive calibration")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            debug_camera_views()
            
        elif choice == "2":
            print("Creating simple BEV...")
            bev_image = create_simple_bev()
            
            # Save the result
            output_path = str(root / "simple_bev.png")
            cv2.imwrite(output_path, bev_image)
            print(f"Simple BEV saved as {output_path}")
            
            # Display
            cv2.namedWindow("Simple Bird's Eye View", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Simple Bird's Eye View", 800, 800)
            cv2.imshow("Simple Bird's Eye View", bev_image)
            cv2.waitKey(0)
            
        elif choice == "3":
            print("Creating debug BEV...")
            bev_image = create_debug_bev()
            
            # Save the result
            output_path = str(root / "debug_bev.png")
            cv2.imwrite(output_path, bev_image)
            print(f"Debug BEV saved as {output_path}")
            
            # Display
            cv2.namedWindow("Debug Bird's Eye View", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Debug Bird's Eye View", 800, 800)
            cv2.imshow("Debug Bird's Eye View", bev_image)
            cv2.waitKey(0)
            
        elif choice == "4":
            interactive_calibration()
            
        else:
            print("Invalid choice. Running debug mode...")
            debug_camera_views()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()