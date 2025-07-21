import cv2
import numpy as np
import matplotlib.pyplot as plt

def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)

    # Define polygon for ROI (adjust based on camera angle)
    polygon = np.array([[
        (int(0.1 * width), height),
        (int(0.4 * width), int(0.6 * height)),
        (int(0.6 * width), int(0.6 * height)),
        (int(0.9 * width), height)
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return line_img

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    cropped_edges = region_of_interest(edges)
    lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)
    line_img = draw_lines(frame, lines)
    combined = cv2.addWeighted(frame, 0.8, line_img, 1, 1)
    return combined

# Load image (replace with your actual path or video frame)
img_path = 'lane_test.jpg'
image = cv2.imread(img_path)

if image is None:
    raise FileNotFoundError(f"Image not found at: {img_path}")

result = process_frame(image)

# Display result in Colab
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("Detected Lane Lines")
plt.axis('off')
plt.show()
