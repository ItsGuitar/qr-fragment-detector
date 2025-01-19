import cv2
import numpy as np
import os
from PIL import Image

# Step 1: Load the image
print("Step 1: Load the image")
image_path = 'camera_image/photo.png'
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")
print("Image loaded successfully.\n")

# Create output directory
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Save the original image
cv2.imwrite(os.path.join(output_dir, 'original_image.jpg'), image)

# Step 2: Convert to grayscale and apply Gaussian blur
print("Step 2: Convert to grayscale and apply Gaussian blur")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
print("Grayscale conversion and Gaussian blur applied successfully.\n")

# Save the grayscale and blurred images
cv2.imwrite(os.path.join(output_dir, 'grayscale_image.jpg'), gray)
cv2.imwrite(os.path.join(output_dir, 'blurred_image.jpg'), blurred)

# Step 3: Color segmentation to isolate white areas
print("Step 3: Color segmentation to isolate white areas")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 30, 255])
mask = cv2.inRange(hsv, lower_white, upper_white)
segmented = cv2.bitwise_and(image, image, mask=mask)
print("Color segmentation applied successfully.\n")

# Save the segmented image
cv2.imwrite(os.path.join(output_dir, 'segmented_image.jpg'), segmented)

# Expand white pixels around by 1 pixel
print("Expanding white pixels around by 1 pixel")
kernel = np.ones((3,3), np.uint8)
expanded_segmented = cv2.dilate(mask, kernel, iterations=1)
print("White pixels expanded successfully.\n")

# Save the expanded segmented image
cv2.imwrite(os.path.join(output_dir, 'expanded_segmented_image.jpg'), expanded_segmented)

# Step 5: Find and save the top 10 biggest enclosed areas with white border
print("Step 5: Find and save the top 10 biggest enclosed areas with white border")
contours, _ = cv2.findContours(expanded_segmented.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
print(f"Found {len(contours)} contours.\n")

# Process each of the top 10 enclosed areas
for i, contour in enumerate(contours):
    # Create a directory for each enclosed area
    area_dir = os.path.join(output_dir, f'enclosed_area_{i+1}')
    os.makedirs(area_dir, exist_ok=True)

    # Draw and save the contour on the original image for visualization
    contour_image = image.copy()
    cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(area_dir, 'top_contour_image.jpg'), contour_image)

    # Save the enclosed area as a separate image
    x, y, w, h = cv2.boundingRect(contour)
    enclosed_area = image[y:y+h, x:x+w]
    cv2.imwrite(os.path.join(area_dir, 'enclosed_area.jpg'), enclosed_area)

    print(f"Enclosed area {i+1} saved successfully.\n")

    # Convert the enclosed area to grayscale
    enclosed_gray = cv2.cvtColor(enclosed_area, cv2.COLOR_BGR2GRAY)

    # Recalculate the enclosed binary image based on brightness
    recalculated_binary = np.where(enclosed_gray > 204, 255, 0).astype(np.uint8)  # 204 is 80% of 255

    # Save the recalculated binary image
    cv2.imwrite(os.path.join(area_dir, 'recalculated_binary_image.jpg'), recalculated_binary)

    # Load the digital form of the pattern
    digital_pattern_path = 'test_image/test1.jpg'
    digital_pattern = cv2.imread(digital_pattern_path, cv2.IMREAD_GRAYSCALE)
    if digital_pattern is None:
        raise FileNotFoundError(f"Digital pattern not found at path: {digital_pattern_path}")
    _, digital_binary = cv2.threshold(digital_pattern, 128, 255, cv2.THRESH_BINARY)
    print(f"Digital pattern loaded and binarized successfully for enclosed area {i+1}.\n")

    # Save the digital binary image
    cv2.imwrite(os.path.join(area_dir, 'digital_binary_image.jpg'), digital_binary)

    # Resize the digital pattern to match the size of the detected pattern
    digital_binary_resized = cv2.resize(digital_binary, (recalculated_binary.shape[1], recalculated_binary.shape[0]))
    print(f"Digital pattern resized successfully for enclosed area {i+1}.\n")

    # Save the resized digital binary image
    cv2.imwrite(os.path.join(area_dir, 'digital_binary_resized.jpg'), digital_binary_resized)

    # Initialize variables to track the best match
    best_match_percentage = 0
    best_rotation_angle = 0

    # Rotate the recalculated binary image by 1 degree increments and compare
    for angle in range(360):
        # Rotate the image
        M = cv2.getRotationMatrix2D((recalculated_binary.shape[1] // 2, recalculated_binary.shape[0] // 2), angle, 1)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((recalculated_binary.shape[0] * sin) + (recalculated_binary.shape[1] * cos))
        new_h = int((recalculated_binary.shape[0] * cos) + (recalculated_binary.shape[1] * sin))
        M[0, 2] += (new_w / 2) - (recalculated_binary.shape[1] / 2)
        M[1, 2] += (new_h / 2) - (recalculated_binary.shape[0] / 2)
        rotated_binary = cv2.warpAffine(recalculated_binary, M, (new_w, new_h))

        # Resize the digital pattern to match the size of the rotated image
        digital_binary_resized_rotated = cv2.resize(digital_binary_resized, (rotated_binary.shape[1], rotated_binary.shape[0]))

        # Save the rotated image
        cv2.imwrite(os.path.join(area_dir, f'rotated_binary_{angle}.jpg'), rotated_binary)

        # Compare the patterns
        difference = cv2.absdiff(rotated_binary, digital_binary_resized_rotated)
        match_percentage = (np.sum(difference == 0) / difference.size) * 100

        # Update the best match if the current match is better
        if match_percentage > best_match_percentage:
            best_match_percentage = match_percentage
            best_rotation_angle = angle

    print(f'Best Match Percentage for enclosed area {i+1}: {best_match_percentage:.2f}% at {best_rotation_angle} degrees\n')

    # Save the best match percentage and rotation angle
    with open(os.path.join(area_dir, 'best_match.txt'), 'w') as f:
        f.write(f'Best Match Percentage: {best_match_percentage:.2f}%\n')
        f.write(f'Rotation Angle: {best_rotation_angle} degrees\n')

print("Results saved successfully.\n")