import cv2
import numpy as np
import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def process_contour(contour, image, digital_binary, area_dir, i):
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

    # Resize the digital pattern to match the size of the detected pattern
    digital_binary_resized = cv2.resize(digital_binary, (enclosed_gray.shape[1], enclosed_gray.shape[0]))
    print(f"Digital pattern resized successfully for enclosed area {i+1}.\n")

    # Initialize variables to track the best match
    best_match_percentage = 0
    best_brightness = 0
    best_rotation_angle = 0
    best_binary_image = None
    best_difference_image = None

    # Try different brightness thresholds from 50% to 80% with a coarser step
    for brightness in range(128, 205, 5):  # 128 is 50% of 255, 204 is 80% of 255
        recalculated_binary = np.where(enclosed_gray > brightness, 255, 0).astype(np.uint8)

        # Coarse rotation step
        for angle in range(0, 360, 10):
            M = cv2.getRotationMatrix2D((recalculated_binary.shape[1] // 2, recalculated_binary.shape[0] // 2), angle, 1)
            rotated_binary = cv2.warpAffine(recalculated_binary, M, (recalculated_binary.shape[1], recalculated_binary.shape[0]))

            # Compare the patterns
            difference = cv2.absdiff(rotated_binary, digital_binary_resized)
            match_percentage = (np.sum(difference == 0) / difference.size) * 100

            # Update the best match if the current match is better
            if match_percentage > best_match_percentage:
                best_match_percentage = match_percentage
                best_brightness = brightness
                best_rotation_angle = angle
                best_binary_image = rotated_binary
                best_difference_image = difference

    # Refine brightness step around the best brightness found
    for brightness in range(best_brightness - 4, best_brightness + 5):
        recalculated_binary = np.where(enclosed_gray > brightness, 255, 0).astype(np.uint8)

        # Refine rotation step around the best angle found
        for angle in range(best_rotation_angle - 9, best_rotation_angle + 10):
            M = cv2.getRotationMatrix2D((recalculated_binary.shape[1] // 2, recalculated_binary.shape[0] // 2), angle, 1)
            rotated_binary = cv2.warpAffine(recalculated_binary, M, (recalculated_binary.shape[1], recalculated_binary.shape[0]))

            # Compare the patterns
            difference = cv2.absdiff(rotated_binary, digital_binary_resized)
            match_percentage = (np.sum(difference == 0) / difference.size) * 100

            # Update the best match if the current match is better
            if match_percentage > best_match_percentage:
                best_match_percentage = match_percentage
                best_rotation_angle = angle
                best_binary_image = rotated_binary
                best_difference_image = difference

    print(f'Best Match Percentage for enclosed area {i+1}: {best_match_percentage:.2f}% at brightness {best_brightness} and {best_rotation_angle} degrees\n')

    # Save the best match percentage, brightness, and rotation angle
    with open(os.path.join(area_dir, 'best_match.txt'), 'w') as f:
        f.write(f'Best Match Percentage: {best_match_percentage:.2f}%\n')
        f.write(f'Brightness: {best_brightness}\n')
        f.write(f'Rotation Angle: {best_rotation_angle} degrees\n')

    # Save the best binary image
    cv2.imwrite(os.path.join(area_dir, 'best_binary_image.jpg'), best_binary_image)

    # Save the difference image
    cv2.imwrite(os.path.join(area_dir, 'difference_image.jpg'), best_difference_image)

    return best_match_percentage, best_binary_image, best_difference_image

# Main function
def main():
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

    best_overall_match_percentage = 0
    best_overall_binary_image = None
    best_overall_difference_image = None
    best_test_image = None

    # Process each test image
    for test_image_index in range(1, 8):
        digital_pattern_path = f'test_image/test{test_image_index}.jpg'
        digital_pattern = cv2.imread(digital_pattern_path, cv2.IMREAD_GRAYSCALE)
        if digital_pattern is None:
            raise FileNotFoundError(f"Digital pattern not found at path: {digital_pattern_path}")
        _, digital_binary = cv2.threshold(digital_pattern, 128, 255, cv2.THRESH_BINARY)
        print(f"Digital pattern {test_image_index} loaded and binarized successfully.\n")

        # Process each of the top 10 enclosed areas in parallel
        with ThreadPoolExecutor() as executor:
            futures = []
            for i, contour in enumerate(contours):
                area_dir = os.path.join(output_dir, f'enclosed_area_{i+1}_test{test_image_index}')
                os.makedirs(area_dir, exist_ok=True)
                futures.append(executor.submit(process_contour, contour, image, digital_binary, area_dir, i))

            for future in futures:
                match_percentage, binary_image, difference_image = future.result()
                if match_percentage > best_overall_match_percentage:
                    best_overall_match_percentage = match_percentage
                    best_overall_binary_image = binary_image
                    best_overall_difference_image = difference_image
                    best_test_image = test_image_index

    # Save the best overall binary image and difference image
    if best_overall_binary_image is not None and best_overall_difference_image is not None:
        cv2.imwrite(os.path.join(output_dir, 'best_overall_binary_image.jpg'), best_overall_binary_image)
        cv2.imwrite(os.path.join(output_dir, 'best_overall_difference_image.jpg'), best_overall_difference_image)
        print(f"Best overall match found in test image {best_test_image} with match percentage {best_overall_match_percentage:.2f}%")

    print("Results saved successfully.\n")

if __name__ == "__main__":
    main()