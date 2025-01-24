import cv2
import numpy as np
import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def process_contour(contour, image, digital_binary, area_dir, i):
    contour_image = image.copy()
    cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(area_dir, 'top_contour_image.jpg'), contour_image)

    x, y, w, h = cv2.boundingRect(contour)
    enclosed_area = image[y:y+h, x:x+w]

    if w < 100 or h < 100 or abs(w - h) > 50:
        print(f"Enclosed area {i+1} is invalid (width: {w}, height: {h}). Skipping.\n")
        return 0, None, None, None, None

    cv2.imwrite(os.path.join(area_dir, 'enclosed_area.jpg'), enclosed_area)
    print(f"Enclosed area {i+1} saved successfully.\n")

    enclosed_gray = cv2.cvtColor(enclosed_area, cv2.COLOR_BGR2GRAY)
    digital_binary_resized = cv2.resize(digital_binary, (enclosed_gray.shape[1], enclosed_gray.shape[0]))
    print(f"Digital pattern resized successfully for enclosed area {i+1}.\n")

    best_match_percentage = 0
    best_brightness = 0
    best_rotation_angle = 0
    best_binary_image = None
    best_difference_image = None

    for brightness in range(128, 205, 5):
        recalculated_binary = np.where(enclosed_gray > brightness, 255, 0).astype(np.uint8)
        for angle in range(0, 360, 10):
            M = cv2.getRotationMatrix2D((recalculated_binary.shape[1] // 2, recalculated_binary.shape[0] // 2), angle, 1)
            rotated_binary = cv2.warpAffine(recalculated_binary, M, (recalculated_binary.shape[1], recalculated_binary.shape[0]))
            difference = cv2.absdiff(rotated_binary, digital_binary_resized)
            match_percentage = (np.sum(difference == 0) / difference.size) * 100
            if match_percentage > best_match_percentage:
                best_match_percentage = match_percentage
                best_brightness = brightness
                best_rotation_angle = angle
                best_binary_image = rotated_binary
                best_difference_image = difference

    for brightness in range(best_brightness - 4, best_brightness + 5):
        recalculated_binary = np.where(enclosed_gray > brightness, 255, 0).astype(np.uint8)
        for angle in range(best_rotation_angle - 9, best_rotation_angle + 10):
            M = cv2.getRotationMatrix2D((recalculated_binary.shape[1] // 2, recalculated_binary.shape[0] // 2), angle, 1)
            rotated_binary = cv2.warpAffine(recalculated_binary, M, (recalculated_binary.shape[1], recalculated_binary.shape[0]))
            difference = cv2.absdiff(rotated_binary, digital_binary_resized)
            match_percentage = (np.sum(difference == 0) / difference.size) * 100
            if match_percentage > best_match_percentage:
                best_match_percentage = match_percentage
                best_binary_image = rotated_binary
                best_difference_image = difference

    print(f'Best Match Percentage for enclosed area {i+1}: {best_match_percentage:.2f}%\n')
    with open(os.path.join(area_dir, 'best_match.txt'), 'w') as f:
        f.write(f'Best Match Percentage: {best_match_percentage:.2f}%\n')
        f.write(f'Brightness: {best_brightness}\n')
        f.write(f'Rotation Angle: {best_rotation_angle} degrees\n')

    if best_binary_image is not None:
        cv2.imwrite(os.path.join(area_dir, 'best_binary_image.jpg'), best_binary_image)
    if best_difference_image is not None:
        cv2.imwrite(os.path.join(area_dir, 'difference_image.jpg'), best_difference_image)

    return best_match_percentage, best_binary_image, best_difference_image, enclosed_area, best_rotation_angle

def main():
    print("Step 1: Load the image")
    image_path = 'camera_image/photo.png'
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    print("Image loaded successfully.\n")

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, 'original_image.jpg'), image)

    print("Step 2: Convert to grayscale and apply Gaussian blur")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    print("Grayscale conversion and Gaussian blur applied successfully.\n")

    cv2.imwrite(os.path.join(output_dir, 'grayscale_image.jpg'), gray)
    cv2.imwrite(os.path.join(output_dir, 'blurred_image.jpg'), blurred)

    print("Step 3: Color segmentation to isolate white areas")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    segmented = cv2.bitwise_and(image, image, mask=mask)
    print("Color segmentation applied successfully.\n")
    cv2.imwrite(os.path.join(output_dir, 'segmented_image.jpg'), segmented)

    print("Expanding white pixels around by 1 pixel")
    kernel = np.ones((3, 3), np.uint8)
    expanded_segmented = cv2.dilate(mask, kernel, iterations=1)
    print("White pixels expanded successfully.\n")
    cv2.imwrite(os.path.join(output_dir, 'expanded_segmented_image.jpg'), expanded_segmented)

    print("Step 5: Find and save the top 10 biggest enclosed areas with white border")
    contours, _ = cv2.findContours(expanded_segmented.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    print(f"Found {len(contours)} contours.\n")

    best_overall_match_percentage = 0
    best_overall_binary_image = None
    best_overall_difference_image = None
    best_overall_image = None
    best_test_image = None
    best_rotation_angle = 0

    for test_image_index in range(1, 8):
        digital_pattern_path = f'test_image/test{test_image_index}.jpg'
        digital_pattern = cv2.imread(digital_pattern_path, cv2.IMREAD_GRAYSCALE)
        if digital_pattern is None:
            raise FileNotFoundError(f"Digital pattern not found at path: {digital_pattern_path}")
        _, digital_binary = cv2.threshold(digital_pattern, 128, 255, cv2.THRESH_BINARY)
        print(f"Digital pattern {test_image_index} loaded and binarized successfully.\n")

        with ThreadPoolExecutor() as executor:
            futures = []
            for i, contour in enumerate(contours):
                area_dir = os.path.join(output_dir, f'enclosed_area_{i+1}_test{test_image_index}')
                os.makedirs(area_dir, exist_ok=True)
                futures.append(executor.submit(process_contour, contour, image, digital_binary, area_dir, i))

            for future in futures:
                match_percentage, binary_image, difference_image, enclosed_area, rotation_angle = future.result()
                if match_percentage > best_overall_match_percentage:
                    best_overall_match_percentage = match_percentage
                    best_overall_binary_image = binary_image
                    best_overall_difference_image = difference_image
                    best_overall_image = enclosed_area
                    best_test_image = test_image_index
                    best_rotation_angle = rotation_angle
                    
    if best_overall_binary_image is not None and best_overall_difference_image is not None:
        cv2.imwrite(os.path.join(output_dir, 'best_overall_binary_image.jpg'), best_overall_binary_image)
        cv2.imwrite(os.path.join(output_dir, 'best_overall_difference_image.jpg'), best_overall_difference_image)
        print(f"Best overall match found in test image {best_test_image} with match percentage {best_overall_match_percentage:.2f}%")
        with open(os.path.join(output_dir, 'best_overall_match.txt'), 'w') as f:
            f.write(f"Best Match Percentage: {best_overall_match_percentage:.2f}%\n")
            f.write(f"Rotation Angle: {best_rotation_angle} degrees\n")
    if best_overall_image is not None:
        cv2.imwrite(os.path.join(output_dir, 'best_overall_image.jpg'), best_overall_image)
    print("Results saved successfully.\n")

if __name__ == "__main__":
    main()