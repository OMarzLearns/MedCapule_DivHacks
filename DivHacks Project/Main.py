import tensorflow as tf
import numpy as np
from spectra import rgb
import time, easyocr, cv2, webcolors


# Load a pre-trained model (e.g., MobileNetV2)
# 3 inputs: color, shape, text


def detect_color(image):
    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    avg_color = np.mean(image_rgb, axis=(0, 1))
    avg_color = np.round(avg_color, decimals=0)
    return avg_color

def detect_shapes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        num_vertices = len(approx)

        if num_vertices == 3:
            shapes.append("Triangle")
        elif num_vertices == 4:
            shapes.append("Rectangle or Square")
        elif num_vertices == 5:
            shapes.append("Polygon")
        elif num_vertices > 9:
            shapes.append("Circle")
        elif num_vertices >= 6:
            shapes.append("Oblong")


    return list(set(shapes))  # Return unique shapes

def detect_text(image):
    # Initialize the EasyOCR reader
    reader = easyocr.Reader(['en'])  # Specify the languages you want to use
    # Use EasyOCR to detect text in the image
    result = reader.readtext(image)

    return result

def capture_image():
    # Access the Mac camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    # Capture a single frame
    time.sleep(.5)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Could not read frame.")
        return None

    return frame

def get_color_name(rgb_tuple):
    try:
        # Convert RGB to hex
        hex_value = webcolors.rgb_to_hex(rgb_tuple)
        # Get the color name directly
        return webcolors.hex_to_name(hex_value)
    except ValueError:
        # If exact match not found, find the closest color
        return closest_color(rgb_tuple)

# URL = "https://www.webmd.com/pill-identification/search-results?"
# URL += "imprint1=" + textOnPill
# URL += "&imprint2=" + textOnBack
# URL += "&color=0&shape=0

def main():
    # Capture an image
    # image = capture_image()
    image = cv2.imread("Test3.jpg")

    if image is not None:
        # Detect color
        rgb_tuple = detect_color(image)
        print(rgb_tuple)
        color = rgb(*rgb_tuple)
        print(color.hexcode)
        color = webcolors.hex_to_name(color.hexcode)
        print(color)  # Output: red
        print(f"Average Color (RGB): {color}")

        # Detect shapes
        shapes = detect_shapes(image)
        print(f"Detected Shapes: {shapes}")

        # Detect text
        detected_text = detect_text(image)
        # # Print detected text
        for (bbox, text, prob) in detected_text:
            print(f"Detected Text: {text}, Confidence: {prob:.2f}")

        # Display the captured image
        cv2.imshow("Captured Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
