import logging
import re
import pytesseract
import tempfile
import cv2

from matplotlib import pyplot as plt
from datetime import datetime
from PIL import Image

def preprocess_image (image_file):
    img = cv2.imread(image_file)
    grayscaled_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, im_bw = cv2.threshold(grayscaled_image, 132, 255, cv2.THRESH_BINARY)
    #no_noise = noise_removal(im_bw)
    no_borders = remove_borders(im_bw)
    return no_borders

def noise_removal(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)

def remove_borders(image):
    contours, heiarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return (crop)


def extract_text_from_image(image_path):
    # # Open an image file
    # # with Image.open(image_path) as img:

    # # Convert the OpenCV image (numpy array) to a PIL image
    # pil_image = Image.fromarray(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB))

    # # See if the conversion went well
    # plt.imshow(pil_image)
    # plt.axis('off')  # Hide axes
    # plt.show()

    # # Use pytesseract to do OCR on the image
    # # Adding custom config
    # # nld+eng: for the languages to detect
    # # --psm 6: Assume a single uniform block of text.
    # custom_config = r'-l nld+eng --psm 6'
    # text = pytesseract.image_to_string(pil_image, config=custom_config)
    # return text

    with Image.open(image_path) as img:
        custom_config = r'-l nld+eng --psm 6'
        text = pytesseract.image_to_string(img, config=custom_config)
    return text

def find_total_and_date(text):
    # Regular expressions for matching total price and date
    # \s*:?\ 
    #       L> \s* = a amount of spaces or tabs, amount not specified so "*"
    #       L> :?  = followed by an optional (= "?"-sign) ":"-sign 
    total_pattern = re.compile(r'(total|totaal)\s*:?\s*\€?\s*([\d.,]+)', re.IGNORECASE)
    # date_pattern = re.compile(r'(datum)\s*:?(\d{2}/\d{2}/\d{4}  |   \d{2}-\d{2}-\d{4}   |   \d{2}\.\d{2}\.\d{4})', re.IGNORECASE)
    date_pattern = re.compile(r'\s*(datum)\s*:?(\d{2}/\d{2}/\d{4}\s*\d{2}:\d{2})', re.IGNORECASE)

    
    # Find total price
    total_matches = total_pattern.findall(text)
    # "if total_matches" is checking if total_matches is non-empty*
    total_price = total_matches[0] if total_matches else "Total not found"
    
    # Find date
    date_matches = date_pattern.findall(text)
    date = date_matches[0] if date_matches else "Date not found"
    
    return total_price, date

def log_extracted_text(text, total, date):
    # Log the extracted text with a timestamp
    logging.info("Extracted Text:\n%s", text)
    logging.info("Total Price: %s", total)
    logging.info("Date: %s", date)

# Example usage
image_path = './images/Ticket3.jpg'
preprocessed_image = preprocess_image(image_path)

# Save the preprocessed image to a temporary file
with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
    cv2.imwrite(temp_file.name, preprocessed_image)
    temp_file_path = temp_file.name

extracted_text = extract_text_from_image(temp_file_path)
total_price, date = find_total_and_date(extracted_text)

# log_extracted_text(extracted_text, total_price, date)

print("Extracted Text:\n", extracted_text)
print("Total Price:", total_price)
print("Date:", date)
