import pytesseract
from PIL import Image

def extract_text_from_image(image_path):
    # Open an image file
    with Image.open(image_path) as img:
        # Use pytesseract to do OCR on the image
        # Adding custom config
        # nld+eng: for the languages to detect
        # --psm 6: Assume a single uniform block of text.
        custom_config = r'-l nld+eng --psm 6'
        text = pytesseract.image_to_string(img, config=custom_config)
    return text

# Example usage
# Preprocessed image as image path
image_path = "./images/no_border_Ticket3.jpg"
extracted_text = extract_text_from_image(image_path)
print(extracted_text)
