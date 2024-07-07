import pytesseract
from PIL import Image

def extract_text_from_image(image_path):
    # Open an image file
    with Image.open(image_path) as img:
        # Use pytesseract to do OCR on the image
        text = pytesseract.image_to_string(img)
    return text

# Example usage
image_path = "./images/Ticket1.jpg"
extracted_text = extract_text_from_image(image_path)
print(extracted_text)
