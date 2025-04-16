from ultralytics import YOLO
import cv2
import supervision as sv
import pytesseract
import numpy as np
import re
import unicodedata
import os


class cv_information_extraction:
<<<<<<< HEAD
    def __init__(self, detection_path='best.pt'):
        self.model = YOLO(detection_path)
=======
    def __init__(self, best_model_path=None):
        if best_model_path is None:
            # Get the directory of the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Create the full path to best.pt
            best_model_path = os.path.join(current_dir, 'best.pt')

            # Optional: Check if it exists
            if os.path.exists(best_model_path):
                print(f"Found model: {best_model_path}")
            else:
                raise ("best.pt not found in current directory")
        self.model = YOLO(best_model_path)
>>>>>>> a5d2206 (Update: improved cv_information_extraction logic)
        self.label_annotator = sv.LabelAnnotator()
        self.box_annotator = sv.BoxAnnotator()
        # self.fill_mask = pipeline("fill-mask", model="bert-base-uncased")

    def detect_and_ocr(self, image=None, image_path=None):
        """
        Detects text regions in an image using YOLO, extracts text with Tesseract,
        and returns structured JSON data while displaying the image with bounding boxes.

        Args:
            image_path (str): Path to the input image.

        Returns:
            dict: JSON-like dictionary with detected text categories.
        """
        if image:
            if type(image) is not np.ndarray:
                img = np.array(image)
            else:
                img = image
        elif image_path:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = self.model(img)[0]
        detection = sv.Detections.from_ultralytics(result)

        # print(detection)
        im_json = {}  # Dictionary to store detected text
        personal_image = None
        for i, rslt in enumerate(detection.xyxy):
            x1, y1, x2, y2 = rslt
            # Convert bounding box coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            class_name = detection.data['class_name'][i]

            # Ensure coordinates stay within image bounds
            cropped_img = img[max(0, y1):min(img.shape[0], y2),
                              max(0, x1):min(img.shape[1], x2)]

            if class_name == "image":
                personal_image = cropped_img
                continue

            # Apply Tesseract OCR
            text = pytesseract.image_to_string(
                cropped_img, config="--psm 6 --oem 1", lang="eng+ara").strip()
            try:
                im_json[class_name] += '\n' + text
            except KeyError:
                im_json[class_name] = text
        label = [
            f"{result.names[class_name]}"
            for class_name in detection.class_id
        ]
        img = self.box_annotator.annotate(img.copy(), detections=detection)
        img = self.label_annotator.annotate(img, detection, label)

        return im_json, img, personal_image  # Return structured JSON data

    def detect_and_ocr_batch(self, images):
        """
        Processes a batch of images for detection and OCR.

        Args:
            images (list): List of PIL images or NumPy arrays.

        Returns:
            List of tuples: (json_info, annotated_img, personal_img)
        """
        img_arrays = []
        for image in images:
            if not isinstance(image, np.ndarray):
                img_arrays.append(np.array(image))
            else:
                img_arrays.append(image)

        # Perform batch inference
        results = self.model(img_arrays)

        output = []

        for i, result in enumerate(results):
            img = img_arrays[i]
            detection = sv.Detections.from_ultralytics(result)
            im_json = {}
            personal_image = None

            for j, rslt in enumerate(detection.xyxy):
                x1, y1, x2, y2 = map(int, rslt)
                class_name = detection.data['class_name'][j]
                cropped_img = img[max(0, y1):min(
                    img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]

                if class_name == "image":
                    personal_image = cropped_img
                    continue

                text = pytesseract.image_to_string(
                    cropped_img, config="--psm 3 --oem 3", lang="eng").strip()
                text = self.clean_text(text)
                try:
                    im_json[class_name] += '\n' + text
                except KeyError:
                    im_json[class_name] = text

            labels = [f"{result.names[c]}" for c in detection.class_id]
            annotated_img = self.box_annotator.annotate(img.copy(), detection)
            annotated_img = self.label_annotator.annotate(
                annotated_img, detection, labels)

            output.append((im_json, annotated_img, personal_image))

        return output

    def correct_sentence(self, sentence):
        words = sentence.split()
        corrected = []
        for i, word in enumerate(words):
            masked = words[:i] + ['[MASK]'] + words[i+1:]
            masked_sentence = ' '.join(masked)
            predictions = self.fill_mask(masked_sentence)
            best_guess = predictions[0]['token_str']
            if best_guess.lower() != word.lower():
                corrected.append(best_guess)
            else:
                corrected.append(word)
        return ' '.join(corrected)

    def clean_text(self, text):
        text = unicodedata.normalize("NFKC", text)  # normalize unicode
        # remove non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
