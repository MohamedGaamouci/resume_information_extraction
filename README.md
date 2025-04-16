# Resume Information Extraction

This project uses a custom-trained YOLO model for **document layout analysis** and combines it with **OCR** and **NLP** to extract structured information (like titles, paragraphs, and lists) from resume documents.

---

## 🧠 Features

- 🔍 **Layout Detection**: YOLOv8 model segments resume sections (e.g., header, education, experience)
- 🧾 **OCR Extraction**: Applies OCR (Tesseract or others) to cropped sections
- 🧠 **NLP Structuring**: Post-processes the OCR results to organize the content

---

## 📁 Project Structure

```bash
resume_information_extraction/
│
├── main.py             # Main script to run YOLO + OCR pipeline
├── ocr_utils.py        # Utility functions for OCR and NLP structuring
├── best.pt   # Custom YOLOv8 model weights
├── README.md           # Project documentation
└── requirements.txt    # (Optional) Python dependencies
