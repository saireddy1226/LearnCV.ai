
# Image Processing Toolkit — Deliverables

**Deadline:** Sep 8, 2025  
**Author:** Harshit V Shah

## How to Run (GUI)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` — Streamlit GUI app (two-panel layout, sidebar ops, status bar, save button, split-screen compare, webcam bonus).
- `utils.py` — Reusable image-processing functions.
- `ImageToolkit.ipynb` — Fundamentals notebook with theory + practice tasks.
- `Report.pdf` — Concise report with notes and auto-generated examples.
- `requirements.txt` — Required Python packages.

## Features Checklist
- Image Info, Color Conversions (RGB/HSV/YCbCr/Gray)
- Transformations (Rotate, Scale, Translate, Affine, Perspective)
- Filtering & Morphology (Mean/Gaussian/Median, Sobel/Laplacian, Dilation/Erosion/Opening/Closing)
- Enhancement (Histogram Eq, Contrast Stretch, Sharpen)
- Edge Detection (Sobel, Canny, Laplacian)
- Compression (JPG/PNG/BMP, size comparison)
- Bonus: Sliders, split-screen compare, webcam frame processing, save processed image
