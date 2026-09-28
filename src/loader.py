# src/loader.py

import os
import fitz


def load_pdf(pdf_path):

    doc = fitz.open(pdf_path)

    pages = []

    for page_num, page in enumerate(doc):
        text = page.get_text()

        if text.strip():
            pages.append({
                "text": text,
                "page": page_num + 1,
                "source": os.path.basename(pdf_path)
            })

    return pages