#!/usr/bin/env python3
"""Parse PDF documents using PaddleOCR."""

import fitz  # PyMuPDF
from paddleocr import PaddleOCR, PaddleOCRVL
from pathlib import Path
import argparse


def parse_pdf(pdf_path: str, output_dir: str = "output", use_vl: bool = False):
    """Parse PDF document and extract text with OCR.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save output files
        use_vl: Use PaddleOCR-VL for complex documents
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Initialize OCR
    if use_vl:
        print("Using PaddleOCR-VL for complex document parsing...")
        ocr = PaddleOCRVL()
    else:
        print("Using PaddleOCR for text recognition...")
        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )
    
    # Open PDF
    doc = fitz.open(pdf_path)
    
    all_results = []
    
    for page_num in range(len(doc)):
        print(f"Processing page {page_num + 1}/{len(doc)}...")
        
        # Render page to image
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
        img_path = f"{output_dir}/page_{page_num + 1}.png"
        pix.save(img_path)
        
        # Run OCR
        result = ocr.predict(input=img_path)
        
        for res in result:
            all_results.append({
                "page": page_num + 1,
                "result": res
            })
            res.print()
            res.save_to_json(f"{output_dir}/page_{page_num + 1}.json")
            
            if use_vl:
                res.save_to_markdown(f"{output_dir}/page_{page_num + 1}.md")
    
    doc.close()
    print(f"\nPDF parsing complete. Results saved to {output_dir}/")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse PDF documents using PaddleOCR")
    parser.add_argument("pdf_file", help="Path to the PDF file")
    parser.add_argument("-o", "--output", default="output", 
                        help="Output directory (default: output)")
    parser.add_argument("--vl", action="store_true",
                        help="Use PaddleOCR-VL for complex documents")
    
    args = parser.parse_args()
    
    parse_pdf(args.pdf_file, args.output, args.vl)
