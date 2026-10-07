"""Crop isolated explanatory footers from three saved figure PDFs."""
import argparse
import hashlib
import json
from pathlib import Path
import pdfplumber
from pypdf import PdfReader, PdfWriter

NAMES=['source_confidence_null','source_confidence_strong','gain_controls']
NOTE='Overlapping curves have equal fractions. All listed series are drawn.'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for name in NAMES:
        relative=Path('results/current/operating')/(name+'.pdf')
        original=args.package_root/relative
        with pdfplumber.open(original) as doc:
            if len(doc.pages)!=1:raise ValueError('A saved figure must have one page.')
            page=doc.pages[0];lines=page.extract_text_lines()
            matches=[line for line in lines if line['text']==NOTE]
            if len(matches)!=1:raise ValueError('The expected footer is not unique.')
            note=matches[0];cut=note['top']-4
            retained=[line for line in lines if line is not note]
            if any(line['bottom']>=cut for line in retained):raise ValueError('The crop would remove a figure label.')
            lower=page.height-cut
            reader=PdfReader(original);page_pdf=reader.pages[0]
            page_pdf.mediabox.lower_left=(float(page_pdf.mediabox.left),lower)
            page_pdf.cropbox.lower_left=(float(page_pdf.cropbox.left),lower)
            writer=PdfWriter();writer.add_page(page_pdf)
            target=args.output/(name+'.pdf');temporary=target.with_suffix('.tmp.pdf')
            with temporary.open('wb') as f:writer.write(f)
            temporary.replace(target)
            records.append({'figure':name,'source':str(relative),'source_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'lower_crop_points':lower,'scientific_text_bottom':max(line['bottom'] for line in retained),'crop_top_coordinate':cut})
    (args.output/'display_footer_crops.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps({'cropped':len(records),'output':str(args.output)}))
if __name__=='__main__':main()
