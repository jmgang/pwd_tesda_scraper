
from utils import load_tesda_regulation_pdf_from_json
from pypdf import PdfReader, PdfWriter
import os


def extract_pdf_pages(pdf_file_path, pages, output_directory):
    base_filename = os.path.splitext(os.path.basename(pdf_file_path))[0]
    pdf_reader = PdfReader(pdf_file_path)
    pdf_writer = PdfWriter()

    for page_num in pages:
        pdf_writer.add_page(pdf_reader.pages[page_num])

    output_file_path = os.path.join(output_directory, f"{base_filename}_core_competencies_short.pdf")
    with open(output_file_path, 'wb') as f:
        pdf_writer.write(f)


if __name__ == "__main__":
    core_short_filepath = 'datasets/tesda_regulations_json_core_short/'
    raw_filepath = 'datasets/raw/'

    for filename in os.listdir(core_short_filepath):
        tesda_regulation_pdf = load_tesda_regulation_pdf_from_json(os.path.join(core_short_filepath, filename))
        cc_short_pages = []
        if len(tesda_regulation_pdf.cc_short_pages) > 1:
            starting_cc_page = tesda_regulation_pdf.cc_short_pages[0] + tesda_regulation_pdf.toc_page
            ending_cc_page = tesda_regulation_pdf.cc_short_pages[1] + tesda_regulation_pdf.toc_page
            cc_short_pages = list(range(starting_cc_page, ending_cc_page+1))
        else:
            cc_short_pages = [tesda_regulation_pdf.cc_short_pages[0] + tesda_regulation_pdf.toc_page]

        pdf_filename = f"{os.path.splitext(filename)[0]}.pdf"
        print(pdf_filename)
        print(cc_short_pages)
        extract_pdf_pages(os.path.join(raw_filepath, pdf_filename), cc_short_pages,
                          'datasets/cc_short_pages_pdf/')

