import json
from tesda_regulation_pdf import TesdaRegulationPDF

def load_tesda_regulation_pdf_from_json(filename: str) -> TesdaRegulationPDF:
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return TesdaRegulationPDF(**data)