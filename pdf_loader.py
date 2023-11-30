import os
import json
import re
from typing import List, Any, Optional
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.prompts.prompt import PromptTemplate
from langchain.docstore.document import Document
from tesda_regulation_pdf import TesdaRegulationPDF
from dotenv import load_dotenv


load_dotenv()  # take environment variables from .env.


def document_to_dict(doc: Document) -> dict:
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata,
        "type": doc.type
    }


def tesda_regulation_to_dict(tesda_doc: TesdaRegulationPDF) -> dict:
    return {
        "name": tesda_doc.name,
        "documents": [document_to_dict(doc) for doc in tesda_doc.documents],
        "toc_page": tesda_doc.toc_page,
        "core_pages": tesda_doc.core_pages,
        "competency_map_pages": tesda_doc.competency_map_pages,
        "trainee_entry_requirements_pages": tesda_doc.trainee_entry_requirements_pages,
        "section1_pages" : tesda_doc.section1_pages
    }


def save_as_json(filename: str, documents: List[dict], path_prefix: str = ''):
    base_filename = os.path.splitext(filename)[0]
    with open(f'{os.path.join(path_prefix, base_filename)}.json', 'w', encoding='utf-8') as file:
        json.dump(documents, file, ensure_ascii=False, indent=4)

def save_as_json_tesda(filename: str, tesda_pdf: TesdaRegulationPDF, path_prefix: str = ''):
    base_filename = os.path.splitext(filename)[0]
    with open(f'{os.path.join(path_prefix, base_filename)}.json', 'w', encoding='utf-8') as file:
        json.dump(tesda_regulation_to_dict(tesda_pdf), file, ensure_ascii=False, indent=4)

def load_documents_from_json(filename: str, path_prefix: str = '') -> TesdaRegulationPDF:
    file_path = os.path.join(path_prefix, filename)
    base_filename = os.path.basename(file_path)
    with open(file_path, 'r', encoding='utf-8') as file:
        documents_data = json.load(file)
    return TesdaRegulationPDF(name=os.path.splitext(base_filename)[0], documents=[Document(**data) for data in documents_data])

def load_tesda_regulation_pdf_from_json(filename: str) -> TesdaRegulationPDF:
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return TesdaRegulationPDF(**data)

def load_documents(filename: str) -> List[Document]:
    loader = PyPDFLoader(filename)
    raw_documents = loader.load_and_split()
    print(f"loaded {len(raw_documents)} documents")
    # print(raw_documents[:5])
    # print('\n')

    return raw_documents

def retrieve_page_numbers(keyword: str, document: str) -> Any:
    # Appropriately find the '{keyword}' and give me its starting and ending page numbers separated by a hyphen. (for example: 26-52).
    template = """Given the following unstructured Table of Contents from a PDF below, 
        Find the page numbers for '{keyword}'. It is usually found on the fourth line of page numbers. If there's no 
        fourth line of page numbers, it is on the third line. Even if the page numbers are right after '{keyword}', 
        double check it and find the correct page numbers according to the ordering.
        Refer to the 4 examples below:
        
        EXAMPLE 1:
        COMPETENCY STANDARDS   
  
          BASIC COMPETENCIES  
          COMMON COMPETENCIES  
          CORE COMPETENCIES   02 - 15  
        16 - 31  
        32 - 50
        
        SAMPLE RESPONSE 1: 
        32-50
        
        EXAMPLE 2:
        COMPETENCY  STANDARDS  
          BASIC COMPETENCIES  
          COMMON COMPETENCIES  
          CORE COMPETENCIES   
        3 -  5 0  
        3   -  1 5  
        1 6  -  3 4  
        3 3  -  5 0
        
        SAMPLE RESPONSE 2: 
        33-50
        
        EXAMPLE 3:
        COMPETENCY STANDARDS  
         BASIC COMPETENCIES  
         COMMON COMPETENCIES  
         CORE COMPETEN CIES 
         ELECTIVE COMPETENCY   
        2- 59 
        2   - 15 
        16 - 29 
        29 - 53 
        54 - 59
        
        SAMPLE RESPONSE 3: 
        29-53
        
        EXAMPLE 4:
        COMPETENCY STANDARDS   
        3 - 52  
          BASIC COMPETENCIES  
          COMMON COMP ETENCIES  
          CORE COMPETENCY  3 - 21  
        22 - 49  
        50 - 52
        
        SAMPLE RESPONSE 4:
        50-52 
        
        UNSTRUCTURED TABLE OF CONTENTS:
        {table_of_contents}
        
        YOUR RESPONSE:
        """

    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model='gpt-3.5-turbo-1106'
    )

    prompt = PromptTemplate(
        input_variables=['keyword', 'table_of_contents'], template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

    return chain.run({'keyword': keyword, 'table_of_contents': document})


def retrieve_page_number(keyword: str, document: str) -> Any:
    # Appropriately find the '{keyword}' and give me its starting and ending page numbers separated by a hyphen. (for example: 26-52).
    template = """Given the following unstructured Table of Contents from a PDF below, 
        Find the page numbers for '{keyword}'. 
        If you see that the page numbers of '{keyword}' has a hypen, return those page numbers. 
        If you see only one, return that page number. It can only be one or the other. 
        Your response should only be the format mentioned above. Do not add any more details. See examples below.
        
        EXAMPLE 1:
        COMPETENCY MAP  
  
        116  
  
        SAMPLE RESPONSE 1:
        116
        
        EXAMPLE 2:
        COMPETENCY MAP  138 - 139
        
        SAMPLE RESPONSE 2:
        138-139

        UNSTRUCTURED TABLE OF CONTENTS:
        {table_of_contents}

        YOUR RESPONSE:
        """

    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model='gpt-3.5-turbo-1106'
    )

    prompt = PromptTemplate(
        input_variables=['keyword', 'table_of_contents'], template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

    return chain.run({'keyword': keyword, 'table_of_contents': document})

def retrieve_trainee_requirements_number(keyword: str, document: str) -> Any:
    # Appropriately find the '{keyword}' and give me its starting and ending page numbers separated by a hyphen. (for example: 26-52).
    template = """Given the following unstructured Table of Contents from a PDF below, 
        Find the page numbers for '{keyword}'. 
        These page numbers are ordered according to the subsections and each subsection has their own numbering like 
        3.1, 3.2, etc. Now since these were scraped from a PDF, the page numbers are all messed up and it's not obvious
        what page numbers each subsection are being referred to but you can actually still see the correct page numbers
        based on the ordering of the subsections. These examples have explanations on them for you to understand but
        your response should only be the page number or page numbers if there are multiple page numbers. If somehow
        the page numbers are not visible or the text is None then return -1. Refer to the samples below.

        EXAMPLE 1:
        TRAINING ARRANGEMENTS  
        3.1.  Curriculum Design  
        3.1.1.  Basic  
        3.1.2.  Common  
        3.1.3.  Core  
        3.2.  Training Delivery  
        3.3.  Trainee Entry Requirements  
        3.4.  List of Tools, Equipment and Materials  
        3.5.  Training Facilities  
        3.6.  Trainers’ Qualifications  
        3.7.  Institutional Assessment  
          
        78  -  134  
        78  
        79 - 94  
        95 - 104  
        129  
        130  
        132  
        132  
        133  
        133  
        134  
        
        EXPLANATION 1:
        Since Trainee Entry Requirements is the 6th from the subsections list, the right page number would also be sixth
        from the list of page numbers, in this case it is 132.

        SAMPLE RESPONSE 1:
        132

        EXAMPLE 2:
        TRAINING ARRANGEMENTS   
 
        3.1  Curriculum Design  7 7  -  82  
        3.2  Training Delivery  82  -  83  
        3.3  Trainee Entry Requirements      8 3  
        3.4  List of Tools, Equipment and Materials  8 3  -  87  
        3.5  Training Facilities  8 7  
        3.6  Trainer’s Qualifications  8 8  
        3.7  Institutional Assessment  8 8 
        
        EXPLANATION 2:
        In this case, the subsections and their page numbers are much more obvious and we can see that the Trainee Entry
        Requirements is on page 83.

        SAMPLE RESPONSE 2:
        83
        
        EXAMPLE 3:
        TRAINING ARRANG E MENTS  
        3.1.  Curriculum Design  
        3.1.1.  Basic  
        3.1.2.  Common  
        3.1.3.  Core  
        3.2.  Training Delivery  
        3.3.  Trainee Entry Requirements  
        3.4.  List of Tools, Equipment and Materials  
        3.5.  Training Facilities  
        3.6.  Trainers’ Qualifications  
        3.7.  Institutional Assessment  
          
        71  -  113  
        71  
        72  
        79  
        89  
        109  
        111  
        111  
        112  
        113  
        113  
        
        EXPLANATION 3:
        In this case, the overall page numbers for the Section is also included. This runs from 71 to 113. Therefore, 
        the Trainee Entry Requirements is on the 7th one on this case instead of the 6th, thus the correct page number
        is page 111. It's beneficial to count the length of the subsections vs the page numbers to see any mismatch.
        
        SAMPLE RESPONSE 3:
        111

        UNSTRUCTURED TABLE OF CONTENTS:
        {table_of_contents}

        YOUR RESPONSE:
        """

    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model='gpt-3.5-turbo-1106'
    )

    prompt = PromptTemplate(
        input_variables=['keyword', 'table_of_contents'], template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

    return chain.run({'keyword': keyword, 'table_of_contents': document})


def retrieve_section1_page(keyword: str, document: str) -> Any:
    # Appropriately find the '{keyword}' and give me its starting and ending page numbers separated by a hyphen. (for example: 26-52).
    template = """Given the following unstructured Table of Contents from a PDF below, 
        Find the page numbers for '{keyword}'. 
        If you see that the page numbers of '{keyword}' has a hypen, return those page numbers. 
        If you see only one, return that page number. It can only be one or the other. 
        The page number or page numbers is/are usually page 1 or page 1-2. 
        Your response should only be the format mentioned above. Do not add any more details. See examples below.

        EXAMPLE 1:
        Agricultural Crops Production NC I  TABLE OF CONTENTS  
        AGRICULTURE AND FISHERY, PROCESSED FOOD AND BEVERAGES SECTOR  
 
        AGRICULTURAL CROPS PRODUCTION NC I  
         
         Page No.  
         
        PREFACE   
        FOREWORD   
          
        SECTION 1   AGRICULTURAL CROPS PRODUCTION  NC I 
        QUALIFICATION  01  


        SAMPLE RESPONSE 1:
        1

        EXAMPLE 2:
        TR - Agroentrepreneurship NCI V  Promulgated February 3, 2017  
         1  TABLE OF CONTENTS  
         
         
        AGRICULTURE, FORESTRY AND FISHERY  SECTOR  
         
        AGROENTREPRENEURSHIP  NCIV  
         
          Page/s  
         
        Section 1  AGROENTREPRENEURSHIP  NCIV  QUALIFICATION  
         2  
         
        Section 2   

        SAMPLE RESPONSE 2:
        2
        
        EXAMPLE 3:
        TABLE OF CONTENTS  
        
        TOURISM SECTOR  
        (HOTELS AND RESTAURANTS)  
         
        HOUSEKEEPING NC III  
         
         
           Page No.  
         
        SECTION 1    HOUSEKEEPING NC III  
           QUALIFICATION    1 -   2   
          
        SECTION 2       COMPETENCY STANDARDS                                                        
         
          B asic Competencies
        
        SAMPLE RESPONSE 3:
        1-2
        
        ACTUAL UNSTRUCTURED TABLE OF CONTENTS:
        {table_of_contents}

        YOUR RESPONSE:
        """

    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model='gpt-3.5-turbo-1106'
    )

    prompt = PromptTemplate(
        input_variables=['keyword', 'table_of_contents'], template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

    return chain.run({'keyword': keyword, 'table_of_contents': document})

def get_keyword_page(keyword, documents: List[Document]) -> Optional[int]:
    for document in documents:
        if keyword in document.page_content:
            return document.metadata.get('page')
    return None

def extract_text_between_sections(text, start_section, end_section):
    text = re.sub(r"SECTION\s+", "SECTION ", text)
    pattern = rf"{start_section}(.*?){end_section}"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None

def find_core_competencies_page_numbers(text):
    normalized_text = re.sub(r'\s+', ' ', text)
    pattern = r"CORE COMPETENCIES\s+(\d+)\s*-\s*(\d+)"
    matches = re.findall(pattern, normalized_text, re.IGNORECASE)
    if matches:
        return matches[0]
    else:
        return None


if __name__ == "__main__":
    # Loading the PDFs using PDFLoader then saving as a JSON file.
    # Do not uncomment if datasets/json directory is already populated
    # folder_path = 'datasets\\raw'
    # count = 1
    # for filename in os.listdir(folder_path):
    #     print(f'Count: {count}')
    #     file_path = os.path.join(folder_path, filename)
    #     documents = load_documents(file_path)
    #     save_as_json(filename, [document_to_dict(doc) for doc in documents], path_prefix='datasets\\json')

    # Getting the table of contents page numbers
    # json_path = 'datasets\\json'
    # tesda_regulation_pdf_list: List[TesdaRegulationPDF] = []
    # for json_document in os.listdir(json_path):
    #     tesda_regulation_pdf = load_documents_from_json(os.path.join(json_path, json_document))
    #     tesda_regulation_pdf.toc_page = get_keyword_page('TABLE OF CONTENTS', tesda_regulation_pdf.documents)
    #     tesda_regulation_pdf_list.append(tesda_regulation_pdf)

    # Getting the JSON files for each module
    # filepath = 'datasets/tesda_regulations_json'
    # tesda_regulation_pdf_list = []
    # for filename in os.listdir(filepath):
    #     tesda_regulation_pdf_list.append(load_tesda_regulation_pdf_from_json(os.path.join(filepath, filename)))
    #
    # print(len(tesda_regulation_pdf_list))
    #
    # Finding the core competencies pages were inadvertently omitted
    #
    # Setting the competency map page
    # count = 0
    # for tesda_regulation_pdf in tesda_regulation_pdf_list:
    #     print(count)
    #     print(tesda_regulation_pdf.name)
    #     print(tesda_regulation_pdf.toc_page)
    #     competency_map_page_num = retrieve_page_number('COMPETENCY MAP',
    #                             tesda_regulation_pdf.documents[tesda_regulation_pdf.toc_page].page_content.upper())
    #     print(competency_map_page_num)
    #     tesda_regulation_pdf.competency_map_pages = [int(page) for page in competency_map_page_num.split('-')]
    #     save_as_json(tesda_regulation_pdf.name, tesda_regulation_pdf,
    #                  path_prefix='datasets\\tesda_regulations_json_competency_map')
    #     count += 1

    # Loading json files from tesda_regulations_json_competency_map directory
    # source_filepath = 'datasets/tesda_regulations_json_competency_map'
    #
    # tesda_regulation_pdf_list = []
    # for filename in os.listdir(source_filepath):
    #     tesda_regulation_pdf_list.append(load_tesda_regulation_pdf_from_json(os.path.join(source_filepath, filename)))

    # Finding the trainee requirements page numbers by first
    # extracting text between section 3 and 4 of table of contents pages
    # then use the LLM to find the right page number
    # count = 1
    # for tesda_regulation_pdf in tesda_regulation_pdf_list[60:]:
    #     print(count)
    #     print(tesda_regulation_pdf.name)
    #     table_of_contents_page = tesda_regulation_pdf.documents[tesda_regulation_pdf.toc_page].page_content
    #     trainee_requirements_number = retrieve_trainee_requirements_number(
    #         '3.3 Trainee Entry Requirements', table_of_contents_page
    #     )
    #     print(trainee_requirements_number)
    #     tesda_regulation_pdf.trainee_entry_requirements_pages = [int(page) for page in trainee_requirements_number.split('-')]
    #     save_as_json_tesda(tesda_regulation_pdf.name, tesda_regulation_pdf,
    #                      path_prefix='datasets\\tesda_regulations_json_trainee_requirements')
    #     count += 1

    # Loading JSON files from datasets/tesda_regulations_json_trainee_requirements
    source_filepath = 'datasets/tesda_regulations_json_trainee_requirements'
    tesda_regulation_pdf_list = []
    for filename in os.listdir(source_filepath):
        tesda_regulation_pdf_list.append(load_tesda_regulation_pdf_from_json(os.path.join(source_filepath, filename)))

    count = 1
    for tesda_regulation_pdf in tesda_regulation_pdf_list:
        print(count)
        print(tesda_regulation_pdf.name)
        table_of_contents_page = tesda_regulation_pdf.documents[tesda_regulation_pdf.toc_page].page_content
        section1_number = retrieve_section1_page('Section 1', table_of_contents_page)
        print(section1_number)
        tesda_regulation_pdf.section1_pages = [int(page) for page in section1_number.split('-')]
        save_as_json_tesda(tesda_regulation_pdf.name, tesda_regulation_pdf,
                             path_prefix='datasets\\tesda_regulations_json_section1')
        count += 1
