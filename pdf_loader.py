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
import pandas as pd


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


def save_as_json(filename: str, documents: Any, path_prefix: str = ''):
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

def extract_documents(documents: List[Document], pages: List[int], toc_page:int) -> str:
    extracted_documents_list = []
    starting_page_number = pages[0] + toc_page
    ending_page_number = pages[1] + toc_page if len(pages) == 2 else starting_page_number
    while starting_page_number <= ending_page_number:
        extracted_documents_list.append(documents[starting_page_number].page_content)
        starting_page_number += 1
    return '\n'.join(extracted_documents_list)


def clean_disjoined_words(text: str) -> Any:
    template = """Given the following unstructured text below, make the text presentable such as correcting disjoined words
    (such as "me al" instead of "meal" or "t o" instead of "to"), separating numbers from words 
    (such as "6.1Assessment" into "6.1 Assessment"), remove newlines, etc. and give me the same exact text 
    with the errors corrected. Do not add or remove any details. Just return back the original 
    text with the corrected words:

    {text}

    YOUR RESPONSE:
    """

    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=os.environ['OPENAI_API_KEY'],
        model='gpt-3.5-turbo-1106'
    )

    prompt = PromptTemplate(
        input_variables=['text'], template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

    return chain.run({'text': text})

def extract_and_clean_text(to_extract: str, addl_details:str, text: str) -> Any:
    template = """Given the following unstructured text below after the hyphens, comprehend the text, 
    extract the '{to_extract}' part or section of the text, and remove any unnecessary sections that are not part of 
    '{to_extract}'. {addl_details}.
    Afterwards, clean the extracted subsection with data cleaning techniques such as correcting disjoined words
    (such as "me al" instead of "meal" or "t o" instead of "to"), separating numbers from words 
    (such as "6.1Assessment" into "6.1 Assessment"), removing newlines, removing special characters, etc.; Return only 
    the cleaned and extracted section as your response, do not add more details and do not add the '{to_extract}' title or
    header since we already know what it is.
    
    -------------------
    {text}

    YOUR RESPONSE:
    """

    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=os.environ['OPENAI_API_KEY'],
        model='gpt-3.5-turbo-1106'
    )

    prompt = PromptTemplate(
        input_variables=['to_extract', 'addl_details', 'text'], template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

    return chain.run({'to_extract': to_extract, 'addl_details': addl_details, 'text': text})

def clean_documents(documents: List[Document], pages: List[int], toc_page:int) -> List[dict]:
    extracted_documents_list = []
    starting_page_number = pages[0] + toc_page
    ending_page_number = pages[1] + toc_page if len(pages) == 2 else starting_page_number
    while starting_page_number <= ending_page_number:
        page_content = documents[starting_page_number].page_content
        page_content_without_disjoined = clean_disjoined_words(page_content.replace('\n', '')
                                                               .replace('_', ''))
        doc = Document(**{
            'page_content': page_content_without_disjoined,
            'metadata':documents[starting_page_number].metadata,
            'type': documents[starting_page_number].type
        })
        extracted_documents_list.append(document_to_dict(doc))
        starting_page_number += 1
    return extracted_documents_list

def extract_and_clean(documents: List[Document], pages: List[int], toc_page:int, to_extract:str, addl_details:str) -> Document:
    extracted_documents_list = []
    starting_page_number = pages[0] + toc_page
    ending_page_number = pages[1] + toc_page if len(pages) == 2 else starting_page_number
    combined_page_content_list = []
    while starting_page_number <= ending_page_number:
        page_content = documents[starting_page_number].page_content
        combined_page_content_list.append(page_content.replace('\n', '')
                                                               .replace('_', ''))
        starting_page_number += 1

    extract_and_cleaned_text = extract_and_clean_text(to_extract, addl_details, '\n'.join(combined_page_content_list))

    return Document(**{
        'page_content': extract_and_cleaned_text,
        'metadata': documents[starting_page_number].metadata,
        'type': documents[starting_page_number].type
    })

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
    # source_filepath = 'datasets/tesda_regulations_json_trainee_requirements'
    # tesda_regulation_pdf_list = []
    # for filename in os.listdir(source_filepath):
    #     tesda_regulation_pdf_list.append(load_tesda_regulation_pdf_from_json(os.path.join(source_filepath, filename)))
    #
    # count = 1
    # for tesda_regulation_pdf in tesda_regulation_pdf_list:
    #     print(count)
    #     print(tesda_regulation_pdf.name)
    #     table_of_contents_page = tesda_regulation_pdf.documents[tesda_regulation_pdf.toc_page].page_content
    #     section1_number = retrieve_section1_page('Section 1', table_of_contents_page)
    #     print(section1_number)
    #     tesda_regulation_pdf.section1_pages = [int(page) for page in section1_number.split('-')]
    #     save_as_json_tesda(tesda_regulation_pdf.name, tesda_regulation_pdf,
    #                          path_prefix='datasets\\tesda_regulations_json_section1')
    #     count += 1

    # Building the dataset using pandas
    # Loading the JSON files
    # source_filepath = 'datasets/tesda_regulations_json_section1_updated'
    # tesda_regulation_pdf_list = []
    # for filename in os.listdir(source_filepath):
    #     tesda_regulation_pdf_list.append(load_tesda_regulation_pdf_from_json(os.path.join(source_filepath, filename)))
    #
    # tesda_dataset_list = list()
    # for tesda_regulation_pdf in tesda_regulation_pdf_list:
    #     tesda_dataset_list.append({
    #         'name': tesda_regulation_pdf.name,
    #         'core_competencies': extract_documents(tesda_regulation_pdf.documents,
    #                                                tesda_regulation_pdf.core_pages,
    #                                                tesda_regulation_pdf.toc_page),
    #         'trainee_entry_requirements': extract_documents(tesda_regulation_pdf.documents,
    #                                                         tesda_regulation_pdf.trainee_entry_requirements_pages,
    #                                                         tesda_regulation_pdf.toc_page),
    #         'section_1': extract_documents(tesda_regulation_pdf.documents,
    #                                        tesda_regulation_pdf.section1_pages,
    #                                        tesda_regulation_pdf.toc_page)
    #     })
    #
    # print(tesda_dataset_list[:5])
    #
    # tesda_df = pd.DataFrame(tesda_dataset_list)
    # print(tesda_df.head())
    #
    # print(tesda_df.loc[0, 'trainee_entry_requirements'])
    #
    # tesda_df.to_csv('datasets/tesda_modules_dataset_v2.csv', index=False)

    # Cleaning
    # source_filepath = 'datasets/tesda_regulations_json_section1_updated'
    # tesda_regulation_pdf_list = []
    # for filename in os.listdir(source_filepath):
    #     tesda_regulation_pdf_list.append(load_tesda_regulation_pdf_from_json(os.path.join(source_filepath, filename)))
    #
    # tesda_dataset_list = list()
    # for tesda_regulation_pdf in tesda_regulation_pdf_list[60:]:
    #     tesda_dict = {
    #         'name': tesda_regulation_pdf.name,
    #         'core_competencies': clean_documents(tesda_regulation_pdf.documents,
    #                                                          tesda_regulation_pdf.core_pages,
    #                                                          tesda_regulation_pdf.toc_page)
    #     }
    #
    #     save_as_json(f'datasets/tesda_regulations_json_cleaned_disjoined/{tesda_regulation_pdf.name}', tesda_dict)

    # Extract Trainee entry requirements
    # for tesda_regulation_pdf in tesda_regulation_pdf_list[15:]:
    #     tesda_dict = {
    #         'name': tesda_regulation_pdf.name,
    #         'trainee_entry_requirements': document_to_dict(extract_and_clean(tesda_regulation_pdf.documents,
    #                                              tesda_regulation_pdf.trainee_entry_requirements_pages,
    #                                              tesda_regulation_pdf.toc_page,
    #                                              'Trainee Entry Requirements'))
    #     }
    #     print(tesda_dict)
    # #
    #     save_as_json(f'datasets/tesda_regulations_json_cleaned_trainee/{tesda_regulation_pdf.name}', tesda_dict)

    # Extract section 1 jobs
    # for tesda_regulation_pdf in tesda_regulation_pdf_list[35:]:
    #     tesda_dict = {
    #         'name': tesda_regulation_pdf.name,
    #         'section1_jobs': document_to_dict(extract_and_clean(tesda_regulation_pdf.documents,
    #                                              tesda_regulation_pdf.section1_pages,
    #                                              tesda_regulation_pdf.toc_page,
    #                                              'Potential jobs that a qualified person is competent to be',
    #                                             'These are usually found at the end of the text. Separate them by'
    #                                             'comma, and don\'t add any more details than what is specified in the text.'))
    #     }
    #     print(tesda_dict['section1_jobs'])
    # #
    #     save_as_json(f'datasets/tesda_regulations_json_cleaned_jobs/{tesda_regulation_pdf.name}', tesda_dict)

    # Loading the cleaned core competencies
    # source_filepath = 'datasets/tesda_regulations_json_cleaned_disjoined/'
    # tesda_dict_list = []
    # for filename in os.listdir(source_filepath):
    #     with open(os.path.join(source_filepath, filename), 'r', encoding='utf-8') as file:
    #         data = json.load(file)
    #         if isinstance(data, list):
    #             target_tesda_module = data[-1]
    #         elif isinstance(data, dict):
    #             target_tesda_module = data
    #         else:
    #             print(f'Invalid: {filename}')
    #         core_competencies_docs = [Document(**doc) for doc in target_tesda_module['core_competencies']]
    #         tesda_dict_list.append({
    #             'name': target_tesda_module['name'],
    #             'core_competencies': '\n'.join([doc.page_content for doc in core_competencies_docs])
    #         })
    #
    # tesda_cc_no_disjoined_df = pd.DataFrame(tesda_dict_list)
    # tesda_cc_no_disjoined_df.to_csv('datasets/csv/tesda_core_competencies_no_disjoined.csv', index=False)
    #
    # print(tesda_cc_no_disjoined_df.info())
    # print(tesda_cc_no_disjoined_df.name.unique())
    # print(tesda_cc_no_disjoined_df.name.nunique())

    # Loading the trainee entry requirements extracted and cleaned
    # source_filepath = 'datasets/tesda_regulations_json_cleaned_trainee/'
    # tesda_dict_list = []
    # for filename in os.listdir(source_filepath):
    #     with open(os.path.join(source_filepath, filename), 'r', encoding='utf-8') as file:
    #         data = json.load(file)
    #         tesda_dict_list.append({
    #             'name': data['name'],
    #             'trainee_entry_requirements': data['trainee_entry_requirements']['page_content']
    #         })
    #
    # tesda_ter_no_disjoined_df = pd.DataFrame(tesda_dict_list)
    # tesda_ter_no_disjoined_df.to_csv('datasets/csv/tesda_ter_no_disjoined.csv', index=False)
    #
    # print(tesda_ter_no_disjoined_df.info())
    # print(tesda_ter_no_disjoined_df.name.unique())
    # print(tesda_ter_no_disjoined_df.name.nunique())

    # Loading the section 1 jobs extracted and cleaned
    # source_filepath = 'datasets/tesda_regulations_json_cleaned_jobs/'
    # tesda_dict_list = []
    # for filename in os.listdir(source_filepath):
    #     with open(os.path.join(source_filepath, filename), 'r', encoding='utf-8') as file:
    #         data = json.load(file)
    #         tesda_dict_list.append({
    #             'name': data['name'],
    #             'section1_jobs': data['section1_jobs']['page_content']
    #         })
    #
    # tesda_jobs_no_disjoined_df = pd.DataFrame(tesda_dict_list)
    # tesda_jobs_no_disjoined_df.to_csv('datasets/csv/tesda_jobs_no_disjoined.csv', index=False)
    #
    # print(tesda_jobs_no_disjoined_df.info())
    # print(tesda_jobs_no_disjoined_df.name.unique())
    # print(tesda_jobs_no_disjoined_df.name.nunique())

    # tesda_cc_no_disjoined_df = pd.read_csv('datasets/csv/tesda_core_competencies_no_disjoined.csv')
    # tesda_ter_no_disjoined_df = pd.read_csv('datasets/csv/tesda_ter_no_disjoined.csv')
    # tesda_jobs_no_disjoined_df = pd.read_csv('datasets/csv/tesda_jobs_no_disjoined.csv')
    #
    # merged_tesda_df = (tesda_cc_no_disjoined_df.merge(tesda_ter_no_disjoined_df, on='name')
    #                    .merge(tesda_jobs_no_disjoined_df, on='name'))
    #
    # print(merged_tesda_df.info())
    # print(merged_tesda_df.name.unique())
    # print(merged_tesda_df.name.nunique())
    #
    # merged_tesda_df.to_csv('datasets/csv/tesda_modules_dataset_v3.csv', index=False)

    pass
