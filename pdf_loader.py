import os
import json
from typing import List, Any, Optional
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.prompts.prompt import PromptTemplate
from langchain.docstore.document import Document
from tesda_regulation_pdf import TesdaRegulationPDF

os.environ["OPENAI_API_KEY"] = ''


def document_to_dict(doc: Document) -> dict:
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata,
        "type": doc.type
    }

def tesda_regulation_to_dict(doc: TesdaRegulationPDF) -> dict:
    return {
        "name": doc.name,
        "documents": doc.documents,
        "toc_page": doc.toc_page,
        "core_pages": doc.core_pages,
        "core_competency": doc.core_competency
    }


def save_as_json(filename: str, documents: List[dict], path_prefix: str = ''):
    base_filename = os.path.splitext(filename)[0]
    print(base_filename)
    with open(f'{os.path.join(path_prefix, base_filename)}.json', 'w', encoding='utf-8') as file:
        json.dump(documents, file, ensure_ascii=False, indent=4)

def load_from_json(filename: str, path_prefix: str = '') -> TesdaRegulationPDF:
    file_path = os.path.join(path_prefix, filename)
    base_filename = os.path.basename(file_path)
    with open(file_path, 'r', encoding='utf-8') as file:
        documents_data = json.load(file)
    return TesdaRegulationPDF(name=os.path.splitext(base_filename)[0], documents=[Document(**data) for data in documents_data])

def load_documents(filename: str) -> List[Document]:
    loader = PyPDFLoader(filename)
    raw_documents = loader.load_and_split()
    print(f"loaded {len(raw_documents)} documents")
    # print(raw_documents[:5])
    # print('\n')

    return raw_documents

def retrieve_page_numbers(keyword: str, document: Document) -> Any:
    template = """Given the following unstructured Table of Contents from a PDF below, 
        Appropriately find the '{keyword}' and give me its starting and ending page numbers separated by a hyphen. (for example: 26-52).
        You response should only be the format specified above, do not add any more details.
        
        UNSTRUCTURED TABLE OF CONTENTS:
        {table_of_contents}
        
        EXAMPLE RESPONSE:
        32-46
        
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


def get_table_contents_pages(documents: List[Document]) -> Optional[int]:
    for document in documents:
        if 'TABLE OF CONTENTS' in document.page_content:
            return document.metadata.get('page')
    return None

if __name__ == "__main__":
    # for dataset in datasets:
    #     documents = load('datasets/' + dataset)
    #     page_numbers_response = retrieve_page_numbers('Core Competencies', documents[1])
    #     page_numbers_response_dict[dataset] = page_numbers_response
    #
    #
    # print(page_numbers_response_dict)

    # Loading the PDFs using PDFLoader then saving as a JSON file.
    # Do not uncomment if datasets/json directory is already populated
    # folder_path = 'datasets\\raw'
    # count = 1
    # for filename in os.listdir(folder_path):
    #     print(f'Count: {count}')
    #     file_path = os.path.join(folder_path, filename)
    #     documents = load_documents(file_path)
    #     save_as_json(filename, [document_to_dict(doc) for doc in documents], path_prefix='datasets\\json')

    json_path = 'datasets\\json'
    tesda_regulation_pdf_list: List[TesdaRegulationPDF] = []
    for json_document in os.listdir(json_path):
        tesda_regulation_pdf = load_from_json(os.path.join(json_path, json_document))
        tesda_regulation_pdf.toc_page = get_table_contents_pages(tesda_regulation_pdf.documents)
        tesda_regulation_pdf_list.append(tesda_regulation_pdf)

    for tesda_regulation_pdf in tesda_regulation_pdf_list:
        print(tesda_regulation_pdf.__repr__())
        #print(tesda_regulation_pdf.documents[tesda_regulation_pdf.toc_page].page_content)

    print(len(tesda_regulation_pdf_list))

    # populating the core competency pages
    for tesda_regulation_pdf in tesda_regulation_pdf_list[:15]:
        table_of_contents_page = retrieve_page_numbers('Core Competencies',
                                                       tesda_regulation_pdf.documents[tesda_regulation_pdf.toc_page])
        tesda_regulation_pdf.core_pages = [int(page) for page in table_of_contents_page.split('-')]

