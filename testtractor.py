import os
from tqdm import tqdm
from textractor import Textractor
from textractor.data.constants import TextractFeatures
from textractcaller import Query


def extract_tables(s3_file_path, filename, output_directory):
    query = Query(text='What are the core competencies?')

    extractor = Textractor(profile_name="admin-general", region_name='us-east-1')
    # Assuming 'TABLES' is the correct feature constant
    document = extractor.start_document_analysis(file_source=s3_file_path + filename,
                                                 features=[TextractFeatures.TABLES
                                                         #  , TextractFeatures.QUERIES
                                                           ],
                                                 #queries=[query]
                                                 )

    for j, page in enumerate(document.pages):
        for i, table in enumerate(page.tables):
            print(f'On page {j}, table {i}')
            with open(f'{output_directory}/{os.path.splitext(filename)[0]}_p{j}_t{i}.csv', 'w') as csv_file:
                csv_file.write(table.to_csv())

    return document


if __name__ == "__main__":
    # Example usage
    filenames = []
    for filename in os.listdir('datasets/cc_short_pages_pdf/'):
        filenames.append(filename)

    for filename in tqdm(filenames[45:]):
        document = extract_tables('s3://pwd-tesda/input/', filename,
                                  'datasets/cc_short_csv/')
