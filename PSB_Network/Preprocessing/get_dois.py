import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pdfplumber
import re
import pandas as pd
from tqdm import tqdm
import csv
import nltk
import re
from nltk.corpus import stopwords
import metapub
from time import sleep
import pybliometrics
from pybliometrics.scopus import AbstractRetrieval
from pybliometrics.scopus import AuthorRetrieval
from pybliometrics.scopus.exception import Scopus404Error

os.environ['NCBI_API_KEY'] = '4216f1a2a91c969d346d66f491930ec94508'
# Download the stop words list
nltk.download('stopwords')

# Define the function to clean the text
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[-:;()"\',]', ' ', text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words and word != 'abstract']
    
    # Join the cleaned words back into a single string
    cleaned_text = ' '.join(cleaned_words)
    return cleaned_text

def query_pubmed(title, authors, year, detailed=False):
    base_url = "https://pubmed.ncbi.nlm.nih.gov/api/citmatch/"
    if detailed:
        params = {
            "method": "auto",
            "raw-text": f"{title.replace(' ', '+')}+{authors.replace(' ', '+')}+{year}",
            "journal": "Pac Symp Biocomput",
            "retmode": "json"
        }
    else:
       params = {
            "method": "auto",
            "raw-text": title,
            "journal": "Pac Symp Biocomput",
            "retmode": "json"
        } 
            
    response = requests.get(base_url, params=params)
    data = response.json()
    # Extract PubMed ID from the response
    pubmed_ids = data.get('result', {}).get('uids', [])
    return pubmed_ids

# Clean the titles, query PubMed API, and store PubMed IDs
pubmed_results = []
for index, row in tqdm(df.iterrows()):
    title = row['Title']
    authors = row['Authors']
    year = row['Year']
    cleaned_title = clean_text(title)
    pubmed_ids = query_pubmed(cleaned_title, authors, year)
    if pubmed_ids == []:
        pubmed_ids = query_pubmed(cleaned_title, authors, year, True)
#         if pubmed_ids == []:
#             print(f"no record found for title: {cleaned_title}")
#             print(f"{cleaned_title.replace(' ', '+')}+{authors.replace(' ', '+')}+{year}")
#             print(f"{title.replace(' ', '+')}+{authors.replace(' ', '+')}+{year}")
    pubmed_results.append({'Title': title, "Authors": authors, "Year": year, "DOI": row['DOI'], 'PubMed IDs': pubmed_ids})

# Create a DataFrame with the results
new_results_df = pd.DataFrame(pubmed_results)

pubmed_data = new_results_df
pubmed_data.insert(0, 'Original Title', '')

def get_journal_name_and_title(fetch, pubmed_id):
    article = fetch.article_by_pmid(pubmed_id)
    return (article.journal, article.title)

fetch = metapub.PubMedFetcher()
# Iterate through each row and check the PubMed IDs
for index, row in tqdm(pubmed_data.iterrows()):
    orig_title = row['Title']
    pubmed_ids = [item['pubmed'] for item in row['PubMed IDs']]
    for pubmed_id in pubmed_ids:
        journal_name, title = get_journal_name_and_title(fetch, pubmed_id)
        if journal_name == 'Pac Symp Biocomput':
            correct_pubmed_id = pubmed_id
            break
        # Be polite and avoid hitting the server too hard
        sleep(0.5)
    if correct_pubmed_id:
        pubmed_data.at[index, 'PubMed IDs'] = correct_pubmed_id
        pubmed_data.at[index, 'Title'] = title
        pubmed_data.at[index, 'Original Title'] = orig_title
    else:
        "no correct id found"
    


pybliometrics.scopus.init() #includes Elsevier API Key
df = pubmed_data
def query_scopus_authors_cited(pmid):
    full_auth = {}
    ab = AbstractRetrieval(pmid)
    for author in ab.authors:
        id = author.auid
        au = AuthorRetrieval(id)
        full_auth[id] = f"{au.given_name} {au.surname}"
    return (full_auth, ab.citedby_count)

df['Full Authors'] = None
df['Cited By Count'] = None

default = "pmid"
# Iterate through each PubMed ID and query Scopus
for index, row in tqdm(df.iterrows()):
    pmid = row['PubMed IDs']
    doi = row['DOI']
    if default == "pmid":
        try:
            (full_authors, cited_by_count) = query_scopus_authors_cited(pmid)
        except Scopus404Error:
            print(f"Scopus404Error for PMID: {pmid}. Trying alternative method...")
            default = "doi"
            (full_authors, cited_by_count) = query_scopus_authors_cited(doi)
    else:
        (full_authors, cited_by_count) = query_scopus_authors_cited(doi)
    df.at[index, 'Full Authors'] = full_authors
    df.at[index, 'Cited By Count'] = cited_by_count

def concatenate_csv_files(folder_path, output_file):
    # List to hold all dataframes
    all_dataframes = []
    # Iterate over all CSV files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            # Read the CSV file
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            
            # Extract the year from the filename
            year = filename.split('.')[0]
            
            # Add the new column for filenames formatted as YYYY/NN_main_body.txt
            df['filename'] = df.apply(lambda row: f"{year}\\{int(row['Unnamed: 0'])}_main_body.txt", axis=1)
            
            # Append the dataframe to the list
            all_dataframes.append(df)
    # Concatenate all dataframes
    final_df = pd.concat(all_dataframes, ignore_index=True)

    # Save the combined dataframe to a new CSV file
    final_df.to_csv(output_file, index=False)

    return final_df

# Define the folder path and output file path
folder_path = 'Paper_CSV'
output_file = 'Titles_and_Filenames.csv'

# Call the function
combined_df = concatenate_csv_files(folder_path, output_file)

lda_topics_documents = pd.read_csv('LDA_topics_documents.csv')
combined_topic_data = pd.read_csv('Titles_and_Filenames.csv')
full_author_results = pd.read_csv('full_author_results.csv', encoding='ISO-8859-1')

# Merge LDA_topics_documents with combined_topic_data based on document and filename
merged_lda = pd.merge(lda_topics_documents, combined_topic_data, left_on='document', right_on='filename', how='left')

combined_topic_data.rename(columns={'titles': 'Original Title'}, inplace=True)
merged_data = pd.merge(full_author_results, combined_topic_data, on='Original Title', how='left')

augmented_data = pd.merge(merged_data, lda_topics_documents, left_on='filename', right_on='document', how='left')

topic_distributions = augmented_data[['Original Title', 'distr']]

# Merge this new dataframe back with full_author_results to preserve all rows
final_full_author_results = pd.merge(full_author_results, topic_distributions, on='Original Title', how='left')
