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


def csv_to_dict(csv_file, year):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    if year == 1996:
        if 'Title' in df.columns and ('Author' in df.columns):
            result_dict = {row['Title']: (row['Author'], year) for index, row in df.iterrows()}
            return result_dict
        else:
            raise ValueError("CSV file must contain 'title', 'author', and 'year' columns")
    elif year == 1997:
        if 'Title' in df.columns and ('Authors' in df.columns):
            result_dict = {row['Title']: (row['Authors'], year) for index, row in df.iterrows()}
            return result_dict
        else:
            raise ValueError("CSV file must contain 'title', 'author', and 'year' columns")
    else:
        raise ValueError("year must be 1996 or 1997")

paper_meta = csv_to_dict('1996.csv',1996)
paper_meta.update(csv_to_dict('1997.csv',1997))

def fetch_html(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {str(e)}")
        return None

def clean_author(author_text):
    author_text = re.sub(r'\s+', ' ', author_text)  # Replace multiple spaces with a single space
    author_text = author_text.replace('\n', ' ')  # Replace newlines with a space
    return author_text

def clean_title(title):
    # Remove excess whitespace and line breaks
    title = re.sub(r'\s+', ' ', title)  # Replace multiple spaces with a single space
    title = title.replace('\n', ' ')  # Replace newlines with a space
    title = title.strip()  # Remove leading/trailing whitespace
    return title
def extract_data(html, year, entries):
    if not html:
        return {}
    soup = BeautifulSoup(html, 'html.parser')
    tags = soup.find_all(['p', 'dt', 'h3'])
    
    skip_section = False

    for tag in tags:
        if tag.name == 'h3':
            if 'workshops' in tag.text.lower():
                skip_section = True
                continue  # Skip the workshops section header itself
        
        if skip_section:
            continue  # Skip all tags when in the Workshops section
        
        if tag.name in ['p', 'dt']:
            a_tag = tag.find('a', href=lambda href: href and href.lower().endswith('.pdf'))
            if a_tag:
                title = clean_title(a_tag.text.strip())
                if title not in ["Preface", "Session Introduction", "Introduction"] and not title.endswith("introduction"):
                    author_tag = tag.find('i') or tag.find('b') if tag.name == 'p' else tag.find_next_sibling('dd')
                    if author_tag:
                        author_text = author_tag.text.strip().split(';')[0]
                        if title not in entries:
                            entries[title] = (clean_author(author_text), year)
                    
    return entries

def extract_data_2002(html, year, entries):
    if not html:
        return {}
    soup = BeautifulSoup(html, 'html.parser')

    # Find all dt and dd tags
    dt_tags = soup.find_all('dt')
    dd_tags = soup.find_all('dd')

    for dt, dd in zip(dt_tags, dd_tags):
        # Get the title, which is the text of the a tag within dt
        title_tag = dt.find('a')
        title = clean_title(title_tag.text.strip())
        if title != "Session Introduction":
            # Get the authors, which is the text of the i tag within dd
            authors_tag = dd.find('i')
            if authors_tag:
                author_text = authors_tag.text.strip().split(';')[0]
                authors = authors_tag.text.strip()
                entries[title] = (clean_author(author_text), year)
                    
    return entries

def collect_titles_authors_years(start_year, end_year):
    base_url = "http://psb.stanford.edu/psb-online/proceedings/psb"
    titles_authors_years = {}

    for year in range(start_year, end_year + 1):
        year_suffix = str(year)[-2:]
        url = f"{base_url}{year_suffix}/"
        html_content = fetch_html(url)
        if year == 2002:
            data = extract_data_2002(html_content, year, titles_authors_years)
        else:
            data = extract_data(html_content, year, titles_authors_years)

    return titles_authors_years

paper_meta.update(collect_titles_authors_years(1998, 2024))
for title, info in paper_meta.items():
    if info[1] == 2020:
        print(f"Title: {title}, First Author: {info[0]}, Year: {info[1]}")

def get_doi(title, author, year):
    url = "https://api.crossref.org/works"
    params = {
        "query.bibliographic": f"{title.replace(' ', '+')}+{author.replace(' ', '+')}+{year}",
        "rows": 3,
        "mailto": "sameeksha.garg@dartmouth.edu"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API request failed with status code {response.status_code}")

def extract_doi(data, year):
    # Extract the top two results
    items = data['message']['items']
    
    # Find the DOI that starts with 10.1142
    for item in items:
        doi = item['DOI']
        if doi.startswith('10.1142') and item["container-title"] == [f"Biocomputing {(int(year))}"]:
            return doi
    return None

for title, info in tqdm(paper_meta.items(), desc="Getting DOIs"):
    doi = extract_doi(get_doi(title, info[0], info[1]), info[1])
    paper_meta[title] = (info[0], info[1], doi)
#     print(f"DOI: {doi}")

data = []
for title, details in paper_meta.items():
    authors, year = details
    data.append({'Title': title, 'Authors': authors, 'Year': year})

df = pd.DataFrame(data)
os.environ['NCBI_API_KEY'] = 'temp'
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

# Print the results
display(new_results_df)

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
        
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(pubmed_data)

