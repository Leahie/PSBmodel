import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import pandas as pd
from tqdm import tqdm

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

def extract_doi(data, year):
    # Extract the top two results
    items = data['message']['items']
    
    # Find the DOI that starts with 10.1142
    for item in items:
        doi = item['DOI']
        if doi.startswith('10.1142') and item["container-title"] == [f"Biocomputing {(int(year))}"]:
            return doi
    return None

data = []
for title, details in paper_meta.items():
    authors, year = details
    data.append({'Title': title, 'Authors': authors, 'Year': year})

df = pd.DataFrame(data)