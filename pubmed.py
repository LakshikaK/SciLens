import requests
import requests
import pandas as pd
from bs4 import BeautifulSoup

def search_pubmed(query, max_results=10):

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json"
    }

    response = requests.get(url, params=params)

    data = response.json()

    return data["esearchresult"]["idlist"]

def fetch_abstracts(id_list):

    ids = ",".join(id_list)

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    params = {
        "db": "pubmed",
        "id": ids,
        "retmode": "xml"
    }

    response = requests.get(url, params=params)

    return response.text
def parse_articles(xml_text):

    soup = BeautifulSoup(xml_text, "xml")

    articles = []

    for article in soup.find_all("PubmedArticle"):

        pmid_tag = article.find("PMID")
        title_tag = article.find("ArticleTitle")

        abstract_tags = article.find_all("AbstractText")

        pmid = pmid_tag.text if pmid_tag else ""

        title = title_tag.text if title_tag else ""

        abstract = " ".join(
            tag.text for tag in abstract_tags
        )

        articles.append({
            "PMID": pmid,
            "Title": title,
            "Abstract": abstract
        })

    return pd.DataFrame(articles)

if __name__ == "__main__":

    ids = search_pubmed(
        "creatine cognition",
        10
    )

    xml_data = fetch_abstracts(ids)

    papers_df = parse_articles(xml_data)

    print(papers_df.head())
    
papers_df.to_csv(
    "data/papers.csv",
    index=False
)

print("Saved!")