from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


class PubMedError(RuntimeError):
    """Raised when NCBI E-utilities cannot be queried."""


def _ncbi_params(extra: dict[str, Any]) -> dict[str, Any]:
    params = {
        "tool": "SciLens",
        "email": os.getenv("NCBI_EMAIL", "scilens@example.com"),
        **extra,
    }
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    return params


def search_pubmed(query: str, max_results: int = 20, session: requests.Session | None = None) -> list[str]:
    """Return PubMed IDs for a free-text or PubMed-syntax query."""
    client = session or requests.Session()
    response = client.get(
        ESEARCH_URL,
        params=_ncbi_params(
            {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "sort": "relevance",
            }
        ),
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    try:
        return list(payload["esearchresult"]["idlist"])
    except (KeyError, TypeError) as exc:
        raise PubMedError("Unexpected PubMed search response") from exc


def fetch_abstracts(id_list: list[str], session: requests.Session | None = None) -> str:
    """Fetch PubMed XML for a list of PMIDs."""
    if not id_list:
        return ""
    client = session or requests.Session()
    if not os.getenv("NCBI_API_KEY"):
        time.sleep(0.34)
    response = client.get(
        EFETCH_URL,
        params=_ncbi_params(
            {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "xml",
            }
        ),
        timeout=60,
    )
    response.raise_for_status()
    return response.text


def _text(tag) -> str:
    return tag.get_text(" ", strip=True) if tag else ""


def parse_articles(xml_text: str) -> pd.DataFrame:
    """Parse PubMed XML into a dataframe of papers."""
    if not xml_text.strip():
        return pd.DataFrame(columns=["pmid", "title", "abstract", "year", "journal", "url"])

    soup = BeautifulSoup(xml_text, "xml")
    articles: list[dict[str, str]] = []

    for article in soup.find_all("PubmedArticle"):
        pmid = _text(article.find("PMID"))
        title = _text(article.find("ArticleTitle"))
        abstract = " ".join(_text(tag) for tag in article.find_all("AbstractText"))
        year_tag = article.find("PubDate")
        year = _text(year_tag.find("Year")) if year_tag else ""
        journal_tag = article.find("Journal")
        journal = ""
        if journal_tag:
            journal = _text(journal_tag.find("ISOAbbreviation")) or _text(journal_tag.find("Title"))
        articles.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "year": year,
                "journal": journal,
                "url": PUBMED_URL.format(pmid=pmid) if pmid else "",
            }
        )

    return pd.DataFrame(articles)


def retrieve_papers(query: str, max_results: int = 20, session: requests.Session | None = None) -> pd.DataFrame:
    """Search PubMed and return parsed articles with abstracts when available."""
    ids = search_pubmed(query, max_results=max_results, session=session)
    xml_text = fetch_abstracts(ids, session=session)
    papers = parse_articles(xml_text)
    if papers.empty:
        return papers
    has_abstract = papers["abstract"].fillna("").str.len() > 40
    return papers.loc[has_abstract].reset_index(drop=True)
