from scilens.classifier import normalize_label
from scilens.pubmed import parse_articles
from scilens.retriever import filter_relevant, rank_papers


class FakeEmbedder:
    """Deterministic bag-of-words vectors so tests never download MiniLM."""

    vocab = ["creatine", "cognition", "memory", "muscle", "dystrophy", "gene"]

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([1.0 if token in lowered else 0.0 for token in self.vocab])
        return vectors


SAMPLE_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>1</PMID>
      <Article>
        <Journal><Title>Brain Res</Title><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
        <ArticleTitle>Creatine supplementation and memory in adults</ArticleTitle>
        <Abstract>
          <AbstractText>Creatine improved working memory and cognition in a randomized trial of healthy adults.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>2</PMID>
      <Article>
        <Journal><ISOAbbreviation>J Strength</ISOAbbreviation><JournalIssue><PubDate><Year>2023</Year></PubDate></JournalIssue></Journal>
        <ArticleTitle>Gene therapy for muscular dystrophy</ArticleTitle>
        <Abstract>
          <AbstractText>Micro-utrophin gene therapy restored muscle function in a mouse model of dystrophy.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


def test_parse_articles_extracts_fields():
    df = parse_articles(SAMPLE_XML)
    assert list(df["pmid"]) == ["1", "2"]
    assert "Creatine" in df.iloc[0]["title"]
    assert df.iloc[0]["year"] == "2024"
    assert df.iloc[1]["journal"] == "J Strength"
    assert "pubmed.ncbi.nlm.nih.gov/1" in df.iloc[0]["url"]


def test_rank_and_filter_drops_off_topic_papers():
    papers = parse_articles(SAMPLE_XML)
    ranked = rank_papers("Does creatine improve cognition?", papers, embedder=FakeEmbedder())
    assert ranked.iloc[0]["pmid"] == "1"
    relevant = filter_relevant(ranked, min_similarity=0.4)
    assert list(relevant["pmid"]) == ["1"]


def test_normalize_label():
    assert normalize_label("supports: creatine improves cognition") == "Supports"
    assert normalize_label("Contradicts") == "Contradicts"
    assert normalize_label("mixed evidence regarding: x") == "Mixed"
