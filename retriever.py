#goal is to find most relevant papers

#embedded model
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

#retrieval function
#user question->vector
#paper->vector
#compute similarity
#sort from most relev to least relev
def rank_papers(question, papers):

    question_embedding = model.encode(question)

    paper_texts = (
        papers["Title"].fillna("") +
        " " +
        papers["Abstract"].fillna("")
    )

    paper_embeddings = model.encode(
        paper_texts.tolist()
    )

    similarities = cosine_similarity(
        [question_embedding],
        paper_embeddings
    )[0]

    papers = papers.copy()
    papers["Similarity"] = similarities

    papers = papers.sort_values(
        "Similarity",
        ascending=False
    )

    return papers