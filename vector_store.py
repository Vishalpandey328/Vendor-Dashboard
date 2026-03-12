import faiss
import numpy as np

def build_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    return index


def search_index(index, query_embedding):

    distances, indices = index.search(query_embedding, 1)

    return distances, indices