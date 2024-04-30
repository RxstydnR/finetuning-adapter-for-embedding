import os
import numpy as np
from langchain.vectorstores import FAISS

def make_faiss_db(documents,embedding_model,save_name):
    assert not os.path.isdir(save_name),f"{save_name} already exists."
    faiss_db = FAISS.from_documents(documents, embedding_model)
    faiss_db.save_local(save_name)
    return faiss_db


def get_vectors_from_FaissDB(db):
    """
    Retrieve vectors from a Faiss index.

    Args:
    - db (faiss.Index): Faiss index to retrieve vectors from.

    Returns:
    - np.array: Array of vectors retrieved from the Faiss index.
    """
    n_data = db.index.ntotal
    vecs=[]
    for i in range(n_data):
        vec = db.index.reconstruct_n(i,1)[0]
        vecs.append(vec)
    return np.array(vecs)


def replace_vectors_from_FaissDB(db, vecs):
    """
    Replace vectors in a Faiss index with new vectors.

    Args:
    - db (faiss.Index): Faiss index to replace vectors in.
    - vecs (np.array): Array of new vectors to replace existing vectors in the Faiss index with.

    Returns:
    - faiss.Index: Faiss index with replaced vectors.
    """
    n_data = db.index.ntotal
    for i in range(n_data):
        db.index.reconstruct_n(i,1)[0] = vecs[i]
    return db