import asyncio
import os
from functools import partial
from typing import Any

import bm25s
import lancedb
import litellm
import numpy as np
import Stemmer
import weave
from lancedb.index import FTS
from litellm import aembedding
from litellm.caching.caching import Cache
from pydantic import PrivateAttr
from rerankers import Reranker as AnsReranker
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer

litellm.cache = Cache(disk_cache_dir="data/cache")


class TfidfSearchEngine:
    """
    A retriever model that uses TF-IDF for indexing and searching documents.

    Attributes:
        vectorizer (TfidfVectorizer): The TF-IDF vectorizer.
        index (list): The indexed data.
        data (list): The data to be indexed.
    """

    def __init__(self):
        self._vectorizer = TfidfVectorizer()
        self._index = None
        self._data = None

    async def fit(self, data):
        """
        Indexes the provided data using TF-IDF.

        Args:
            data (list): A list of documents to be indexed. Each document should be a dictionary
                         containing a key 'cleaned_content' with the text to be indexed.
        """
        # store the original data as is so we can use it for retrieval later
        self._data = data
        # just extract the text from the documents
        docs = [doc["text"] for doc in self._data]
        # our instance is simply an instance of TfidfVectorizer from scikit-learn
        # we use the fit_transform method to vectorize the documents
        self._index = self._vectorizer.fit_transform(docs)
        return self

    async def search(
        self, query: str, top_k: int = 5, **kwargs
    ) -> list[dict[str, Any]]:
        """
        Searches the indexed data for the given query using cosine similarity.

        Args:
            query (str): The search query.
            top_k (int): The number of top results to return. Default is 5.

        Returns:
            list: A list of dictionaries containing the source, text, and score of the top-k results.
        """
        assert self._index is not None, "Index is not set"
        assert self._data is not None, "Data is not set"
        # vectorize the query
        query_vec = self._vectorizer.transform([query])
        # compute the cosine distance between the query vector and the indexed vectors
        cosine_distances = cdist(
            query_vec.todense(), self._index.todense(), metric="cosine"
        )[0]
        # get the top-k indices of the indexed vectors that are most similar to the query vector
        top_k_indices = cosine_distances.argsort()[:top_k]
        # create the output list of dictionaries
        output = []
        # iterate over the top-k indices and append the corresponding document to the output list
        for idx in top_k_indices:
            output.append(
                {"score": round(float(1 - cosine_distances[idx]), 4), **self._data[idx]}
            )
        # return the output list of dictionaries
        return output


class BM25SearchEngine:
    """
    A retriever model that uses BM25 for indexing and searching documents.

    Attributes:
        index (bm25s.BM25): The BM25 index.
        _stemmer (Stemmer.Stemmer): The stemmer.
        data (list): The data to be indexed.
    """

    def __init__(self):
        self._index = bm25s.BM25()
        self._stemmer = Stemmer.Stemmer("english")
        self._data = None

    async def fit(self, data):
        """
        Indexes the provided data using BM25.

        Args:
            data (list): A list of documents to be indexed. Each document should be a dictionary
                         containing a key 'cleaned_content' with the text to be indexed.
        """
        # store the original data as is so we can use it for retrieval later
        self._data = data
        # just extract the text from the documents
        corpus = [doc["text"] for doc in data]
        # our instance is simply an instance of bm25s.BM25 from the bm25s library
        # we use the tokenize method to tokenize the documents
        # we use the index method to index the documents
        corpus_tokens = bm25s.tokenize(corpus, show_progress=False, stopwords="english")
        # index the documents and store the corpus tokens in the index
        self._index.index(corpus_tokens, show_progress=False)
        return self

    async def search(self, query, top_k=5, **kwargs):
        """
        Searches the indexed data for the given query using BM25.

        Args:
            query (str): The search query.
            top_k (int): The number of top results to return. Default is 5.

        Returns:
            list: A list of dictionaries containing the source, text, and score of the top-k results.
        """
        assert self._index is not None, "Index is not set"
        assert self._data is not None, "Data is not set"
        # tokenize the query
        query_tokens = bm25s.tokenize(query, show_progress=False, stopwords="english")
        # get the top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k)
        results, scores = self._index.retrieve(
            query_tokens, corpus=self._data, k=top_k, show_progress=False
        )

        output = []
        for idx in range(results.shape[1]):
            output.append(
                {
                    "score": round(float(scores[0, idx]), 4),
                    **results[0, idx],
                }
            )
        return output


async def batch_embed(
    vectorizer, docs, input_type=None, dimensions=None, batch_size=50
):
    if input_type is None:
        _vectorizer = vectorizer
    else:
        _vectorizer = partial(vectorizer, input_type=input_type)
    if dimensions is None:
        __vectorizer = _vectorizer
    else:
        __vectorizer = partial(_vectorizer, dimensions=dimensions)
    all_embeddings = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        embeddings = await __vectorizer(input=batch)
        all_embeddings.append([embedding["embedding"] for embedding in embeddings.data])
    return np.concatenate(all_embeddings, axis=0)


class DenseSearchEngine:
    """
    A retriever model that uses dense embeddings for indexing and searching documents.

    Attributes:
        vectorizer (Callable): The function used to generate embeddings.
        index (np.ndarray): The indexed embeddings.
        data (list): The data to be indexed.
    """

    def __init__(self, model="text-embedding-3-small"):
        if "embed-" in model.lower():
            self.is_cohere_model = True
        else:
            self.is_cohere_model = False
        self._vectorizer = partial(aembedding, model=model, caching=True)
        self._index = None
        self._data = None

    async def fit(self, data):
        """
        Indexes the provided data using dense embeddings.

        Args:
            data (list): A list of documents to be indexed. Each document should be a dictionary
                         containing a key 'cleaned_content' with the text to be indexed.
        """
        self._data = data
        docs = [doc["text"] for doc in data]
        if self.is_cohere_model:
            embeddings = await batch_embed(
                self._vectorizer, docs, input_type="search_document"
            )
        else:
            embeddings = await batch_embed(self._vectorizer, docs, dimensions=512)
        self._index = np.array(embeddings)
        return self

    async def search(self, query, top_k=5, **kwargs):
        """
        Searches the indexed data for the given query using cosine similarity.

        Args:
            query (str): The search query.
            top_k (int): The number of top results to return. Default is 5.

        Returns:
            list: A list of dictionaries containing the source, text, and score of the top-k results.
        """
        if self.is_cohere_model:
            query_embedding = await batch_embed(
                self._vectorizer,
                [query],
                input_type="search_query",
            )
        else:
            query_embedding = await batch_embed(
                self._vectorizer,
                [query],
                dimensions=512,
                batch_size=500,
            )
        cosine_distances = cdist(query_embedding, self._index, metric="cosine")[0]
        top_k_indices = cosine_distances.argsort()[:top_k]
        output = []
        for idx in top_k_indices:
            output.append(
                {
                    "score": round(float(1 - cosine_distances[idx]), 4),
                    **self._data[idx],
                }
            )
        return output


def make_batches_for_db(
    vectorizer, data, batch_size=50, is_cohere_model=False, dimensions=512
):
    for i in range(0, len(data), batch_size):
        batch_docs = data[i : i + batch_size]
        batch_texts = [doc["text"] for doc in batch_docs]
        if is_cohere_model:
            embeddings = asyncio.run(
                batch_embed(vectorizer, batch_texts, input_type="search_document")
            )
        else:
            embeddings = asyncio.run(
                batch_embed(
                    vectorizer,
                    batch_texts,
                    dimensions=dimensions,
                    batch_size=500,
                )
            )
        yield [
            {"vector": embedding, **doc}
            for embedding, doc in zip(embeddings, batch_docs)
        ]


class VectorStoreSearchEngine:
    def __init__(
        self,
        uri="data/docsdb",
        embedding_model="text-embedding-3-small",
    ):
        if "embed-" in embedding_model.lower():
            self.is_cohere_model = True
        else:
            self.is_cohere_model = False
        self._vectorizer = partial(aembedding, model=embedding_model, caching=True)
        self._uri = uri
        self._db = None
        self._table = None

    async def fit(self, data):
        self._db = await lancedb.connect_async(self._uri)
        self._table = await self._db.create_table(
            "docs",
            data=make_batches_for_db(
                self._vectorizer, data, is_cohere_model=self.is_cohere_model
            ),
            mode="overwrite",
            exist_ok=True,
        )
        await self._table.create_index("text", config=FTS())
        return self

    async def load(self):
        self._db = await lancedb.connect_async(self._uri)
        self._table = await self._db.open_table("docs")
        return self

    async def search(self, query, top_k=5, filters=None):
        if self.is_cohere_model:
            query_embedding = await batch_embed(
                self._vectorizer,
                [query],
                input_type="search_query",
            )
        else:
            query_embedding = await batch_embed(
                self._vectorizer,
                [query],
                dimensions=512,
            )
        if filters is not None:
            results = (
                await self._table.query()
                .nearest_to(query_embedding[0])
                .nearest_to_text(query)
                .where(filters)
                .limit(top_k)
                .to_list()
            )
        else:
            results = (
                await self._table.query()
                .nearest_to(query_embedding[0])
                .nearest_to_text(query)
                .limit(top_k)
                .to_list()
            )
        results = [
            {k: v for k, v in result.items() if k != "vector"} for result in results
        ]
        return results


class Retriever(weave.Model):
    search_engine: Any

    @weave.op
    async def invoke(self, query: str, top_k: int, **kwargs) -> list[dict[str, Any]]:
        return await self.search_engine.search(query, top_k, **kwargs)


class Reranker(weave.Model):
    model: str = "gpt-4o-mini"
    _reranker: Any = PrivateAttr()

    def model_post_init(self, _context) -> None:
        self._reranker = AnsReranker(
            model_name=self.model,
            model_type="rankgpt",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    @weave.op
    async def invoke(self, query: str, documents: list[dict[str, Any]], top_n: int = 5):
        try:
            reranked_docs = await self._reranker.rank_async(
                query=query, docs=[doc["text"] for doc in documents]
            )
            reranked_docs = reranked_docs.top_k(top_n)
            output_docs = []
            for result in reranked_docs:
                reranked_doc = documents[result.document.doc_id]
                if result.score is not None:
                    reranked_doc["score"] = round(float(result.score), 4)
                output_docs.append(reranked_doc)
            return output_docs
        except Exception as e:
            if len(documents) < top_n:
                return documents
            else:
                return documents[:top_n]


class RetrieverWithReranker(Retriever):
    """
    A retriever model that uses dense embeddings for retrieval and a reranker for re-ranking the results.

    Attributes:
        retriever (DenseRetriever): The dense retriever model.
        reranker (Reranker): The reranker model.
    """

    search_engine: Any
    reranker: Any = Reranker()

    @weave.op
    async def invoke(self, query: str, top_k: int = None, top_n: int = None, **kwargs):
        """
        Predicts the top-n results for the given query after re-ranking.

        Args:
            query (str): The search query.
            top_k (int, optional): The number of top results to retrieve before re-ranking. Default is None.
            top_n (int, optional): The number of top results to return after re-ranking. Default is None.

        Returns:
            list: A list of dictionaries containing the source, text, and score of the top-n results.
        """
        if top_k and not top_n:
            top_n = top_k
            top_k = top_k * 2
        elif top_n and not top_k:
            top_k = top_n * 2
        elif top_k == top_n:
            top_k = top_k * 2
        else:
            top_k = 10
            top_n = 5
        retrievals = await self.search_engine.search(query, top_k, **kwargs)
        reranked = await self.reranker.invoke(query, retrievals, top_n)
        return reranked


class HybridRetrieverWithReranker(weave.Model):
    sparse_search_engine: Any
    dense_search_engine: Any
    reranker: Any = Reranker()

    @weave.op
    async def invoke(self, query: str, top_k: int = None, top_n: int = None, **kwargs):
        if top_k and not top_n:
            top_n = top_k
            top_k = top_k * 2
        elif top_n and not top_k:
            top_k = top_n * 2
        elif top_k == top_n:
            top_k = top_k * 2
        else:
            top_k = 10
            top_n = 5

        sparse_retrievals = await self.sparse_search_engine.search(
            query, top_k, **kwargs
        )
        dense_retrievals = await self.dense_search_engine.search(query, top_k, **kwargs)
        retrievals = sparse_retrievals + dense_retrievals
        deduped_retrievals = {}
        for doc in retrievals:
            deduped_retrievals[doc["chunk_id"]] = doc
        deduped_retrievals = list(deduped_retrievals.values())
        reranked = await self.reranker.invoke(query, deduped_retrievals, top_n)
        return reranked
