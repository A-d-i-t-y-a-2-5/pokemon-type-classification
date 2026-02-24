from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter
from app.db.base import VectorDatabase

class QdrantVectorDatabase(VectorDatabase):

    def __init__(
        self,
        collection_name: str,
        vector_size: int = 512,
        host: str = "localhost",
        port: int = 6333,
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        if not self.client.collection_exists(collection_name=collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def insert(
        self, id: str, vector: list[float], metadata: dict | None = None
    ) -> None:
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=id, vector=vector, payload=metadata or {})],
        )

    def insert_many(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadata: list[dict] | None = None,
    ) -> None:
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(id=id, vector=vector, payload=meta)
                for id, vector, meta in zip(ids, vectors, metadata or [{}] * len(ids))
            ],
        )

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        return results.points

    def delete(self, id) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=[id],
        )

    def get(self, id: str) -> dict | None:
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[id],
            with_vectors=True,
            with_payload=True,
        )
        if not results:
            return None
        result = results[0]
        return {
            "id": result.id,
            "vector": result.vector,
            "metadata": result.payload,
        }

    def count(self) -> int:
        return self.client.count(collection_name=self.collection_name).count

    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)