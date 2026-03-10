import asyncio
import logging

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter
from app.db.base import VectorDatabase

# module‑level logger for diagnostics
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # set to DEBUG for detailed logs


class QdrantVectorDatabase(VectorDatabase):

    def __init__(
        self,
        collection_name: str,
        host: str = "localhost",
        port: int = 6333,
    ):
        logger.info(
            f"Initializing QdrantVectorDatabase for collection {collection_name}"
        )
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)

    def create(self, vector_size: int = 512) -> None:
        logger.info(
            f"Creating collection {self.collection_name} with vector size {vector_size}"
        )
        if not self.client.collection_exists(collection_name=self.collection_name):
            logger.info(f"Collection {self.collection_name} does not exist, creating")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created new collection {self.collection_name}")
        logger.info(f"Collection {self.collection_name} created successfully")

    def insert(
        self, id: str, vector: list[float], metadata: dict | None = None
    ) -> None:
        logger.info(f"Inserting point {id} into {self.collection_name}")
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=id, vector=vector, payload=metadata or {})],
        )
        logger.info(f"Inserted point {id}")

    def insert_many(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadata: list[dict] | None = None,
    ) -> None:
        logger.debug(f"Bulk inserting {len(ids)} points into {self.collection_name}")
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(id=id, vector=vector, payload=meta)
                for id, vector, meta in zip(ids, vectors, metadata or [{}] * len(ids))
            ],
        )
        logger.info(f"Inserted {len(ids)} points")

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        logger.info(
            f"Searching {self.collection_name} for top {top_k} similar vectors"
        )
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        logger.info(f"Search returned {len(results.points)} results")
        return results.points

    def delete(self, id) -> None:
        logger.debug(f"Deleting point {id} from {self.collection_name}")
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=[id],
        )
        logger.info(f"Deleted point {id}")
    def get(self, id: str) -> dict | None:
        logger.debug(f"Retrieving point {id} from {self.collection_name}")
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[id],
            with_vectors=True,
            with_payload=True,
        )
        if not results:
            logger.warning(f"Point {id} not found")
            return None
        result = results[0]
        logger.info(f"Retrieved point {id}")
        return {
            "id": result.id,
            "vector": result.vector,
            "metadata": result.payload,
        }

    def count(self) -> int:
        cnt = self.client.count(collection_name=self.collection_name).count
        logger.info(f"Collection {self.collection_name} has {cnt} points")
        return cnt

    def clear(self) -> None:
        logger.warning(f"Clearing collection {self.collection_name}")
        self.client.delete_collection(self.collection_name)
        logger.info(f"Collection {self.collection_name} deleted")


class AsyncQdrantVectorDatabase(VectorDatabase):

    def __init__(
        self,
        collection_name: str,
        host: str = "localhost",
        port: int = 6333,
    ):
        logger.info(
            f"Initializing AsyncQdrantVectorDatabase for collection {collection_name}"
        )
        self.collection_name = collection_name
        self.client = AsyncQdrantClient(host=host, port=port)

    async def create(self, vector_size: int) -> None:
        logger.info(
            f"Creating collection {self.collection_name} with vector size {vector_size}"
        )
        if not await self.client.collection_exists(collection_name=self.collection_name):
            logger.info(f"Collection {self.collection_name} does not exist, creating")
            return await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created new collection {self.collection_name}")
        logger.info(f"Collection {self.collection_name} created successfully")

    async def insert(
        self, id: str, vector: list[float], metadata: dict | None = None
    ) -> None:
        logger.info(f"Inserting point {id} into {self.collection_name}")
        await self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(id=id, vector=vector, payload=metadata or {})],
        )
        logger.info(f"Inserted point {id}")

    async def insert_many(
        self,
        ids: list[str],
        vectors: list[list[float]],
        metadata: list[dict] | None = None,
    ) -> None:
        logger.debug(f"Bulk inserting {len(ids)} points into {self.collection_name}")
        await self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(id=id, vector=vector, payload=meta)
                for id, vector, meta in zip(ids, vectors, metadata or [{}] * len(ids))
            ],
        )
        logger.info(f"Inserted {len(ids)} points")

    async def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        logger.info(
            f"Searching {self.collection_name} for top {top_k} similar vectors"
        )
        results = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        logger.info(f"Search returned {len(results.points)} results")
        return results.points

    async def delete(self, id) -> None:
        logger.debug(f"Deleting point {id} from {self.collection_name}")
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=[id],
        )
        logger.info(f"Deleted point {id}")

    async def get(self, id: str) -> dict | None:
        logger.debug(f"Retrieving point {id} from {self.collection_name}")
        results = await self.client.retrieve(
            collection_name=self.collection_name,
            ids=[id],
            with_vectors=True,
            with_payload=True,
        )
        if not results:
            logger.warning(f"Point {id} not found")
            return None
        result = results[0]
        logger.info(f"Retrieved point {id}")
        return {
            "id": result.id,
            "vector": result.vector,
            "metadata": result.payload,
        }

    async def count(self) -> int:
        cnt = (await self.client.count(collection_name=self.collection_name)).count
        logger.info(f"Collection {self.collection_name} has {cnt} points")
        return cnt

    async def clear(self) -> None:
        logger.warning(f"Clearing collection {self.collection_name}")
        await self.client.delete_collection(collection_name=self.collection_name)
        logger.info(f"Collection {self.collection_name} deleted")


if __name__ == "__main__":
    db = AsyncQdrantVectorDatabase(collection_name="temp")
    asyncio.run(db.create(vector_size=512))