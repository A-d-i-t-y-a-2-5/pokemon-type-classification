from app.db.base import VectorDatabase
from app.db.qdrant import AsyncQdrantVectorDatabase
from app.models.clip import vectorize_images, vectorize_text
from app.rag.constants import VectorDatabaseType

import asyncio

class AsyncVectorService:
    def __init__(self, db: VectorDatabase):
        self.db = db

    def vectorize_query(self, query: str) -> list[float]:
        features = vectorize_text(query)
        return features.tolist()[0]

    async def insert_image(self, filename: str, vector: list[float]) -> None:
        await self.db.insert(id=filename, vector=vector, metadata={"filename": filename})

    async def insert_images(
        self, filenames: list[str | int], vectors: list[list[float]]
    ) -> None:
        metadata = [{"filename": f} for f in filenames]
        ids = list(range(len(filenames)))
        await self.db.insert_many(ids=ids, vectors=vectors, metadata=metadata)

    async def process_images(self, filenames: list[str], root_dir: str = "uploads") -> None:
        loop = asyncio.get_event_loop()
        features = await loop.run_in_executor(None, vectorize_images, filenames, root_dir)
        await self.insert_images(
            filenames=filenames,
            vectors=features.tolist(),
        )

    async def search_similar(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        return await self.db.search(query_vector=query_vector, top_k=top_k)

    async def clear_all(self) -> None:
        await self.db.clear()


class AsyncVectorServiceFactory:
    @staticmethod
    async def create(
        db_type: VectorDatabaseType,
        collection_name: str,
        vector_size: int = 512,
        **kwargs,
    ) -> AsyncVectorService:
        if db_type == VectorDatabaseType.AQDRANT:
            db = AsyncQdrantVectorDatabase(
                collection_name=collection_name,
                host=kwargs.get("host", "localhost"),
                port=kwargs.get("port", 6333),
            )
            await db.create(vector_size=vector_size)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

        return AsyncVectorService(db=db)
