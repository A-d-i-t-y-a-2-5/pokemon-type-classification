from app.db.base import VectorDatabase
from app.db.qdrant import QdrantVectorDatabase
from app.models.clip import vectorize_images, vectorize_text
from app.rag.constants import VectorDatabaseType


class VectorService:
    def __init__(self, db: VectorDatabase):
        self.db = db

    def vectorize_query(self, query: str) -> list[float]:
        features = vectorize_text(query)
        return features.tolist()[0]

    def insert_image(self, filename: str, vector: list[float]) -> None:
        self.db.insert(id=filename, vector=vector, metadata={"filename": filename})

    def insert_images(
        self, filenames: list[str | int], vectors: list[list[float]]
    ) -> None:
        metadata = [{"filename": f} for f in filenames]
        ids = list(range(len(filenames)))
        self.db.insert_many(ids=ids, vectors=vectors, metadata=metadata)

    def process_images(self, filenames: list[str], root_dir: str = "uploads") -> None:
        features = vectorize_images(filenames, root_dir)
        self.insert_images(
            filenames=filenames,
            vectors=features.tolist(),
        )

    def search_similar(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        return self.db.search(query_vector=query_vector, top_k=top_k)

    def clear_all(self) -> None:
        self.db.clear()


class VectorServiceFactory:
    @staticmethod
    def create(
        db_type: VectorDatabaseType,
        collection_name: str,
        vector_size: int = 512,
        **kwargs,
    ) -> VectorService:
        if db_type == VectorDatabaseType.QDRANT:
            db = QdrantVectorDatabase(
                collection_name=collection_name,
                host=kwargs.get("host", "localhost"),
                port=kwargs.get("port", 6333),
            )
            db.create(vector_size=vector_size)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

        return VectorService(db=db)


if __name__ == "__main__":
    vector_service = VectorServiceFactory.create(
        db_type=VectorDatabaseType.QDRANT,
        collection_name="images",
        vector_size=512,
        host="localhost",
        port=6333,
    )
