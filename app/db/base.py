from abc import ABC, abstractmethod

class VectorDatabase(ABC):

    @abstractmethod
    def insert(self, id: str, vector: list[float], metadata: dict | None = None) -> None:
        """Insert a single vector with an associated id and optional metadata."""
        pass

    @abstractmethod
    def insert_many(self, ids: list[str], vectors: list[list[float]], metadata: list[dict] | None = None) -> None:
        """Insert multiple vectors with associated ids and optional metadata."""
        pass

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        """Search for the top_k most similar vectors to the query vector."""
        pass

    @abstractmethod
    def delete(self, id: str) -> None:
        """Delete a vector by id."""
        pass

    @abstractmethod
    def get(self, id: str) -> dict | None:
        """Retrieve a vector and its metadata by id."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return the total number of vectors stored."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Delete all vectors from the database."""
        pass