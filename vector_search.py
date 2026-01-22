"""
Vector Embeddings for Semantic Product Search

This module adds semantic search capabilities to the skincare chatbot.
Users can search by meaning (e.g., "something for oily skin") rather than exact product names.

Requirements (add to requirements.txt):
    sentence-transformers
    numpy
    faiss-cpu  
"""

import sqlite3
import numpy as np
from typing import List, Dict, Union
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

# Path to the SQLite database
DB_PATH = "skincare.sqlite"
# Path to save the vector index
INDEX_PATH = "product_vectors.index"
MAPPING_PATH = "product_mapping.pkl"


class ProductVectorSearch:
    """
    A semantic search engine for skincare products using vector embeddings.
    
    How it works:
    1. Each product's name + description is converted to a vector (embedding)
    2. When a user searches, their query is also converted to a vector
    3. We find products whose vectors are most similar to the query vector
    4. This allows meaning-based search, not just keyword matching
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector search engine.
        
        Args:
            model_name: The sentence-transformer model to use.
                       "all-MiniLM-L6-v2" is fast and good for semantic similarity.
                       Alternatives: "all-mpnet-base-v2" (better but slower)
        """
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.product_ids = []  # Maps index position -> product_id
        self.dimension = 384   # Embedding dimension for MiniLM
        
    def _get_products_from_db(self) -> List[Dict]:
        """Fetch all products from the database."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT product_id, product_name, description, category, stock, price
            FROM products
        """)
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "product_id": row[0],
                "product_name": row[1],
                "description": row[2],
                "category": row[3],
                "stock": row[4],
                "price": row[5]
            }
            for row in rows
        ]
    
    def build_index(self):
        """
        Build the vector index from all products in the database.
        
        This creates embeddings for each product and stores them in a FAISS index
        for fast similarity search.
        """
        products = self._get_products_from_db()
        
        if not products:
            raise ValueError("No products found in database. Run setup.py first.")
        
        print(f"Building vector index for {len(products)} products...")
        
        # Create text representation for each product (combine name + description + category)
        texts = [
            f"{p['product_name']} - {p['category']} - {p['description']}"
            for p in products
        ]
        
        # Generate embeddings for all products
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index (using Inner Product for cosine similarity on normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        
        # Store mapping from index position to product_id
        self.product_ids = [p['product_id'] for p in products]
        
        # Save the index and mapping to disk
        faiss.write_index(self.index, INDEX_PATH)
        with open(MAPPING_PATH, 'wb') as f:
            pickle.dump(self.product_ids, f)
        
        print(f"Vector index built and saved! ({len(products)} products indexed)")
        
    def load_index(self):
        """Load a previously built index from disk."""
        if not os.path.exists(INDEX_PATH) or not os.path.exists(MAPPING_PATH):
            print("No existing index found. Building new index...")
            self.build_index()
            return
            
        self.index = faiss.read_index(INDEX_PATH)
        with open(MAPPING_PATH, 'rb') as f:
            self.product_ids = pickle.load(f)
        print(f"Loaded existing index with {len(self.product_ids)} products")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search for products semantically similar to the query.
        
        Args:
            query: Natural language search query (e.g., "something for dry skin")
            top_k: Number of results to return
            
        Returns:
            List of matching products with similarity scores
        """
        if self.index is None:
            self.load_index()
        
        # Convert query to embedding
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search the index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Fetch full product details from database
        matching_product_ids = [self.product_ids[i] for i in indices[0]]
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        results = []
        for idx, product_id in enumerate(matching_product_ids):
            cursor.execute("""
                SELECT product_id, product_name, description, category, stock, price
                FROM products WHERE product_id = ?
            """, (product_id,))
            row = cursor.fetchone()
            
            if row:
                results.append({
                    "product_id": row[0],
                    "product_name": row[1],
                    "description": row[2],
                    "category": row[3],
                    "stock": row[4],
                    "price": row[5],
                    "similarity_score": float(scores[0][idx])  # How similar (0-1)
                })
        
        conn.close()
        return results


# ============================================================================
# LangChain Tool Integration
# ============================================================================

# Global instance (initialized once)
_vector_search = None

def get_vector_search() -> ProductVectorSearch:
    """Get or create the vector search instance."""
    global _vector_search
    if _vector_search is None:
        _vector_search = ProductVectorSearch()
        _vector_search.load_index()
    return _vector_search


# This is the tool you would add to tools.py
from langchain_core.tools import tool
from typing import Optional, Any


def _parse_price(value: Any) -> Optional[float]:
    """Convert various inputs to float or None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ('null', 'none', ''):
            return None
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _parse_category(value: Any) -> Optional[str]:
    """Convert category input to string or None."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ('null', 'none', ''):
            return None
        return value
    return None


@tool
def semantic_product_search(
    query: str,
    max_price: Optional[Any] = None,
    min_price: Optional[Any] = None,
    category: Optional[Any] = None
) -> Union[List[Dict], Dict]:
    """
    Search for skincare products by description. Only use when user asks about products.
    
    Args:
        query: What the user is looking for (e.g., "dry skin", "acne treatment")
        max_price: Maximum price if user specifies budget (e.g., 25 for "under $25"). Omit if no budget specified.
        min_price: Minimum price if specified. Omit if not needed.
        category: Optional category filter. Omit if not needed.
    """
    try:
        # Parse and clean parameters (handle string 'null' values from LLM)
        max_price_val = _parse_price(max_price)
        min_price_val = _parse_price(min_price)
        category_val = _parse_category(category)
        
        search_engine = get_vector_search()
        # Get more results initially to allow for filtering
        results = search_engine.search(query, top_k=10)
        
        if not results:
            return {"message": "No matching products found. Try describing your skin concern differently."}
        
        # Apply price filters
        if max_price_val is not None:
            results = [r for r in results if r["price"] <= max_price_val]
        if min_price_val is not None:
            results = [r for r in results if r["price"] >= min_price_val]
        
        # Apply category filter
        if category_val is not None:
            results = [r for r in results if r["category"].lower() == category_val.lower()]
        
        # Return top 3 after filtering
        results = results[:3]
        
        if not results:
            filters_applied = []
            if max_price_val is not None:
                filters_applied.append(f"under ${max_price_val}")
            if min_price_val is not None:
                filters_applied.append(f"over ${min_price_val}")
            if category_val is not None:
                filters_applied.append(f"in {category_val}")
            filter_str = ", ".join(filters_applied) if filters_applied else "your criteria"
            return {"message": f"No products found matching '{query}' {filter_str}. Try adjusting your filters or search terms."}
        
        return results
    except Exception as e:
        return {"message": f"Search error: {str(e)}"}


# ============================================================================
# Demo / Testing
# ============================================================================

if __name__ == "__main__":
    # Demo the semantic search
    print("=" * 60)
    print("Skincare Product Semantic Search Demo")
    print("=" * 60)
    
    # Initialize and build index
    search = ProductVectorSearch()
    search.build_index()
    
    # Test queries that show semantic understanding
    test_queries = [
        "something for dry skin",
        "I have acne problems",
        "anti-aging products",
        "sun protection",
        "gentle face wash",
        "hydrating serum for winter",
    ]
    
    print("\n" + "=" * 60)
    print("Testing Semantic Search")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n🔍 Query: \"{query}\"")
        print("-" * 40)
        results = search.search(query, top_k=2)
        
        for i, product in enumerate(results, 1):
            print(f"  {i}. {product['product_name']} ({product['category']})")
            print(f"     Score: {product['similarity_score']:.3f}")
            print(f"     ${product['price']:.2f}")
