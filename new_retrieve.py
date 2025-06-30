import os
import asyncio
from typing import List, Dict, Any
import numpy as np
from dotenv import load_dotenv

# Reuse the embedding functions from your existing code
from transformers import AutoTokenizer, AutoModel
import torch

# Supabase client
from supabase import create_client, Client

load_dotenv()

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Global variables for E5 model to avoid reloading
E5_MODEL = None
E5_TOKENIZER = None

def initialize_e5_model(model_size="small"):
    """Initialize the E5 model and tokenizer once."""
    global E5_MODEL, E5_TOKENIZER
    
    if E5_MODEL is None or E5_TOKENIZER is None:
        model_name = f"intfloat/e5-{model_size}-v2"
        
        print(f"Loading E5 model: {model_name}")
        E5_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        E5_MODEL = AutoModel.from_pretrained(model_name)
        
        # Move to CPU explicitly (helps manage memory)
        E5_MODEL = E5_MODEL.to("cpu")
        
        print("E5 model loaded successfully")
    
    return E5_MODEL, E5_TOKENIZER

def mean_pooling(token_embeddings, attention_mask):
    """Perform mean pooling on token embeddings using the attention mask."""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def resize_to_1536(embedding):
    """Resize embedding to 1536 dimensions."""
    original_dim = embedding.shape[0]
    target_dim = 1536
    
    if original_dim == target_dim:
        return embedding
    
    if original_dim < target_dim:
        # Pad by repeating values
        repetitions = target_dim // original_dim + 1
        repeated = np.tile(embedding, repetitions)
        return repeated[:target_dim]
    else:
        # Simple dimensionality reduction by taking every nth value
        indices = np.round(np.linspace(0, original_dim-1, target_dim)).astype(int)
        return embedding[indices]

async def get_embedding_local(text: str) -> List[float]:
    """Get embedding vector using local E5 model."""
    try:
        # Initialize model if not already loaded
        model, tokenizer = initialize_e5_model(model_size="small")
        
        # Create a coroutine to run the model inference
        def run_model():
            # Prepare input - E5 models expect "query: " prefix for search queries
            inputs = tokenizer(f"query: {text}", padding=True, truncation=True,
                              return_tensors="pt", max_length=512)
            
            # Free up memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Generate embedding with no gradient calculation
            with torch.no_grad():
                outputs = model(**inputs)
                
                # Use mean pooling for better representation
                attention_mask = inputs['attention_mask']
                embeddings = mean_pooling(outputs.last_hidden_state, attention_mask)
                
                # Normalize
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                # Convert to numpy and get first embedding
                embedding = embeddings.squeeze().cpu().numpy()
            
            # Resize to 1536 dimensions
            resized_embedding = resize_to_1536(embedding)
            return resized_embedding
        
        # Run model in the default thread pool
        embedding = await asyncio.to_thread(run_model)
        
        # Convert numpy array to list for storage
        return embedding.tolist()
    
    except Exception as e:
        print(f"Error getting embedding with E5: {e}")
        return [0] * 1536  # Return zero vector only on error

async def search_similar_cves(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search for similar CVEs based on a query string.
    
    Args:
        query: The search query (e.g., CVE ID or description)
        limit: Maximum number of results to return
        
    Returns:
        List of matching documents with similarity scores
    """
    try:
        # Get embedding for the query
        query_embedding = await get_embedding_local(query)
        
        # Perform similarity search using pgvector's cosine distance
        # Note: Supabase implements vector similarity search with the '.search()' method
        result = supabase.table("site_pages") \
            .select("url, title, summary, content, chunk_number, metadata") \
            .search("embedding", query_embedding, distance="cosine") \
            .limit(limit) \
            .execute()
            
        # Process and return results
        if result and hasattr(result, 'data'):
            # Add distance scores if available
            return result.data
        else:
            print("No results found or error in query")
            return []
            
    except Exception as e:
        print(f"Error performing similarity search: {str(e)}")
        return []

async def search_cve_by_id(cve_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search specifically for a CVE ID.
    This creates a more targeted query for CVE numbers.
    
    Args:
        cve_id: The CVE ID to search for (e.g., "CVE-2023-1234")
        limit: Maximum number of results to return
        
    Returns:
        List of matching documents with similarity scores
    """
    # Normalize the CVE ID format
    cve_id = cve_id.upper()
    if not cve_id.startswith("CVE-") and cve_id.startswith("CVE"):
        cve_id = f"CVE-{cve_id[3:]}"
        
    try:
        # Method 1: Try direct text search first (more precise for exact CVE IDs)
        direct_result = supabase.table("site_pages") \
            .select("url, title, summary, content, chunk_number, metadata") \
            .ilike("content", f"%{cve_id}%") \
            .limit(limit) \
            .execute()
            
        # If we found direct matches, return them
        if direct_result and hasattr(direct_result, 'data') and len(direct_result.data) > 0:
            print(f"Found {len(direct_result.data)} direct matches for {cve_id}")
            return direct_result.data
            
        # Method 2: Fall back to semantic search
        print(f"No direct matches for {cve_id}, falling back to semantic search")
        return await search_similar_cves(cve_id, limit)
            
    except Exception as e:
        print(f"Error searching for CVE ID: {str(e)}")
        return []

async def main():
    """Main function to demonstrate the search functionality."""
    # Initialize the E5 model
    initialize_e5_model()
    
    while True:
        # Get user input
        search_input = input("\nEnter a CVE ID or search term (or 'quit' to exit): ")
        
        if search_input.lower() in ['quit', 'exit', 'q']:
            break
            
        # Determine if this is a CVE ID search or general search
        if search_input.lower().startswith("cve") or search_input.lower().startswith("snyk"):
            results = await search_cve_by_id(search_input)
        else:
            results = await search_similar_cves(search_input)
        
        # Display results
        if results:
            print(f"\nFound {len(results)} results:")
            for i, result in enumerate(results):
                print(f"\n--- Result {i+1} ---")
                print(f"Title: {result['title']}")
                print(f"URL: {result['url']}")
                print(f"Summary: {result['summary']}")
                print(f"Chunk: {result['chunk_number']}")
                # Print a snippet of the content
                content_preview = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
                print(f"Content Preview: {content_preview}")
        else:
            print("No results found for your query.")

if __name__ == "__main__":
    asyncio.run(main())