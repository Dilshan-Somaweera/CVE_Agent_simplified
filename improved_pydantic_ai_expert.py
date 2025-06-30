from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List, Dict, Any

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

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

# Initialize E5 model at startup
initialize_e5_model()

@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI

system_prompt = """
You are an expert in CVEs, with access to a database of vulnerabilities, including descriptions, severity, affected systems, and mitigations.

You only assist with CVE-related questions.

Always check the database using RAG before answering. If needed, retrieve specific CVE details.

If no relevant info is found, inform the user. If asks for configurations refer the related data about the specified CVE and give configurations based on your knowledge
"""

pydantic_ai_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

async def get_embedding_local(text: str) -> List[float]:
    """Get embedding vector using local E5 model."""
    try:
        # Get the model and tokenizer instances
        model, tokenizer = E5_MODEL, E5_TOKENIZER
        
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

@pydantic_ai_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the embedding for the query using local E5 model
        query_embedding = await get_embedding_local(user_query)
        
        # Query Supabase for relevant documents using cosine similarity
        # This leverages the vector similarity search
        result = ctx.deps.supabase.table("site_pages") \
            .select("url, title, summary, content, chunk_number, metadata") \
            .search("embedding", query_embedding, distance="cosine") \
            .limit(5) \
            .execute()
        
        if not result.data:
            return "No relevant documentation found."
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

## Source URL
{doc['url']}

## Content
{doc['content']}
"""
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@pydantic_ai_expert.tool
async def search_cve_by_id(ctx: RunContext[PydanticAIDeps], cve_id: str) -> str:
    """
    Search specifically for a CVE ID.
    This creates a more targeted query for CVE numbers.
    
    Args:
        ctx: The context including the Supabase client
        cve_id: The CVE ID to search for (e.g., "CVE-2023-1234")
        
    Returns:
        A formatted string containing the results
    """
    # Normalize the CVE ID format
    cve_id = cve_id.upper()
    if not cve_id.startswith("CVE-") and cve_id.startswith("CVE"):
        cve_id = f"CVE-{cve_id[3:]}"
    
    # Also handle SNYK IDs
    is_snyk = cve_id.upper().startswith("SNYK")
        
    try:
        # Method 1: Try direct text search first (more precise for exact CVE IDs)
        if is_snyk:
            direct_result = ctx.deps.supabase.table("site_pages") \
                .select("url, title, summary, content, chunk_number, metadata") \
                .ilike("content", f"%{cve_id}%") \
                .limit(5) \
                .execute()
        else:
            direct_result = ctx.deps.supabase.table("site_pages") \
                .select("url, title, summary, content, chunk_number, metadata") \
                .ilike("content", f"%{cve_id}%") \
                .limit(5) \
                .execute()
            
        # If we found direct matches, return them
        if direct_result and hasattr(direct_result, 'data') and len(direct_result.data) > 0:
            # Format the results
            formatted_chunks = []
            for doc in direct_result.data:
                chunk_text = f"""
# {doc['title']}

## Source URL
{doc['url']}

## Content
{doc['content']}
"""
                formatted_chunks.append(chunk_text)
                
            # Join all chunks with a separator
            return "\n\n---\n\n".join(formatted_chunks)
        
        # Method 2: Fall back to semantic search
        print(f"No direct matches for {cve_id}, falling back to semantic search")
        query_embedding = await get_embedding_local(cve_id)
        
        semantic_result = ctx.deps.supabase.table("site_pages") \
            .select("url, title, summary, content, chunk_number, metadata") \
            .search("embedding", query_embedding, distance="cosine") \
            .limit(5) \
            .execute()
        
        if semantic_result and hasattr(semantic_result, 'data') and len(semantic_result.data) > 0:
            # Format the results
            formatted_chunks = []
            for doc in semantic_result.data:
                chunk_text = f"""
# {doc['title']}

## Source URL
{doc['url']}

## Content
{doc['content']}
"""
                formatted_chunks.append(chunk_text)
                
            # Join all chunks with a separator
            return "\n\n---\n\n".join(formatted_chunks)
            
        return f"No information found for {cve_id}"
            
    except Exception as e:
        print(f"Error searching for CVE ID: {str(e)}")
        return f"Error searching for CVE ID: {str(e)}"

@pydantic_ai_expert.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query Supabase for unique URLs
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@pydantic_ai_expert.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title']
        formatted_content = [f"# {page_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"