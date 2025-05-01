import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv

# Import for E5 model
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

# New function using local E5 model instead of OpenAI
async def get_embedding_local(text: str) -> List[float]:
    """Get embedding vector using local E5 model."""
    try:
        # Initialize model if not already loaded
        model, tokenizer = initialize_e5_model(model_size="small")  # Use small for 8GB RAM
        
        # Create a coroutine to run the model inference
        def run_model():
            # Prepare input - E5 models expect "passage: " prefix
            inputs = tokenizer(f"passage: {text}", padding=True, truncation=True,
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

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        # Print the data being sent (optional, for debugging)
        print(f"Attempting to insert data for chunk {chunk.chunk_number}")
        
        # Use upsert with explicit conflict handling
        result = supabase.table("site_pages").upsert(
            data,
            on_conflict="url,chunk_number",
            returning="minimal"  # Only return minimal info to reduce response size
        ).execute()
        
        print(f"Supabase response for chunk {chunk.chunk_number}: {result}")
        return result
    except Exception as e:
        print(f"Detailed error upserting chunk {chunk.chunk_number}: {str(e)}")
        return None

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text with rate limiting."""
    # Add delay between requests
    await asyncio.sleep(1)  # 1 second delay
    
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)
    
    # Add delay before embedding request
    await asyncio.sleep(1)
    
    # Get embedding using local E5 model instead of OpenAI
    embedding = await get_embedding_local(chunk)
    
    # Create metadata
    metadata = {
        "source": "pydantic_ai_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,
        metadata=metadata,
        embedding=embedding
    )

async def process_and_store_document(url: str, markdown: str, skip_existing: bool = False):  # Set default to False
    """Process a document and store its chunks with rate limiting."""
    # Split into chunks
    chunks = chunk_text(markdown)
    print(f"Created {len(chunks)} chunks from document")
    
    # Process chunks sequentially instead of in parallel
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1} of {len(chunks)}")
        processed_chunk = await process_chunk(chunk, i, url)
        processed_chunks.append(processed_chunk)
        await asyncio.sleep(1)  # Add delay between chunks
    
    # Store chunks sequentially
    for i, chunk in enumerate(processed_chunks):
        print(f"Storing chunk {i+1} of {len(processed_chunks)}")
        result = await insert_chunk(chunk)
        if result:
            print(f"Successfully stored chunk {i+1}")
        else:
            print(f"Failed to store chunk {i+1}")
        await asyncio.sleep(0.5)

async def crawl_parallel(urls: List[str], max_concurrent: int = 2):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_document(url, result.markdown_v2.raw_markdown)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

def get_pydantic_ai_docs_urls() -> List[str]:
    """Get URLs from Pydantic AI docs sitemap."""
    sitemap_url = "https://ai.pydantic.dev/sitemap.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Extract all URLs from the sitemap
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

async def main():
    # Preload E5 model at startup
    initialize_e5_model()
    
    # Get URLs from your list
    urls = [
    "https://security.snyk.io/vuln/SNYK-SLES150-SYSTEMUSERROOT-2752907",
    "https://stackoverflow.com/questions/38030116/is-alpine-linux-an-implementation-of-a-unikernel/60532451#60532451",
    "https://security.snyk.io/vuln/SNYK-ALPINE321-NGINX-8484840",
    "https://security.snyk.io/vuln/SNYK-ALPINE321-NGINX-8484898",
    "https://security.snyk.io/vuln/SNYK-ALPINE321-NGINX-8484847",
    "https://security.snyk.io/vuln/SNYK-ALPINE321-NGINX-8484822",
    "https://security.snyk.io/vuln/SNYK-ALPINE321-XZ-8487516",
    "https://stackoverflow.com/questions/28229095/apache-commons-compress-using-7zip/28277555#28277555",
    "https://security.snyk.io/vuln/SNYK-ALPINE321-OPENSSL-8485258",
    "https://stackoverflow.com/questions/76806358/cannot-scan-a-local-unpushed-image-for-cves-with-anchore-grype-workarounds/76806506#76806506",
    "https://security.snyk.io/vuln/SNYK-ORACLE9-NGINXALLMODULES-9564022",
    "https://security.snyk.io/vuln/SNYK-ALPINE321-OPENSSH-8485234"
    ]


    # Load checkpoint if it exists
    processed_urls = []
    try:
        if os.path.exists("checkpoint.json"):
            with open("checkpoint.json", "r") as f:
                processed_urls = json.load(f)
                print(f"Loaded checkpoint with {len(processed_urls)} already processed URLs")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
    
    # Filter out already processed URLs
    urls_to_process = [url for url in urls if url not in processed_urls]
    print(f"Found {len(urls_to_process)} URLs to process")
    
    if not urls_to_process:
        print("No new URLs to process")
        return
    
    try:
        # Test Supabase connection
        test_query = supabase.table("site_pages").select("count").limit(1).execute()
        print("Supabase connection test successful")
        print(f"Current row count: {test_query}")
    except Exception as e:
        print(f"Supabase connection test failed: {str(e)}")
        return
    
    # Process URLs in batches to avoid overwhelming API limits
    batch_size = 1  # Process 3 URLs per batch
    base_delay = 5  # Base delay between API calls in seconds
    
    # Create the crawler instance
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    
    try:
        for batch_start in range(0, len(urls_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(urls_to_process))
            batch_urls = urls_to_process[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//batch_size + 1}: URLs {batch_start+1}-{batch_end} of {len(urls_to_process)}")
            
            # Process each URL in the current batch
            for i, url in enumerate(batch_urls):
                try:
                    print(f"Processing URL {batch_start+i+1}/{len(urls_to_process)}: {url}")
                    
                    # Crawl the URL
                    result = await crawler.arun(
                        url=url,
                        config=crawl_config,
                        session_id="session1"
                    )
                    
                    if result.success:
                        print(f"Successfully crawled: {url}")
                        
                        # Process document with dynamic delay to avoid rate limiting
                        # Calculate delay based on position in the batch to spread out API calls
                        current_delay = base_delay * (1 + (i * 0.5))
                        
                        # Split into chunks
                        chunks = chunk_text(result.markdown_v2.raw_markdown)
                        print(f"Created {len(chunks)} chunks from document")
                        
                        # Process chunks sequentially with increasing delays
                        processed_chunks = []
                        for j, chunk in enumerate(chunks):
                            print(f"Processing chunk {j+1} of {len(chunks)}")
                            # Add increasing delay between chunks based on position
                            await asyncio.sleep(current_delay + (j * 0.5))
                            processed_chunk = await process_chunk(chunk, j, url)
                            processed_chunks.append(processed_chunk)
                        
                        # Store chunks sequentially
                        for j, chunk in enumerate(processed_chunks):
                            print(f"Storing chunk {j+1} of {len(processed_chunks)}")
                            result = await insert_chunk(chunk)
                            if result:
                                print(f"Successfully stored chunk {j+1}")
                            else:
                                print(f"Failed to store chunk {j+1}")
                            await asyncio.sleep(0.5)
                        
                        # Add URL to processed list and save checkpoint
                        processed_urls.append(url)
                        with open("checkpoint.json", "w") as f:
                            json.dump(processed_urls, f)
                        
                        print(f"Completed processing URL {batch_start+i+1}/{len(urls_to_process)}: {url}")
                    else:
                        print(f"Failed to crawl URL {batch_start+i+1}/{len(urls_to_process)}: {url} - Error: {result.error_message}")
                
                except Exception as e:
                    print(f"Error processing {url}: {str(e)}")
                    # Continue with next URL instead of stopping completely
                    continue
                
                # Add delay between URLs in the same batch
                await asyncio.sleep(base_delay + (i * 2))
            
            # Add a longer delay between batches
            if batch_end < len(urls_to_process):
                print(f"Completed batch. Waiting 60 seconds before starting next batch...")
                await asyncio.sleep(5)  # 1 minute between batches
                
    finally:
        await crawler.close()
        print("Crawler closed. Processing complete.")

if __name__ == "__main__":
    asyncio.run(main())