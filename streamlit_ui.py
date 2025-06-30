from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

import streamlit as st
import json
import logfire
from supabase import Client, create_client
from openai import AsyncOpenAI

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)

# Import our modified pydantic_ai_expert
# Make sure this imports the new version with local embeddings
from improved_pydantic_ai_expert import pydantic_ai_expert, PydanticAIDeps, initialize_e5_model

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setup clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

# Initialize E5 model at startup
initialize_e5_model()

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""
    role: Literal['user', 'model']
    timestamp: str
    content: str


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)


async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Prepare dependencies
    deps = PydanticAIDeps(
        supabase=supabase,
        openai_client=openai_client
    )

    # Show a spinner while loading
    with st.spinner("Searching databases..."):
        # Run the agent in a stream
        async with pydantic_ai_expert.run_stream(
            user_input,
            deps=deps,
            message_history=st.session_state.messages[:-1],  # pass entire conversation so far
        ) as result:
            # We'll gather partial text to show incrementally
            partial_text = ""
            message_placeholder = st.empty()

            # Render partial text as it arrives
            async for chunk in result.stream_text(delta=True):
                partial_text += chunk
                message_placeholder.markdown(partial_text)

            # Now that the stream is finished, we have a final result.
            # Add new messages from this run, excluding user-prompt messages
            #filtered_messages = [msg for msg in result.new_messages() 
                                #if not (hasattr(msg, 'parts') and 
                                        #any(part.part_kind == 'user-prompt' for part in msg.parts))]
            #st.session_state.messages.extend(filtered_messages)

            # Add the final response to the messages
            st.session_state.messages.append(
                ModelResponse(parts=[TextPart(content=partial_text)])
            )


async def main():
    st.set_page_config(
        page_title="CVE Expert - Agentic RAG System",
        page_icon="🔐",
        layout="wide"
    )

    # Custom CSS to make the layout work properly with fixed input at bottom
    st.markdown("""
        <style>
        /* Main container styling */
        .main > div {
            padding-bottom: 150px; /* Reserve space for input area */
        }
        
        /* Input container fixed at bottom */
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: var(--background-color);
            padding: 1rem;
            border-top: 1px solid var(--border-color);
            z-index: 1000;
        }
        
        /* Ensure proper scrolling */
        .element-container {
            margin-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)
        
    st.title("🔐 CVE Expert - Agentic RAG System")
    st.write("Ask any question about CVEs, vulnerabilities, or security advisories.")
    
    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Add sidebar with instructions
    with st.sidebar:
        st.header("About this app")
        st.markdown("""
        This application uses a local E5 embedding model to search through a database of CVEs and security advisories.
        
        ### Example queries:
        - "Tell me about CVE-2023-1234"
        - "What is SNYK-DOTNET-SNOWFLAKEDATA-9901391?"
        - "What are the recent Log4j vulnerabilities?"
        - "How can I mitigate Spring4Shell?"
        - "Analyze my Dockerfile for vulnerabilities" (with file attached)
        - "What CVEs affect the base images in my Docker file?"
        - "Suggest security fixes for my container setup"
        """)
        
        st.header("System Info")
        st.markdown("""
        - Uses local E5 embeddings for semantic search
        - Powered by PydanticAI and LLM backend
        - Vector database with Supabase
        """)

    # Chat messages container - this will show above the input
    chat_container = st.container()
    
    # Display existing chat messages in the chat container
    with chat_container:
        # Display all messages from the conversation so far
        for msg in st.session_state.messages:
            if isinstance(msg, ModelRequest):
                # Handle user messages
                for part in msg.parts:
                    if part.part_kind == 'user-prompt':
                        with st.chat_message("user"):
                            # Extract just the user's original input (before file content)
                            content = part.content
                            # Split at the ATTACHED FILE section to show only user input
                            user_only_content = content.split("\n\nATTACHED FILE")[0].strip()
                            st.markdown(user_only_content)
                            
                            # Show attached files if they exist in the message
                            if "ATTACHED FILE" in content:
                                if 'uploaded_files' in st.session_state and st.session_state.uploaded_files:
                                    for filename, file_data in st.session_state.uploaded_files.items():
                                        with st.expander(f"📎 {filename}"):
                                            # Determine language for syntax highlighting
                                            file_ext = filename.split('.')[-1].lower()
                                            if file_ext in ['dockerfile', 'docker','File']:
                                                lang = 'dockerfile'
                                            elif file_ext in ['py']:
                                                lang = 'python'
                                            elif file_ext in ['js']:
                                                lang = 'javascript'
                                            elif file_ext in ['json']:
                                                lang = 'json'
                                            elif file_ext in ['yaml', 'yml']:
                                                lang = 'yaml'
                                            else:
                                                lang = 'text'
                                            
                                            #st.code(file_data['content'], language=lang)
                    else:
                        display_message_part(part)
            elif isinstance(msg, ModelResponse):
                # Handle assistant messages
                for part in msg.parts:
                    display_message_part(part)

    # Add some spacing before the input section
    st.markdown("---")
    
    # Input container at the bottom
    input_container = st.container()
    
    with input_container:
        # File attachment and chat input
        col1, col2 = st.columns([1, 8])

        with col1:
            # File upload button with paperclip icon
            uploaded_file = st.file_uploader(
                "📎",
                type=None,  # Allow all file types
                help="Attach a file for analysis",
                label_visibility="collapsed",
                key="file_uploader"
            )
        
        with col2:
            # Chat input in the larger column
            user_input = st.chat_input("Ask about CVEs, vulnerabilities, or security advisories...")

    # Handle file upload
    if uploaded_file is not None:
        if uploaded_file.name not in st.session_state.get('uploaded_files', {}):
            # Store file content
            if 'uploaded_files' not in st.session_state:
                st.session_state.uploaded_files = {}
            
            try:
                st.session_state.uploaded_files[uploaded_file.name] = {
                    'name': uploaded_file.name,
                    'content': uploaded_file.read().decode('utf-8'),
                    'type': uploaded_file.type or 'text/plain'
                }
                st.rerun()
            except UnicodeDecodeError:
                st.error(f"Could not read {uploaded_file.name}. Please ensure it's a text file.")

    # Handle chat input
    if user_input:
        # Check if we have uploaded files to include in the query
        attached_files_content = ""
        if 'uploaded_files' in st.session_state and st.session_state.uploaded_files:
            for filename, file_data in st.session_state.uploaded_files.items():
                # Determine file type for better formatting
                file_ext = filename.split('.')[-1].lower()
                if file_ext in ['dockerfile', 'docker','File']:
                    lang = 'dockerfile'
                elif file_ext in ['py']:
                    lang = 'python'
                elif file_ext in ['js']:
                    lang = 'javascript'
                elif file_ext in ['json']:
                    lang = 'json'
                elif file_ext in ['yaml', 'yml']:
                    lang = 'yaml'
                else:
                    lang = 'text'
                
                attached_files_content += f"""

ATTACHED FILE ({filename}):
```{lang}
{file_data['content']}
```
"""
        
        if attached_files_content:
            enhanced_input = f"{user_input}{attached_files_content}\n\nthe attached docker file has above mentioned cves please find suitable fixes using your cve database and give fixed docker file."
        else:
            enhanced_input = user_input
        
        # Add the new request to the conversation
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=enhanced_input)])
        )
        
        # Display the assistant's response with streaming in the chat container
        with chat_container:
            with st.chat_message("assistant"):
                # Run the agent with streaming
                await run_agent_with_streaming(enhanced_input)
        
        # Rerun to update the display and show the new message
        st.rerun()


if __name__ == "__main__":
    asyncio.run(main())