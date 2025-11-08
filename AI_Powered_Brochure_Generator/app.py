import streamlit as st
import requests
from typing import Optional, Dict, Any
import os
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
import json
from bs4 import BeautifulSoup

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

def initialize_model(api_provider: str, api_key: str, model_name: str, base_url: Optional[str] = None):
    """Initialize the model client based on provider"""
    try:
        if api_provider == "OpenAI":
            if base_url:
                client = OpenAI(api_key=api_key, base_url=base_url)
            else:
                client = OpenAI(api_key=api_key)
            return {
                "client": client, 
                "model_name": model_name, 
                "type": "openai"
            }
        
        elif api_provider == "Anthropic":
            if base_url:
                client = Anthropic(api_key=api_key, base_url=base_url)
            else:
                client = Anthropic(api_key=api_key)
            return {
                "client": client, 
                "model_name": model_name, 
                "type": "anthropic"
            }
        
        elif api_provider == "Google (Gemini)":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            return {
                "client": model, 
                "model_name": model_name, 
                "type": "gemini"
            }
        
        elif api_provider == "Other":
            if not base_url:
                raise ValueError("Base URL is required for custom API providers")
            client = OpenAI(api_key=api_key, base_url=base_url)
            return {
                "client": client, 
                "model_name": model_name, 
                "type": "custom"
            }
        
        else:
            raise ValueError(f"Unsupported API provider: {api_provider}")
    
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None

def generate_responses(model_config: Dict[str, Any], temperature: float, max_tokens: int, 
                      messages: list, response_format: Dict[str, str] = {"type": "text"}):
    """Generate responses using the initialized model"""
    provider = model_config["type"]
    client = model_config["client"]
    model_name = model_config["model_name"]

    try:
        if provider in ["openai", "custom"]:
            supports_response_format = (
                provider == "openai" and
                model_name.startswith(("gpt-4o", "gpt-4.1", "gpt-4-turbo"))
            )

            if supports_response_format:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    response_format=response_format,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

            return response.choices[0].message.content

        elif provider == "anthropic":
            response = client.messages.create(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages
            )

            if hasattr(response, "content") and len(response.content) > 0:
                return response.content[0].text
            else:
                return None

        elif provider == "gemini":
            text_input = "\n".join([msg["content"] for msg in messages if msg["role"] in ["user", "system"]])

            response = client.generate_content(
                text_input,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                }
            )

            if hasattr(response, "text"):
                return response.text
            elif hasattr(response, "candidates"):
                return response.candidates[0].content.parts[0].text
            else:
                return None

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    except Exception as e:
        st.error(f"Error generating response from {provider}: {e}")
        return None

def fetch_information_from_web(web_url: str) -> str:
    """Fetch and parse information from a web URL"""
    try:
        response = requests.get(web_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        title = soup.title.string if soup.title else "No Title Found"
        
        if soup.body:
            for irrelevant in soup.body(['script', 'style', 'img', 'input']):
                irrelevant.decompose()
            text = soup.body.get_text(separator='\n', strip=True)
        else:
            text = ""
        
        return (title + "\n\n" + text)[:2000]
    except Exception as e:
        st.warning(f"Error fetching {web_url}: {str(e)}")
        return f"Could not fetch content from {web_url}"

def fetch_website_links(url: str) -> list:
    """Return the links on the website at the given url"""
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        links = [link.get("href") for link in soup.find_all("a")]
        return [link for link in links if link]
    except Exception as e:
        st.warning(f"Error fetching links from {url}: {str(e)}")
        return []

def get_links_user_prompt(url: str) -> str:
    """Create a prompt for selecting relevant links"""
    user_prompt = f"""
Here is the list of links on the website {url} -
Please decide which of these are relevant web links for a brochure about the company, 
respond with the full https URL in JSON format.
Do not include Terms of Service, Privacy, email links.

Links (some might be relative links):

"""
    links = fetch_website_links(url)
    user_prompt += "\n".join(links)
    return user_prompt

def select_relevant_links(url: str, model_config: Dict[str, Any], 
                         temperature: float, max_tokens: int) -> Dict[str, list]:
    """Select relevant links from the website"""
    link_system_prompt = """
You are provided with a list of links found on a webpage.
You are able to decide which of the links would be most relevant to include in a brochure about the company,
such as links to an About page, or a Company page, or Careers/Jobs pages.
You should respond in JSON as in this example:

{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page", "url": "https://another.full.url/careers"}
    ]
}
"""
    try:
        result = generate_responses(
            model_config=model_config,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": link_system_prompt},
                {"role": "user", "content": get_links_user_prompt(url)}
            ],
            response_format={"type": "json_object"}
        )
        
        if result:
            links = json.loads(result)
            return links
        else:
            return {"links": []}
    except Exception as e:
        st.warning(f"Error selecting relevant links: {str(e)}")
        return {"links": []}

def fetch_page_and_all_relevant_links(url: str, model_config: Dict[str, Any], 
                                      temperature: float, max_tokens: int) -> str:
    """Fetch the main page and all relevant linked pages"""
    contents = fetch_information_from_web(url)
    relevant_links = select_relevant_links(url, model_config, temperature, max_tokens)
    result = f"## Landing Page:\n\n{contents}\n## Relevant Links:\n"
    
    for link in relevant_links.get('links', []):
        result += f"\n\n### Link: {link['type']}\n"
        result += fetch_information_from_web(link["url"])
    
    return result

def get_brochure_user_prompt(url: str, model_config: Dict[str, Any], 
                            temperature: float, max_tokens: int) -> str:
    """Create the user prompt for brochure generation"""
    user_prompt = f"""
You are looking at a website with url: {url}
Here are the contents of its landing page and other relevant pages;
use this information to build a short brochure of the company in markdown without code blocks.\n\n
"""
    user_prompt += fetch_page_and_all_relevant_links(url, model_config, temperature, max_tokens)
    user_prompt = user_prompt[:5_000]  # Truncate if more than 5,000 characters
    return user_prompt

def create_brochure_helper(model_config: Dict[str, Any], url: str, temperature: float, 
                          max_tokens: int, brochure_style: str, additional_context: str = '') -> str:
    """Helper function to create the brochure"""
    brochure_system_prompt = f"""
You are an assistant that analyzes the contents of several relevant pages from a company website
and creates a short {brochure_style} brochure about the company for prospective customers, investors and recruits.
Respond in markdown without code blocks.
Include details of company culture, customers and careers/jobs if you have the information.
"""
    
    if len(additional_context) > 0:
        brochure_system_prompt += f"\n\nNote this additional context while creating the brochure: {additional_context}"

    result = generate_responses(
        model_config=model_config,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": brochure_system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(url, model_config, temperature, max_tokens)}
        ]
    )
    return result

def create_brochure(url: str, model_config: Dict[str, Any], temperature: float,
                   max_tokens: int, context: str = "", style: str = "Professional") -> Dict[str, Any]:
    """Main function to create a brochure"""
    try:
        brochure_content = create_brochure_helper(
            model_config=model_config,
            url=url,
            temperature=temperature,
            max_tokens=max_tokens,
            additional_context=context,
            brochure_style=style
        )
        
        if brochure_content:
            return {
                "success": True,
                "content": brochure_content,
                "metadata": {
                    "url": url,
                    "model": model_config["model_name"],
                    "provider": model_config["type"],
                    "style": style,
                    "temperature": temperature
                }
            }
        else:
            return {
                "success": False,
                "error": "Failed to generate brochure content"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# ============ STREAMLIT UI ============

# Set page config
st.set_page_config(
    page_title="AI Brochure Maker",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">📄 AI Brochure Maker</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar for API configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_provider = st.selectbox(
        "Select API Provider",
        ["OpenAI", "Anthropic", "Google (Gemini)"]
    )
    
    api_key = st.text_input(
        "API Key",
        type="password",
        help="Enter your API key for the selected provider"
    )
    
    # Model Selection
    if api_provider == "OpenAI":
        model_options = ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini","gpt-4.1-mini","o1-mini","gpt-5-nano"]
    elif api_provider == "Anthropic":
        model_options = ["claude-sonnet-4-5-20250929", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"]
    elif api_provider == "Google (Gemini)":
        model_options = ["gemini-pro", "gemini-1.5-pro"]
    else:
        model_options = []
    
    if model_options:
        model = st.selectbox("Select Model", model_options)
    else:
        model = st.text_input("Enter Model Name")
    
    st.subheader("Advanced Settings")
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Controls randomness in generation"
    )
    
    max_tokens = st.number_input(
        "Max Tokens",
        min_value=100,
        max_value=4000,
        value=2000,
        step=100
    )

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Input Details")
    
    url_input = st.text_input(
        "Enter Website URL",
        placeholder="https://example.com",
        help="Provide the URL of the website you want to create a brochure from"
    )
    
    additional_context = st.text_area(
        "Additional Context (Optional)",
        placeholder="Add any specific instructions or context for the brochure...",
        height=100
    )
    
    brochure_style = st.selectbox(
        "Brochure Style",
        ["Professional", "Creative", "Minimalist", "Corporate", "Modern"]
    )

with col2:
    st.subheader("ℹ️ Instructions")
    st.info("""
    **How to use:**
    1. Enter your API credentials in the sidebar
    2. Select your preferred model
    3. Enter the website URL
    4. Optionally add context
    5. Click 'Generate Brochure'
    
    **Supported downloads:**
    - TXT
    - Markdown
    - HTML
    """)

# Generate button
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

with col_btn2:
    generate_button = st.button("🚀 Generate Brochure", use_container_width=True)

# Handle button click
if generate_button:
    if not api_key:
        st.error("⚠️ Please enter your API key in the sidebar")
    elif not url_input:
        st.error("⚠️ Please enter a website URL")
    elif not url_input.startswith(("http://", "https://")):
        st.error("⚠️ Please enter a valid URL (starting with http:// or https://)")
    else:
        with st.spinner("🔄 Initializing model..."):
            model_config = initialize_model(api_provider, api_key, model_name=model, base_url=None)
        
        if model_config is None:
            st.error("❌ Failed to initialize model. Please check your credentials.")
        else:
            with st.spinner("🔄 Generating your brochure... This may take a moment."):
                result = create_brochure(
                    url=url_input,
                    model_config=model_config,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    context=additional_context,
                    style=brochure_style
                )
            
            if result["success"]:
                st.success("✅ Brochure generated successfully!")
                
                st.subheader("📄 Generated Brochure")
                st.markdown("---")
                st.markdown(result["content"])
                
                with st.expander("📊 Generation Details"):
                    st.json(result["metadata"])
                
                st.subheader("💾 Download Options")
                col_d1, col_d2, col_d3 = st.columns(3)
                
                with col_d1:
                    st.download_button(
                        label="Download as TXT",
                        data=result["content"],
                        file_name="brochure.txt",
                        mime="text/plain"
                    )
                
                with col_d2:
                    st.download_button(
                        label="Download as MD",
                        data=result["content"],
                        file_name="brochure.md",
                        mime="text/markdown"
                    )
                
                with col_d3:
                    html_content = f"<html><body><pre>{result['content']}</pre></body></html>"
                    st.download_button(
                        label="Download as HTML",
                        data=html_content,
                        file_name="brochure.html",
                        mime="text/html"
                    )
            else:
                st.error(f"❌ Error generating brochure: {result['error']}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <p>Made with ❤️ using Streamlit | Powered by AI</p>
    </div>
    """,
    unsafe_allow_html=True
)