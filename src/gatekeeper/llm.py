import os
from pathlib import Path
from google import genai
from typing import Optional

_CLIENT: Optional[genai.Client] = None

def get_client() -> genai.Client:
    global _CLIENT
    if _CLIENT is None:
        # Try reading from api.txt in the project root first
        # Structure: src/gatekeeper/llm.py -> root is 3 levels up
        root_dir = Path(__file__).resolve().parent.parent.parent
        api_file = root_dir / "api.txt"
        
        api_key = None
        if api_file.exists():
            try:
                api_key = api_file.read_text("utf-8").strip()
            except Exception:
                pass
        
        # Fallback to environment variable
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY")
            
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in api.txt or environment variables")
            
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT

def call_gemini(
    prompt: str,
    model_name: str = "gemini-2.5-flash",
    json_mode: bool = False,
    temperature: float = 0.0
) -> str:
    """
    Synchronous call to Gemini API.
    """
    client = get_client()
    
    config_args = {"temperature": temperature}
    if json_mode:
        config_args["response_mime_type"] = "application/json"
        
    config = genai.types.GenerateContentConfig(**config_args)
        
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config
    )
    return response.text

