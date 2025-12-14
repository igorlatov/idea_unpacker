"""
Configuration for the Idea Unpacker agent flow.
Set your API keys in a .env file or as environment variables.
"""

from dotenv import load_dotenv
load_dotenv()

import os

# API Keys - set these as environment variables
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# Model identifiers
CLAUDE_MODEL = "claude-sonnet-4-20250514"
GPT_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-1.5-flash"
DEEPSEEK_MODEL = "deepseek-chat"

# Flow settings
MAX_REFINEMENT_CYCLES = 3
PLATEAU_THRESHOLD = 0.5
SCORE_DIVERGENCE_THRESHOLD = 2
WORD_LIMIT = 150
MINIMUM_BAR_FLOOR = 8.0