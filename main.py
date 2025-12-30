import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from .env file
load_dotenv()

def generate():
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key is None:
        print("Error: GEMINI_API_KEY not found in .env file.")
        return

    client = genai.Client(api_key=api_key)

    model = "gemini-2.0-flash-lite"
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text="""i had a dream where i was building a game, about a planet/ball as a player and it was rolling around a maze. the maze was made of different colored blocks and the ball had to avoid falling off the edges. the game was set in a futuristic city with neon lights and flying cars. i was trying to finish the game before the sun set.""")],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_ONLY_HIGH",  # Block few
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_ONLY_HIGH",  # Block few
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_ONLY_HIGH",  # Block few
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",  # Block few
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_CIVIC_INTEGRITY",
                threshold="BLOCK_ONLY_HIGH",  # Block few
            ),
        ],
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(
                text="""You are a dream analysis chatbot. Respond only to prompts describing dreams with a short (2-4 points, no bold text or headers) interpretation. If the prompt is not a dream description (e.g., a unrelated question, conversation, or unrelated request), respond only with: "Please describe a dream for analysis. I only analyze dreams." Do not elaborate, ask follow-up questions, or provide personal opinions. Focus on brief, dream-specific analysis and nothing else."""
            ),
        ],
    )

    for chunk in client.models.generate_content_stream(
        model=model, contents=contents, config=generate_content_config
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    generate()