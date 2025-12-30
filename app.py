import os
import time
import markdown
from flask import Flask, request, render_template
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError # Import the specific error

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze-dream', methods=['POST'])
def analyze_dream():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return render_template('index.html', error="API Key missing. Please check .env file.")

    dream_description = request.form.get('dream_description')
    if not dream_description:
        return render_template('index.html', error="Please describe a dream for analysis.")

    # 1. Input Validation
    non_dream_keywords = ["hii", "hello", "how are you"]
    if any(keyword in dream_description.lower() for keyword in non_dream_keywords):
        return render_template('index.html', error="I only analyze dreams. Please describe a dream.")

    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-flash"
    
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=dream_description)],
        ),
    ]

    # 2. Configuration
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        system_instruction=[
            types.Part.from_text(
                text="You are a dream analysis chatbot. Please analyze any user-described dream. Provide a brief (2-4 point) interpretation focused only on the dream’s meaning."
            ),
        ],
    )

    # 3. Robust Execution with Retry Logic (Fix for 429 Errors)
    response_text = None
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            # We use non-streaming here because we need the full text for Markdown conversion
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config
            )
            response_text = response.text
            break # Success! Exit the loop.

        except ClientError as e:
            # Check if it's a Rate Limit error (429)
            if e.code == 429:
                wait_time = 2 ** retry_count # 1s, 2s, 4s...
                print(f"Rate limit hit. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retry_count += 1
            else:
                # If it's a different error (e.g., Key Invalid), crash or handle it
                print(f"API Error: {e}")
                return render_template('index.html', error=f"API Error: {e.message}")
        except Exception as e:
            print(f"Unexpected Error: {e}")
            return render_template('index.html', error="An unexpected error occurred.")

    # If we exited the loop without a response
    if not response_text:
        return render_template('index.html', error="Server is currently busy (Rate Limit Exceeded). Please try again in 1 minute.")

    # 4. Success Handling
    formatted_response = markdown.markdown(response_text)

    # CRITICAL FIX: Render directly. Do not use redirect() for large text.
    return render_template('result.html', analysis=formatted_response)

# Note: You can remove the separate /result route now, it's not needed.

if __name__ == '__main__':
    app.run(debug=True, port=5002)