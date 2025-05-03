import openai
import os
from dotenv import load_dotenv

# Load environment variables (store your API key in a .env file)
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_cause_from_gpt(pest, weather_data):
    """
    Get probable cause from GPT based on pest and weather data
    """
    prompt = f"The pest causing the plant disease is: {pest}. The current weather data is: {weather_data}. What could be the probable cause of the issue (e.g., nutrient deficiency, climate change)?"

    response = openai.Completion.create(
        engine="text-davinci-003",  # You can use the newer models if you prefer
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )

    cause = response.choices[0].text.strip()
    return cause


def get_remedy_from_gpt(pest, cause):
    """
    Get a simple home remedy for the plant issue based on pest and cause
    """
    prompt = f"Plant disease caused by {pest} and the probable cause is: {cause}. Can you suggest a simple home remedy to fix this?"

    response = openai.Completion.create(
        engine="text-davinci-003",  # Same model as before
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )

    remedy = response.choices[0].text.strip()
    return remedy
