from transformers import pipeline
from src.config import AppConfig


generator = pipeline(
    "text2text-generation",
    model=AppConfig.GENERATOR_MODEL
)


def generate_answer(prompt: str) -> str:

    response = generator(
        prompt,
        max_new_tokens=AppConfig.MAX_NEW_TOKENS,
        
        repetition_penalty=1.2,
        do_sample=False
    )

    return response[0]["generated_text"]
