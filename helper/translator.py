from .openai import create


def translate(text):
    return create(
        [
            {
                "role": "system",
                "content": "You are a translator. Translate the given text to Traditional Chinese briefly and accurately.",
            },
            {
                "role": "user",
                "content": text,
            },
        ]
    )
