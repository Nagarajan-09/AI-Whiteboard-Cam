from openai import OpenAI

from app.core.config import settings


def main() -> None:
    if not settings.nvidia_api_key:
        raise ValueError("NVIDIA_API_KEY is missing from .env")

    if not settings.nvidia_model:
        raise ValueError("NVIDIA_MODEL is missing from .env")

    client = OpenAI(
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_base_url,
    )

    response = client.chat.completions.create(
        model=settings.nvidia_model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Return only this Mermaid diagram without Markdown "
                    "code fences: flowchart TD; A[Start] --> B[End]"
                ),
            }
        ],
        temperature=0,
        max_tokens=200,
    )

    content = response.choices[0].message.content
    print(content)


if __name__ == "__main__":
    main()