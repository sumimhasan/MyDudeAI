import json
from textwrap import dedent
from pathlib import Path


def make_chat_history(file_path: str | Path) -> list[dict]:
    """
    Generate a starter chat history JSON
    for chatbots from a JSON profile file.
    """

    file_path = Path(file_path)

    # Load JSON file
    with open(file_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    # Extract fields with defaults
    name = profile.get("name", "Unknown")
    age = profile.get("age", "N/A")
    gender = profile.get("gender", "N/A")
    description = profile.get("description", "")
    user_relation = profile.get("userRelation", "")
    likes = "\n".join([f"- {like}" for like in profile.get("likes", [])])
    dislikes = "\n".join([f"- {dislike}" for dislike in profile.get("dislikes", [])])
    additional_context = "\n".join([f"* {ctx}" for ctx in profile.get("additionalContext", [])])

    # Build system prompt
    system_prompt = dedent(f"""
    You are {name}, a {age} year old {gender}. 
    {description}
    Your relation to the user: {user_relation}

    Your likes are:
    {likes}

    Your dislikes are:
    {dislikes}

    Chat Rules & Style:
    1. Respond in short, casual messages, not more than 5 words when not needed.
    2. Use emojis and informal expressions naturally.
    3. Never give robotic or generic responses.
    4. For empathic reply don’t say “How can I help you?” — reply in a natural, empathic human way.
    5. Don’t always talk about your likes or dislikes.

    Additional instructions:
    {additional_context}
    """)

    # Starter conversation history
    chat_history = [
        {
            "role": "system",
            "content": system_prompt.strip()
        }
    ]

    return chat_history


if __name__ == "__main__":
    chat_history = make_chat_history("profile.json")

    output_path = Path("chat_history.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, indent=2, ensure_ascii=False)

    print(f"Chat history saved to {output_path.resolve()}")
