
import os
import json
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

APP_TITLE = "DevAssist AI"
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
FALLBACK_MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]
DATA_FILE = Path("devassist_usage.json")
DEFAULT_CHAT_MESSAGE = {
    "role": "assistant",
    "content": (
        "Hi, I am your Developer Chat assistant. Ask me about errors, deployment, "
        "networking, APIs, project ideas, or how to improve your code."
    ),
}


FEATURES = [
    "Home",
    "Developer Chat",
    "Bug Finder",
    "Code Reviewer",
    "Code Explainer",
    "Documentation Generator",
    "Code Optimizer",
    "Code Generator",
    "Image Generator",
    "Interview Questions",
    "Productivity Report",
]


STRUCTURED_OUTPUT_RULES = """
Format the answer in a polished, easy-to-scan style.
Avoid long paragraphs. Use short sections and bullets.
Do not over-explain simple ideas.
Use this structure when suitable:

### Quick Summary
- One or two direct points.

### Key Findings
- Specific issues, observations, or decisions.

### Suggested Fix
- Clear action steps.

### Corrected / Example Code
Include code only when useful.

### Next Step
- One practical next action.
"""


PROMPTS = {
    "Bug Finder": {
        "button": "Find Bugs",
        "input_label": "Paste your code here",
        "placeholder": "Paste code with a possible error...",
        "system": (
            "You are a careful software debugging assistant. Find real bugs, syntax errors, "
            "logic mistakes, and risky edge cases. Give the issue, why it matters, and a corrected version. "
            "Keep the answer structured and beginner-friendly."
        ),
        "user": "Find bugs in this code and suggest fixes:\n\n{content}",
    },
    "Code Reviewer": {
        "button": "Review Code",
        "input_label": "Paste your code here",
        "placeholder": "Paste code to review...",
        "system": (
            "You are a senior code reviewer. Review for correctness, readability, performance, security, "
            "maintainability, and best practices. Be practical, concise, and beginner-friendly."
        ),
        "user": "Review this code and provide a quality score plus improvement suggestions:\n\n{content}",
    },
    "Code Explainer": {
        "button": "Explain Code",
        "input_label": "Paste your code here",
        "placeholder": "Paste code to explain...",
        "system": (
            "You are a teaching assistant for programming students. Explain code clearly, including purpose, "
            "important concepts, and time complexity when relevant. Keep it simple and structured."
        ),
        "user": "Explain this code clearly:\n\n{content}",
    },
    "Documentation Generator": {
        "button": "Generate Documentation",
        "input_label": "Paste your code here",
        "placeholder": "Paste code to document...",
        "system": (
            "You generate useful developer documentation. Create function descriptions, usage notes, "
            "comments where helpful, and README-style documentation when appropriate. Keep it polished."
        ),
        "user": "Generate documentation for this code:\n\n{content}",
    },
    "Code Optimizer": {
        "button": "Optimize Code",
        "input_label": "Paste your code here",
        "placeholder": "Paste code to optimize...",
        "system": (
            "You are a code optimization assistant. Improve clarity, speed, and maintainability without "
            "changing intended behavior. Explain what changed and why in a short structured format."
        ),
        "user": "Optimize this code and explain the improvements:\n\n{content}",
    },
    "Code Generator": {
        "button": "Generate Code",
        "input_label": "Enter your requirement",
        "placeholder": "Example: Create a Java program to check palindrome",
        "system": (
            "You generate clean, complete, beginner-friendly code. Include short explanations and mention "
            "how to run the code when useful. Avoid unnecessary theory."
        ),
        "user": "Generate code for this requirement:\n\n{content}",
    },
}


INTERVIEW_TOPICS = ["Python", "Java", "SQL", "HTML", "CSS", "AI/ML"]

CHAT_SYSTEM_PROMPT = """
You are DevAssist AI's developer chat mentor.
Help developers discuss coding errors, deployment ideas, hosting choices, networking basics,
project architecture, database choices, GitHub, Streamlit Cloud, API errors, and debugging plans.
Ask a short follow-up question only when necessary.
Keep answers practical, friendly, and structured.
"""


def default_state():
    return {
        "reviews": 0,
        "bugs": 0,
        "explanations": 0,
        "docs": 0,
        "optimizations": 0,
        "generations": 0,
        "questions": 0,
        "images": 0,
        "chats": 0,
        "activity": [],
        "chat_messages": [DEFAULT_CHAT_MESSAGE],
    }


def load_saved_state():
    if not DATA_FILE.exists():
        return {}

    try:
        with DATA_FILE.open("r", encoding="utf-8") as file:
            saved_state = json.load(file)
            return saved_state if isinstance(saved_state, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_state():
    keys_to_save = default_state().keys()
    data = {key: st.session_state.get(key) for key in keys_to_save}

    try:
        with DATA_FILE.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
    except OSError:
        st.toast("Usage data could not be saved on this system.")


def init_state():
    defaults = default_state()
    saved_state = load_saved_state()
    for key, value in defaults.items():
        st.session_state.setdefault(key, saved_state.get(key, value))


def get_secret(name):
    try:
        return st.secrets.get(name)
    except Exception:
        return None

def get_api_key():
    return os.getenv("OPENROUTER_API_KEY") or get_secret("OPENROUTER_API_KEY")


def model_name():
    return get_secret("GEMINI_MODEL") or DEFAULT_MODEL


def model_choices():
    preferred_model = model_name()
    choices = [preferred_model]
    for fallback_model in FALLBACK_MODELS:
        if fallback_model not in choices:
            choices.append(fallback_model)
    return choices


def call_ai(system_prompt, user_prompt):
    api_key = get_api_key()

    if not api_key:
        st.warning("Add your OpenRouter API key to use this feature.")
        st.code("OPENROUTER_API_KEY=your_openrouter_api_key_here", language="text")
        return None

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        OPENROUTER_MODELS = [
            "openai/gpt-4o-mini",
            "anthropic/claude-3-haiku",
            "meta-llama/llama-3.1-8b-instruct"
        ]

        for model in OPENROUTER_MODELS:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt + "\n\n" + STRUCTURED_OUTPUT_RULES
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],
                    temperature=0.7,
                    max_tokens=1024
                )

                return response.choices[0].message.content

            except Exception as e:
                continue

        st.error("❌ All OpenRouter models failed. Check your API key or credits.")
        return None

    except Exception as error:
        st.error("❌ OpenRouter API request failed.")
        st.caption(str(error))
        return None
    
def record_activity(feature):
    mapping = {
        "Bug Finder": "bugs",
        "Code Reviewer": "reviews",
        "Code Explainer": "explanations",
        "Documentation Generator": "docs",
        "Code Optimizer": "optimizations",
        "Code Generator": "generations",
        "Image Generator": "images",
        "Interview Questions": "questions",
        "Developer Chat": "chats",
    }
    key = mapping.get(feature)
    if key:
        st.session_state[key] += 1
    st.session_state.activity.insert(
        0,
        {
            "time": datetime.now().strftime("%I:%M %p"),
            "feature": feature,
        },
    )
    save_state()


def render_home():
    st.header("Developer Productivity Dashboard")
    st.write(
        "DevAssist AI helps developers review code, find bugs, explain logic, generate documentation, "
        "optimize code, generate programs, and prepare for interviews."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Code Reviews", st.session_state.reviews)
    col2.metric("Bugs Checked", st.session_state.bugs)
    col3.metric("Docs Generated", st.session_state.docs)

    st.divider()
    st.subheader("Suggested Demo Input")
    st.code("for i in range(5)\n    print(i)", language="python")
    st.caption("Try this in Bug Finder to show a missing colon error during your review.")


def render_ai_feature(feature):
    config = PROMPTS[feature]
    st.header(feature)

    if feature == "Code Generator":
        content = st.text_area(
            config["input_label"],
            height=120,
            placeholder=config["placeholder"],
        )
    else:
        content = st.text_area(
            config["input_label"],
            height=260,
            placeholder=config["placeholder"],
        )

    if st.button(config["button"], type="primary"):
        if not content.strip():
            st.warning("Please enter some input first.")
            return

        with st.spinner("Working on it..."):
            result = call_ai(config["system"], config["user"].format(content=content))

        if result:
            record_activity(feature)
            st.subheader("Result")
            st.markdown(result)
            st.download_button(
                "Download Result",
                data=result,
                file_name=f"{feature.lower().replace(' ', '_')}_result.md",
                mime="text/markdown",
            )


def render_developer_chat():
    st.header("Developer Chat")
    st.caption("Ask freely about errors, deployment, networking, project ideas, APIs, or code decisions.")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_message = st.chat_input("Ask about your code, error, deployment, or project idea...")
    if not user_message:
        return

    st.session_state.chat_messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    recent_history = st.session_state.chat_messages[-8:]
    history_text = "\n".join(
        f"{message['role'].title()}: {message['content']}" for message in recent_history
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = call_ai(
                CHAT_SYSTEM_PROMPT,
                f"Conversation so far:\n{history_text}\n\nLatest developer message:\n{user_message}",
            )

        if response:
            st.markdown(response)
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            record_activity("Developer Chat")
        else:
            st.warning("I could not generate a chat reply right now. Try again in a moment.")

    if st.button("Clear Chat"):
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "Chat cleared. Ask me about errors, deployment, networking, APIs, or project ideas.",
            }
        ]
        save_state()
        st.rerun()


def render_image_generator():
    st.header("Image Generator")
    st.caption("Create demo images from text prompts for UI mockups, posters, diagrams, and project visuals.")

    prompt = st.text_area(
        "Describe the image you want",
        height=140,
        placeholder="Example: A modern dashboard illustration for an AI developer productivity assistant",
    )

    col1, col2 = st.columns(2)
    style = col1.selectbox(
        "Style",
        ["Modern UI", "Realistic", "3D Render", "Illustration", "Poster", "Logo Concept", "Infographic"],
    )
    size = col2.selectbox("Size", ["1024x1024", "1280x720", "720x1280"])

    enhance_prompt = st.checkbox("Improve prompt with Gemini first", value=True)

    if st.button("Generate Image", type="primary"):
        if not prompt.strip():
            st.warning("Please enter an image prompt first.")
            return

        final_prompt = f"{prompt}, {style} style, high quality, clean composition"

        if enhance_prompt:
            with st.spinner("Improving prompt..."):
                improved_prompt = call_ai(
                    "You are an image prompt engineer. Convert the user idea into one concise visual prompt. Do not add explanation.",
                    f"Create a polished image generation prompt for this idea:\n\n{prompt}\n\nStyle: {style}",
                )
            if improved_prompt:
                final_prompt = improved_prompt.strip()

        width, height = size.split("x")
        image_url = (
            "https://gen.pollinations.ai/image/"
            f"{quote_plus(final_prompt)}?width={width}&height={height}&nologo=true"
        )

        record_activity("Image Generator")
        st.subheader("Generated Image")
        st.image(image_url, caption=final_prompt, use_container_width=True)
        st.markdown(f"[Open image in new tab]({image_url})")
        st.download_button(
            "Download Prompt",
            data=final_prompt,
            file_name="image_prompt.txt",
            mime="text/plain",
        )


def render_interview_questions():
    st.header("Interview Questions")
    topic = st.selectbox("Choose topic", INTERVIEW_TOPICS)
    level = st.selectbox("Difficulty", ["Beginner", "Intermediate", "Advanced"])

    if st.button("Generate Questions", type="primary"):
        system = (
            "You are an interview preparation assistant. Generate practical interview questions "
            "with short, clear answers for students and junior developers."
        )
        prompt = f"Create 10 {level.lower()} interview questions and answers for {topic}."

        with st.spinner("Preparing questions..."):
            result = call_ai(system, prompt)

        if result:
            record_activity("Interview Questions")
            st.subheader("Questions and Answers")
            st.markdown(result)


def render_report():
    st.header("Productivity Report")

    total = (
        st.session_state.reviews
        + st.session_state.chats
        + st.session_state.bugs
        + st.session_state.explanations
        + st.session_state.docs
        + st.session_state.optimizations
        + st.session_state.generations
        + st.session_state.questions
        + st.session_state.images
    )
    estimated_minutes_saved = total * 8

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total AI Tasks", total)
    col2.metric("Bugs Checked", st.session_state.bugs)
    col3.metric("Code Reviews", st.session_state.reviews)
    col4.metric("Estimated Time Saved", f"{estimated_minutes_saved} min")

    data = {
        "Feature": [
            "Developer Chat",
            "Bug Finder",
            "Code Reviewer",
            "Code Explainer",
            "Documentation Generator",
            "Code Optimizer",
            "Code Generator",
            "Image Generator",
            "Interview Questions",
        ],
        "Usage Count": [
            st.session_state.chats,
            st.session_state.bugs,
            st.session_state.reviews,
            st.session_state.explanations,
            st.session_state.docs,
            st.session_state.optimizations,
            st.session_state.generations,
            st.session_state.images,
            st.session_state.questions,
        ],
    }
    st.bar_chart(data, x="Feature", y="Usage Count")

    if st.session_state.activity:
        st.subheader("Recent Activity")
        st.table(st.session_state.activity[:8])
    else:
        st.info("Use any feature to start building your productivity report.")


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_state()

    st.title(APP_TITLE)
    st.caption("Intelligent Developer Productivity Assistant")

    with st.sidebar:
        st.subheader("Navigation")
        menu = st.selectbox("Select feature", FEATURES)

    if menu == "Home":
        render_home()
    elif menu == "Developer Chat":
        render_developer_chat()
    elif menu == "Image Generator":
        render_image_generator()
    elif menu == "Interview Questions":
        render_interview_questions()
    elif menu == "Productivity Report":
        render_report()
    else:
        render_ai_feature(menu)


if __name__ == "__main__":
    main()
