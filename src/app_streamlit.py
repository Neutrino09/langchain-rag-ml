# src/app_streamlit.py

import streamlit as st
from rag_chain import ask_question  # same-folder import

# Page settings
st.set_page_config(
    page_title="ML RAG Chatbot",
    page_icon=None,
    layout="wide",
)


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": (
                    "You are connected to a question-answering system built on top of the "
                    "Machine Learning Wikipedia page. Ask any question about machine learning, "
                    "and I will answer using that knowledge base."
                ),
                "sources": [],
            }
        ]


def render_sidebar():
    with st.sidebar:
        st.markdown("### About this app")
        st.markdown(
            """
This is a **Retrieval-Augmented Generation (RAG)** prototype built with:

- **LangChain** for orchestration  
- **OpenAI** for embeddings and generation  
- **FAISS** as the vector store  
- **Streamlit** for the UI  

The knowledge base is a local copy of the **Machine Learning Wikipedia page**.
All responses are grounded on that source.
            """
        )

        st.markdown("---")

        st.markdown("### How to use")
        st.markdown(
            """
- Ask a question about machine learning  
- The system retrieves relevant text chunks  
- The LLM generates a grounded answer using those chunks  
            """
        )


def render_chat_messages():
    # Inject styles for message bubbles + source tags
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.1rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1rem;
            color: #555;
            margin-bottom: 1.5rem;
        }
        .source-tag {
            display: inline-block;
            padding: 0.18rem 0.60rem;
            margin-right: 0.25rem;
            margin-bottom: 0.25rem;
            border-radius: 999px;
            border: 1px solid #888;
            font-size: 0.75rem;
            color: #444;
            background-color: #f5f5f5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-header">Machine Learning Q&A</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        "Ask focused questions about machine learning. Answers are generated using a RAG pipeline."
        "</div>",
        unsafe_allow_html=True,
    )

    # Display conversation
    for msg in st.session_state["messages"]:
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])

        if hasattr(st, "chat_message"):
            with st.chat_message("user" if role == "user" else "assistant"):
                st.markdown(content)

                if role == "assistant" and sources:
                    st.markdown("**Sources**")
                    src_tags = "".join(
                        f'<span class="source-tag">{src}</span>' for src in sources
                    )
                    st.markdown(src_tags, unsafe_allow_html=True)
        else:
            # Fallback for older Streamlit
            st.markdown(f"**{role.capitalize()}:** {content}")
            if sources:
                st.markdown("**Sources:** " + ", ".join(sources))


def handle_user_input(user_input: str):
    # Store user message
    st.session_state["messages"].append(
        {"role": "user", "content": user_input, "sources": []}
    )

    # Call RAG backend
    with st.spinner("Retrieving context and generating answer..."):
        try:
            result = ask_question(user_input)
            answer = result.get("answer", "")
            sources = result.get("sources", [])
        except Exception as e:
            answer = f"An error occurred: {e}"
            sources = []

    # Store assistant message
    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


def main():
    init_session_state()
    render_sidebar()

    # 1) Get user input first
    if hasattr(st, "chat_input"):
        user_input = st.chat_input("Ask a question about machine learning")
    else:
        user_input = st.text_input("Ask a question about machine learning")

    if user_input:
        handle_user_input(user_input)

    # 2) Then render messages (including the latest answer)
    render_chat_messages()


if __name__ == "__main__":
    main()
