"""Lesson 07 — RAG with local embeddings + Chroma.

Retrieval-Augmented Generation: embed your documents into a vector store, retrieve the
most relevant chunks for a question, and stuff them into the prompt so the model answers
from YOUR data.

Groq has no embeddings API, so we embed locally with a small HuggingFace model
(all-MiniLM-L6-v2). The first run downloads the model weights (~90 MB), then caches them.

Run:  uv run python examples/07_rag_local_embeddings.py
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from agentlab.config import get_chat_model, get_embeddings

# A tiny in-code "knowledge base". In practice you'd load PDFs / web pages / docs
# and split them into chunks with a TextSplitter.
DOCS = [
    Document(page_content="Groq is an inference company whose LPU hardware runs LLMs at very high token throughput."),
    Document(page_content="The langchain-groq package exposes ChatGroq, which supports bind_tools() and with_structured_output()."),
    Document(page_content="Groq does not offer an embeddings API, so RAG apps use local or third-party embedding models."),
    Document(page_content="llama-3.3-70b-versatile and llama-3.1-8b-instant are popular free Groq chat models with 128K context."),
    Document(page_content="LangGraph's create_react_agent builds a tool-calling agent loop out of a model and a list of tools."),
]


def main() -> None:
    llm = get_chat_model()
    embeddings = get_embeddings()

    print("Building the vector store (downloads embed model on first run)...")
    # from_documents embeds each doc and stores the vectors. Kept in memory here.
    vectorstore = Chroma.from_documents(DOCS, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # A classic RAG prompt: answer ONLY from the retrieved context.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the question using ONLY the context below. If it's not in the context, say you don't know.\n\nContext:\n{context}"),
            ("human", "{question}"),
        ]
    )

    def format_docs(docs) -> str:
        return "\n".join(f"- {d.page_content}" for d in docs)

    # LCEL chain: retrieve -> format -> prompt -> model -> string.
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    for question in [
        "Does Groq provide embeddings?",
        "What can ChatGroq do?",
        "Who won the 2018 World Cup?",  # not in the docs -> should say it doesn't know
    ]:
        print(f"\nQ: {question}")
        print(f"A: {rag_chain.invoke(question)}")


if __name__ == "__main__":
    main()
