"""
Grounded inference: retrieves from HippoRAG, then feeds the retrieved passages to
the chat model as context before it answers -- unlike the earlier version, where
retrieval was only printed alongside an unrelated raw chat call.

Reading level and topic scope (below) are applied as instructions on top of the
original, un-simplified index (outputs/grays_anatomy_ollama), not by pre-simplifying
the indexed text -- that keeps retrieval and entity linking working against the
book's real wording, and lets you change reading level or scope at any time without
re-indexing.

Requirements:
    pip install openai
    ollama serve   (with OLLAMA_CONTEXT_LENGTH set high enough for your prompts --
    retrieved passages add to what the model has to read, so this needs headroom)
    SAVE_DIR below must already contain a HippoRAG index. openie_list_shim.py must
    sit next to this file.

Usage:
    python infer.py                                  # interactive chat loop
    python infer.py "List the bones of the wrist"     # single prompt, print, exit

Set USE_RETRIEVAL = False to fall back to plain LLM-only inference (no HippoRAG,
no grounding).
"""

import sys

from openai import OpenAI

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "qwen2.5:7b-instruct"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
SAVE_DIR = "outputs/grays_anatomy_ollama"   # the original technical index

USE_RETRIEVAL = True
NUM_TO_RETRIEVE = 5
SHOW_RETRIEVED_PASSAGES = True   # print what was retrieved, for your own visibility

IMPORTANT = (
"""
The passages above are the complete and only source of information for your answer.

Use ONLY facts that are explicitly stated in the passages.
Do NOT use your own knowledge, training knowledge, or information from any other source.
Do NOT guess, infer, assume, or fill in missing information.

If the passages do not contain enough information to answer the question,
say exactly: "The text does not cover this."

Answer the question directly and factually.

Use simple vocabulary and short sentences.
Use no more than 3 sentences.
Do not address the reader.
Do not use a conversational, playful, or teaching style.
If a medical term is necessary, keep the term and briefly explain it in simple words.

Do not explain your reasoning.
Do not repeat the question.
"""
)

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")  # placeholder key; Ollama ignores it

hipporag = None
if USE_RETRIEVAL:
    import openie_list_shim  # noqa: F401  (only needed if retrieve() ever re-runs OpenIE)
    from hipporag import HippoRAG

    print(f"Loading HippoRAG index from {SAVE_DIR} ...")
    hipporag = HippoRAG(
        save_dir=SAVE_DIR,
        llm_model_name=MODEL,
        llm_base_url=OLLAMA_BASE_URL,
        embedding_model_name=EMBEDDING_MODEL,
        embedding_provider="openai",
        embedding_base_url=OLLAMA_BASE_URL,
    )


def retrieve_passages(query):
    """Return the top passages HippoRAG's graph retrieval finds for this query."""
    result = hipporag.retrieve(queries=[query], num_to_retrieve=NUM_TO_RETRIEVE)[0]
    if SHOW_RETRIEVED_PASSAGES:
        print(f"\n--- HippoRAG retrieval (top {NUM_TO_RETRIEVE}) ---")
        for i, (doc, score) in enumerate(zip(result.docs, result.doc_scores), 1):
            snippet = doc[:200].replace("\n", " ")
            print(f"  {i}. [{score:.8f}] {snippet}...")
        print("--- end retrieval ---\n")
    return result.docs


def build_user_message(question, passages):
    """Wrap the question with retrieved passages as context, if there are any."""
    if not passages:
        return question
    context = "\n\n".join(f"Passage {i}: {p}" for i, p in enumerate(passages, 1))
    return f"Passages:\n{context}\n\nQuestion: {question}\n\nImportant: {IMPORTANT}"


def ask(messages, stream=True):
    """Send a message list to the model. Prints as it streams and returns the full text."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=stream,
    )

    if not stream:
        text = response.choices[0].message.content or ""
        print(text)
        return text

    chunks = []
    for event in response:
        delta = event.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            chunks.append(delta)
    print()
    return "".join(chunks)


def answer(question, history=None):
    """Retrieve, build a grounded prompt, and answer. Returns (user_msg, answer_text)."""
    passages = retrieve_passages(question) if hipporag is not None else []
    user_msg = build_user_message(question, passages)

    messages = list(history or [])
    messages.append({"role": "user", "content": user_msg})

    print("Model: ", end="", flush=True)
    answer_text = ask(messages)
    return user_msg, answer_text


def main():
    if len(sys.argv) > 1:
        # Single-shot mode: one prompt from argv, print the answer, exit.
        question = " ".join(sys.argv[1:])
        answer(question)
        return

    # Interactive mode: keeps conversation history until you type 'exit' or Ctrl+C.
    print(f"Chatting with {MODEL} via {OLLAMA_BASE_URL}. Type 'exit' to quit.\n")
    history = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        user_msg, answer_text = answer(user_input, history)
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": answer_text})
        print()


if __name__ == "__main__":
    main()