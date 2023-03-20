import os
import pickle
from pathlib import Path

from bs4 import BeautifulSoup
import mistune
import openai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


os.environ['TOKENIZERS_PARALLELISM'] = 'false'
openai.api_key = os.environ.get("OPENAI_API_KEY")
name = os.environ.get("USER")

def markdown_to_text(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    parser = mistune.create_markdown(renderer=mistune.HTMLRenderer())
    html = parser(content)
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text()

def get_embeddings(texts, model):
    return model.encode(texts)

def save_embeddings_to_disk(embeddings, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings_from_disk(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def filter_relevant_entries(prompt, diary_entries, entry_embeddings, model, threshold=0.3):
    prompt_embedding = get_embeddings([prompt], model)[0]
    similarity_scores = cosine_similarity([prompt_embedding], entry_embeddings)
    relevant_entry_indices = [i for i, score in enumerate(similarity_scores[0]) if score > threshold]
    return [diary_entries[i] for i in relevant_entry_indices]

def generate_response(prompt, input_text, model_engine="gpt-3.5-turbo"):
    if not input_text:
        return "Error: No relevant diary entries found. Cannot generate a response."

    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=[
            {
                "role": "system",
                "content": f"You are a clone of {name} that uses entries from their diary to answer questions in their style. Your purpose is to provide insightful and accurate responses based on the given diary entries."
            },
            {
                "role": "user",
                "content": f"Here are some relevant diary entries:\n{input_text}"
            },
            {
                "role": "user",
                "content": f"What is your response to this question: {prompt}?"
            },
            {
                "role": "user",
                "content": f"Remember to answer as if you are {name}, and do not mention that you are a clone or talk in the third person. Do not say 'As {name}'."
            }
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def load_markdown_files(folder_path):
    return [
        file_path
        for file_path in folder_path.glob("*.md")
        if "excalidraw" not in file_path.read_text()
    ]

def process_user_input(prompt, diary_entries, entry_embeddings, model):
    relevant_entries = filter_relevant_entries(prompt, diary_entries, entry_embeddings, model)
    filtered_diary = "\n".join(relevant_entries)
    filtered_diary = filtered_diary[:5000]
    response = generate_response(prompt, filtered_diary)
    return response

def main_loop(model, diary_entries, entry_embeddings):
    while True:
        try:
            prompt = input("\nEnter a prompt or type 'exit' to quit: ")
            if prompt.lower() == "exit":
                break
            response = process_user_input(prompt, diary_entries, entry_embeddings, model)
            print(f"\n{name}: {response}")
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected. Exiting the loop...")
            break

def test_models_on_questions(model_names, questions, diary_entries):
    embeddings_cache = {}

    for model_name in model_names:
        print(f"Testing model: {model_name}")
        model = SentenceTransformer(model_name)

        # Generate or load embeddings for the current model
        embeddings_file_path = f"diary_embeddings_{model_name}.pkl"
        if os.path.exists(embeddings_file_path):
            entry_embeddings = load_embeddings_from_disk(embeddings_file_path)
        else:
            print(f"Building the index for {model_name}. This is a one-time process and might take a few minutes\n")
            entry_embeddings = get_embeddings(diary_entries, model)
            save_embeddings_to_disk(entry_embeddings, embeddings_file_path)
        embeddings_cache[model_name] = entry_embeddings

        # Test the model on the given questions
        for question in questions:
            response = process_user_input(question, diary_entries, entry_embeddings, model)
            print(f"\nQuestion: {question}")
            print(f"\033[32mAnswer ({model_name}): {response}\033[0m\n")


def main():
    model_names = [
        "paraphrase-mpnet-base-v2",
        "distilroberta-base-paraphrase-v1",
        "paraphrase-distilroberta-base-v1",
        "paraphrase-TinyBERT-L6-v2",
        "paraphrase-MiniLM-L12-v2",
        "paraphrase-xlm-r-multilingual-v1",
        "stsb-roberta-base",
        "stsb-bert-base",
        "nli-roberta-base",
        "nli-bert-base",
        "multi-qa-MiniLM-L6-cos-v1",
        "nq-distilbert-base-v1",
        "all-mpnet-base-v2"
    ]

    questions = [
        "Ask a very personal question to see how the model behaves"
    ]

    folder_path = Path('/Users/alentodorov/Library/Mobile Documents/iCloud~md~obsidian/Documents/diary')
    markdown_files = load_markdown_files(folder_path)
    diary_entries = [markdown_to_text(file) for file in markdown_files]

    test_models_on_questions(model_names, questions, diary_entries)

if __name__ == "__main__":
    main()

