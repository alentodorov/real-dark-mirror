import os
import mistune
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import pickle

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
openai.api_key = os.environ.get("OPENAI_API_KEY")
name = "Your Name"

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

def filter_relevant_entries(prompt, diary_entries, entry_embeddings, model, threshold=0.2):
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
                "content": f"You are a clone of {name} that uses entries from their diary to answer questions in their style",
            },
            {
                "role": "user",
                "content": (
                    f"Based on these diary entries"
                    f"{input_text}"
                    f"what is the response to this question: {prompt}."
                    f"Remember to answer as if you are {name} and do not mention that you are a clone or talk in the third person. "
                    f"Do not say 'As {name}'"
                ),
            },
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content

# Load the Sentence Transformer model
model = SentenceTransformer("paraphrase-mpnet-base-v2")

# Read markdown files from a folder
folder_path = r'/Users/alentodorov/Library/Mobile Documents/iCloud~md~obsidian/Documents/diary'
markdown_files = [
    os.path.join(folder_path, f)
    for f in os.listdir(folder_path)
    if f.endswith(".md") and "excalidraw" not in open(os.path.join(folder_path, f)).read()
]


# Combine all entries into a list
diary_entries = [markdown_to_text(file) for file in markdown_files]

# Check if embeddings file exists, if not, create and save the embeddings
embeddings_file_path = "diary_embeddings.pkl"

if os.path.exists(embeddings_file_path):
    entry_embeddings = load_embeddings_from_disk(embeddings_file_path)
else:
    entry_embeddings = get_embeddings(diary_entries, model)
    save_embeddings_to_disk(entry_embeddings, embeddings_file_path)

# Prompt
prompt = "What do you like to do?"

# Filter relevant entries
relevant_entries = filter_relevant_entries(prompt, diary_entries, entry_embeddings, model)

# Combine relevant entries into a single string
filtered_diary = "\n".join(relevant_entries)

# Keep only the first 4000 characters so that it fits the model
filtered_diary = filtered_diary[:4000]

# print(filtered_diary)

# Generate a response using OpenAI API
response = generate_response(prompt, filtered_diary)
print(response)
