# This is a weird app

This app uses ChatGPT 3.5 API and your diary to answer questions as if it's you.

Install using these steps
```
pip install -r requirements.txt
export OPENAI_API_KEY=your_api_key
update folder_path to your Markdown files 
python main.py "Are you happy?"
```

### How it works
The script performs the following steps:

- It reads all markdown files from a specified directory and extracts the text content from them using the markdown_to_text() function.
- It then uses the Sentence Transformer library to encode them into dense vector representations using the get_embeddings() function.
- It saves the diary entry embeddings to disk using the save_embeddings_to_disk() function.
- It uses cosine similarity to filter the most relevant diary entries that match a given prompt using the filter_relevant_entries() function.
- It combines the filtered diary entries into a single string and generates a response to the prompt using OpenAI's API via the generate_response() function.

### How to calibrate
1. The threshold of the embeddings `filter_relevant_entries` — greater means more entries can be matched
2. The temperature of the response on `generate_response`. 

PS: This is the first time I've ever worked in Python. This app is based on asking Chat GPT 4 questions on "How to accomplish X?"