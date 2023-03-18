# This is a weird app

This app uses ChatGPT 3.5 API and your diary to answer questions as if it's you.

Install using these steps
`pip install -r requirements.txt`
`export OPENAI_API_KEY=your_api_key`
`update name variable in main.py to your name`
`update folder_path to your Markdown files`
After, just update the `prompt` var to ask yourself something.

### Things to calibrate
1. The threshold of the embeddings `filter_relevant_entries`. 
2. The temperature of the response on `generate_response`.

PS: This is the first time I've ever worked in Python. This app is based on asking Chat GPT 4 questions on "How to accomplis X?"