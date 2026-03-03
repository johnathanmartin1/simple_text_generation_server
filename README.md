# A Simple Local Server For Text Generation and Interactive Graph

Utilising the huggingface gpt2 model and fastapi to produce a local server. The rest api will take an input text and utilise the LLM to generate output text. The local server  also contain links that utilise the rest api to navigate between pages. The graph page contains an interactive plot form another project.

Ensure all the packages form the requirments.txt folder are installed via:

pip install -r requirements.txt

Then the local server can be run with command prompt (opened in the same directory as the python app file) using :

python3 -m uvicorn app:app --reload

Then open a browser to the address that uvicorn is running (eg http://127.0.0.1:8000)


