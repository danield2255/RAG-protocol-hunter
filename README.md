# RAG-protocol-hunter
A basic RAG architecture to respond to technical documentation and guideline questions based on a given corpus of documents.


### Setup 
We will use poetry to to manage packages and setup. You can use 
```bash
poetry install
```
to manage getting you the necessary packages and then we will use poetry to run our code. 

We wil be using open source huggingface models for the tool, so you will likely need to run 
```bash
huggingface-cli login
```
This will prompt you to enter or create a huggingface token. 


You will move then into the ```src``` folder and then 
```bash
cd src
poetry run python3 query_assistant.py 
```

This will then output a response to your question and then prompt you to ask another question if you want. 
