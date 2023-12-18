# Retrieve-Augment Generative (RAG) Template Application

Designed for offline use, this RAG application template is based on Andrej Baranovskij's tutorials. It offers a starting point for building your own local RAG pipeline, independent of online APIs and cloud-based LLM services like OpenAI. This empowers developers to experiment and deploy RAG applications in controlled environments.

## The Stack

This RAG application runs entirely offline, utilizing your local CPU to generate/retrieve/rank responses without needing internet access. This RAG deployment relies solely on your local CPU for computation. Please note that processing large datasets or using resource-intensive models might slow down performance.

1. Large Language Model - We use Ollama (https://ollama.ai/) to run our LLM locally. Any model supported by Ollama can be configured to be used using the config.yml file found in this application. 
2. Vector Store - We use Weaviate (https://weaviate.io) as the Vector Store. We run Weaviate as a Docker container The URL of the Weaviate instance can be configured using the config.yml file.
3. Index - We use LlamaIndex (https://www.llamaindex.ai) as the core of this RAG application acting as the index of our private data structures. The sample code provided by the template ingests a file into the index. 
4. Vector Embeddings - We use Langchain (https://www.langchain.com) and HuggingFace (https://huggingface.co) to maintain a local embedding model.

___

## Quickstart

1. Run the Weaviate local Vector Store with Docker:
   
```
docker compose up -d
```

2. Install Python requirements: 

```
pip install -r requirements.txt
```

3. Install <a href="https://ollama.ai">Ollama</a> and pull the preferred LLM model specified in config.yml

4. Copy text PDF files to the `data` folder

5. Run the script, to convert text to vector embeddings and save in Weaviate: 

```
python ingest.py
```

1. Run main.py to process data with LLM RAG pipeline defined in pipeline.py and return the answer: 

```
python main.py "Who are you?"
```

Answer:

```
Answer:
I am an AI language model, designed to assist and provide information based on the context provided. In this case, the context is related to an invoice from Chapman, Kim and Green to Rodriguez-Stevens for various items such as wine glasses, stemware storage, corkscrew parts, and stemless wine glasses.

Here are some key details from the invoice:
- Invoice number: 61356291
- Date of issue: 09/06/2012
- Seller: Chapman, Kim and Green
- Buyer: Rodriguez-Stevens
- VAT rate: 10%

The invoice includes several items with their respective quantities, unit measures (UM), net prices, net worth, gross worth, and taxes. The summary section provides the total net worth, VAT amount, and gross worth of the invoice.
==================================================
Time to retrieve answer: 37.36918904201593

```

You can find more prompts in prompts.txt to test the template application. Once yo have read through the codebase, expand the RAG to your specific needs.

# License
Apache 2.0


~ Tyrell Perera 
