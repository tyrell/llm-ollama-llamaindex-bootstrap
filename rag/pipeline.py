from llama_index import VectorStoreIndex, ServiceContext
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.llms import Ollama
from llama_index.vector_stores import WeaviateVectorStore
import weaviate
import box
import yaml


def load_embedding_model(model_name):
    embeddings = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name=model_name)
    )
    return embeddings


def load_index(chunk_size, llm, embed_model, weaviate_client, index_name):
    service_context = ServiceContext.from_defaults(
        chunk_size=chunk_size,
        llm=llm,
        embed_model=embed_model
    )

    vector_store = WeaviateVectorStore(weaviate_client=weaviate_client, index_name=index_name)

    index = VectorStoreIndex.from_vector_store(
        vector_store, service_context=service_context
    )

    return index

def build_rag_pipeline():
    """
    Constructs and configures a RAG pipeline for retrieval-augmented generation tasks.

    This function performs the following steps to set up the RAG pipeline:

    1. **Configuration Loading:**
        - Reads configuration variables from a specified YAML file (`config.yml`).
        - Stores the loaded configuration as a `box.Box` object for convenient access.

    2. **Weaviate Connection:**
        - Establishes a connection to the Weaviate server using the provided URL in the configuration.
        - Creates a Weaviate client object for interacting with the Weaviate database.

    3. **LLAMA Model Loading:**
        - Loads the specified Ollama language model based on the `LLM` key in the configuration.
        - Sets the model temperature to 0 for a more deterministic response generation.

    4. **Embedding Model Loading:**
        - Utilizes the `load_embedding_model` function to retrieve a pre-trained Hugging Face model configured for Langchain.
        - This model will be used to embed documents and queries for efficient search and retrieval.

    5. **Vector Store Index Loading:**
        - Fetches the pre-built Weaviate Vector Store index named in the configuration (`INDEX_NAME`).
        - Connects the index to the Weaviate client and embeds relevant context using the selected service context.

    6. **Query Engine Construction:**
        - Converts the loaded Vector Store index into a dedicated query engine for efficient retrieval.
        - Sets the `streaming` flag to `False` to return the final response after the entire query is processed.

    7. **Pipeline Return:**
        - Returns the fully constructed and configured RAG pipeline represented by the `query_engine` object.

    Notes:
        - This function relies on a separate `config.yml` file for storing configuration values.
        - Ensure that the configuration file contains valid values for all required keys.

    """
    # Import configuration specified in config.yml
    with open('config.yml', 'r', encoding='utf8') as ymlfile:
        cfg = box.Box(yaml.safe_load(ymlfile))

    print("Connecting to Weaviate")
    client = weaviate.Client(cfg.WEAVIATE_URL)

    print("Loading Ollama...")
    llm = Ollama(model=cfg.LLM, temperature=0)

    print("Loading embedding model...")
    embeddings = load_embedding_model(model_name=cfg.EMBEDDINGS)

    print("Loading index...")
    index = load_index(cfg.CHUNK_SIZE, llm, embeddings, client, cfg.INDEX_NAME)

    print("Constructing query engine...")
    query_engine = index.as_query_engine(streaming=False)

    return query_engine
