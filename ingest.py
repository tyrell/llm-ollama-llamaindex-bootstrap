import weaviate
from llama_index import StorageContext, SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index.vector_stores import WeaviateVectorStore
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import box
import yaml

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_documents(docs_path):
    """
    This function retrieves and returns a list of JPEG documents from a specified directory.

    It utilizes the `SimpleDirectoryReader` class from the `pathlib` module to efficiently iterate through all files
    with the `.jpg` extension within the provided directory. The function then prints the total number of loaded documents
    before returning them as a list.

    Args:
        docs_path: The absolute or relative path to the directory containing the target JPEG documents.

    Returns:
        A list of `SimpleDirectoryReader` objects, each representing an individual JPEG file within the directory.

    Raises:
        FileNotFoundError: If the specified directory path is invalid or inaccessible.
        OSError: If an error occurs during file iteration or processing.

    """
    documents = SimpleDirectoryReader(docs_path, required_exts=[".jpg"]).load_data()
    print(f"Loaded {len(documents)} documents")
    return documents


def load_embedding_model(model_name):
    """
    Creates and returns a LangchainEmbedding object based on a specified Hugging Face model name.

    Args:
        model_name: The string identifier of the desired pre-trained embedding model from Hugging Face.

    Returns:
        A LangchainEmbedding object configured with the chosen Hugging Face model.

    Notes:
        This function utilizes the `LangchainEmbedding` wrapper around the underlying `HuggingFaceEmbeddings` class.

    """
    embeddings = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name=model_name)
    )
    return embeddings


def build_index(weaviate_client, embed_model, documents, index_name):
    """
    Constructs and populates a Weaviate Vector Store index with embedded document representations.

    This function takes several inputs and performs the following tasks:

    1. **Context Creation:**
        - Constructs a `ServiceContext` object using the provided embedding model and sets `llm` to None (assuming no language model involved).
        - Creates a `WeaviateVectorStore` object for accessing the specified Weaviate client and index name.
        - Builds a `StorageContext` based on the generated vector store.

    2. **Index Building:**
        - Utilizes the `VectorStoreIndex.from_documents` method to construct an index from the provided list of documents.
        - This process involves embedding each document using the injected `embed_model` within the service context.
        - The generated vector representations are then stored in the specified Weaviate index through the storage context.

    3. **Index Return:**
        - Finally, the function returns the constructed `VectorStoreIndex` object, representing the populated Weaviate index.

    Args:
        weaviate_client: An instance of the Weaviate client for accessing the target server.
        embed_model: An object capable of generating vector representations for the documents.
        documents: A list of documents to be indexed in Weaviate.
        index_name: The name of the Weaviate Vector Store index to populate.

    Returns:
        A `VectorStoreIndex` object representing the newly created and populated Weaviate index.

    Notes:
        This function assumes that the `embed_model` and provided documents are compatible for generating suitable vector representations.
        Additionally, ensure the Weaviate client has access and appropriate permissions to the specified index.

    """
    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=None)
    vector_store = WeaviateVectorStore(weaviate_client=weaviate_client, index_name=index_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        service_context=service_context,
        storage_context=storage_context,
    )

    return index


if __name__ == "__main__":
    
    # Import configuration specified in config.yml
    with open('config.yml', 'r', encoding='utf8') as configuration:
        cfg = box.Box(yaml.safe_load(configuration))

    print("Connecting to Weaviate")
    client = weaviate.Client(cfg.WEAVIATE_URL)

    print("Loading documents...")
    documents = load_documents(cfg.DATA_PATH)

    print("Loading embedding model...")
    embeddings = load_embedding_model(model_name=cfg.EMBEDDINGS)

    print("Building index...")
    index = build_index(client, embeddings, documents, cfg.INDEX_NAME)

