# For File Input
pages_and_chunks_df_filepath:str = "pages_and_chunks_df.csv"
embeddings_df_save_path : str = "text_chunks_and_embeddings_df.csv"
on_fly_embeddings_df_save_path : str = "ON_FLY_text_chunks_and_embeddings_df.csv"

text_wrap_length : int = 80
#topk_resources_to_return : int = 5


# Document Chunking details :
# Sentence chunks
sentences_per_chunk : int = 8
# There is high probability that these chunks represent : Url, Chapter heading , fig. description etc.,
is_min_token_length_required : bool = False
min_token_length = 30



# EMbedding Model details :
embedding_model_name="all-mpnet-base-v2"
local_embed_model_dir="HF_Embedding_Models"



# Tokenizer and LLM details
model_name : str = "gemma2b"   # phi3 or gemma2b, gemma3_4b
#llm_model_id :str = "google/gemma-2-2b-it"
#local_dir_llm: str = "src/agentic_rag/HF_LLM_Models/gemma"
max_output_tokens_initial_decision : int = 2 
max_output_tokens_final : int = 192 # 192, 256, 128, 



# Sematic Search details:
topk_semantic_results : int = 20


# Reranker model HuggingFace details:
reranker_model_name: str = "BAAI/bge-reranker-base"
reranker_model_local_dir: str = "./HF_Reranker_Model"
topk_reranker_results : int = 5




# For URL Input
weburl_filename_path : str = "cleaned_weburl_text.pdf"
URL_pages_and_chunks_df_filepath:str = "URL_pages_and_chunks_df.csv"
URL_embeddings_df_save_path : str =  "URL_text_chunks_and_embeddings_df.csv"
weburl_text_output_json : str = "weburl_text_output.json"
URL_sentences_per_chunk : int = 3


# Online Search ON-FLY
METAPHOR_API_KEY = "a4ecdad2-9a0c-416e-8eaa-a36f1f365842"
ONFLY_topk_resources_to_return : int = 10
ONFLY_topk_URLs_to_return : int = 2






