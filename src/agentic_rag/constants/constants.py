# Sentence chunks
sentences_per_chunk : int = 6




# For File Input
pages_and_chunks_df_filepath:str = "pages_and_chunks_df.csv"
embeddings_df_save_path : str = "text_chunks_and_embeddings_df.csv"
on_fly_embeddings_df_save_path : str = "ON_FLY_text_chunks_and_embeddings_df.csv"

text_wrap_length : int = 80
topk_resources_to_return : int = 5

ONFLY_topk_resources_to_return : int = 3

# Tokenizer and LLM details
model_id :str = "google/gemma-2-2b-it"
local_dir: str = "src/agentic_rag/HF_LLM_Models/gemma"


METAPHOR_API_KEY = "a4ecdad2-9a0c-416e-8eaa-a36f1f365842"




# For URL Input
weburl_filename_path : str = "cleaned_weburl_text.pdf"
URL_pages_and_chunks_df_filepath:str = "URL_pages_and_chunks_df.csv"
URL_embeddings_df_save_path : str =  "URL_text_chunks_and_embeddings_df.csv"


