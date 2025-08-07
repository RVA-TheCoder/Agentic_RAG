import streamlit as st
import os
from markdown import markdown  # Add this import at the top of your file

from src.agentic_rag.rag_input_preprocessor.text_preprocessing import TextPreprocessing
from src.agentic_rag.constants.constants import pages_and_chunks_df_filepath, sentences_per_chunk
from src.agentic_rag.utils_methods.basic_methods import (format_llm_output,
                                                         device_availabilty ,
                                                         convert_markdown_to_html,
                                                         get_model_mem_size)

from src.agentic_rag.embedding_model.embedding_model import EmbeddingModel

from src.agentic_rag.constants.constants import (embeddings_df_save_path,
                                                 is_min_token_length_required,
                                                 #topk_resources_to_return,
                                                 weburl_filename_path,
                                                 URL_embeddings_df_save_path,
                                                 embedding_model_name,
                                                 local_embed_model_dir,
                                                 #llm_model_id,
                                                 #local_dir_llm,
                                                 topk_semantic_results,
                                                 max_output_tokens_final,
                                                 model_name)

from src.agentic_rag.semantic_search.semantic_search import SemanticSearch
from src.agentic_rag.generate_text.augmented_generation import AugmentedGeneration
from src.agentic_rag.ask_from_url.ask_from_url import AskUrl


import torch
#Frees unused GPU memory (non-destructive)
torch.cuda.empty_cache()
# Resets stats for profiling (no runtime impact)
torch.cuda.reset_peak_memory_stats()


# ──────────────────────────────────────────────────────────────
# Initialize
device = device_availabilty()
semantic_search_object = SemanticSearch()

# Load LLM and tokenizer
@st.cache_resource(show_spinner="Loading LLM and tokenizer...")
def load_tokenizer_and_llm(device, model_name,):
    
    from src.agentic_rag.tokenizer_and_llm_model.tokenizer_and_llm import Tokenizer_and_LLM
    
    # tokenizer_and_llm_object = Tokenizer_and_LLM(
    #     device=device,
    #     model_id=model_id,
    #     local_dir=local_dir,
    #     use_quantization_config=True,
    # )
    
    tokenizer_and_llm_object = Tokenizer_and_LLM(
                                                device=device,
                                                model_name=model_name,   # phi3 or gemma2b
                                                
                                                )
    
    return tokenizer_and_llm_object.tokenizer_n_LLM()

#model_id = "google/gemma-2-2b-it"
#local_dir = "HF_LLM_Models/gemma"
tokenizer, llm_model = load_tokenizer_and_llm(device, model_name=model_name,)
print(get_model_mem_size(model=llm_model))

# ──────────────────────────────────────────────────────────────
# Session State Initialization
if "data_ready" not in st.session_state:
    st.session_state.data_ready = False
if "embedding_data" not in st.session_state:
    st.session_state.embedding_data = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None
if "text_df" not in st.session_state:
    st.session_state.text_df = None
if "mode" not in st.session_state:
    st.session_state.mode = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ──────────────────────────────────────────────────────────────
# UI
st.image("agentic rag resize.png", use_container_width=True)
st.subheader("Agentic RAG LLM Application 🤖", divider="rainbow")

# Query Input
user_query = st.text_input("Ask a question about the content of your file:")

# Sidebar Input
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "txt"])
    url = st.text_input("Enter a web URL:", placeholder="Paste Web URL", key="url_input")
    add_data = st.button("Add Data")

    if st.session_state.mode:
        mode_label = "🌐 URL" if st.session_state.mode == "url" else "📄 File"
        st.info(f"**Current Mode:** {mode_label}")

# ──────────────────────────────────────────────────────────────

# Clear chat history helper
def clear_chat_history():
    st.session_state.chat_history = []


# Data Handling
if add_data:
    # Clear previous chat history
    #st.session_state.history = ""
    clear_chat_history()
    
    if uploaded_file and url.strip():
        st.error("❌ Please use only one input method.")
        st.session_state.data_ready = False

    elif uploaded_file:
        original_name = uploaded_file.name
        new_file_name = f"uploaded_{original_name}"
        filepath = os.path.abspath(new_file_name)

        with open(filepath, "wb") as f:
            f.write(uploaded_file.read())

        # Preprocessing
        text_preprocessing = TextPreprocessing(
                                    filepath=filepath,
                                    sentences_per_chunk=sentences_per_chunk,
                                    is_min_token_length_required=is_min_token_length_required,
                                    pages_and_chunks_df_filepath=pages_and_chunks_df_filepath,
                                    save_pages_and_chunks_df=True,
                                )
        
        pages_and_chunks = text_preprocessing.run_pipeline()

        # Embedding
        embedding_model_object = EmbeddingModel(
                                embedding_model_name=embedding_model_name,
                                local_embed_model_dir=local_embed_model_dir,
                                device=device,
                                embeddings_df_save_path=embeddings_df_save_path,
                            )

        embedding_model, embeddings, _, df = embedding_model_object.run_embedding_pipeline(
                                                pages_and_chunks=pages_and_chunks, replace=True
                                            )

        # Save session state 
        with st.sidebar:
            st.session_state.embedding_model = embedding_model
            st.session_state.embedding_data = embeddings
            st.session_state.text_df = df
            st.session_state.data_ready = True
            st.session_state.mode = "file"
            st.success("✅ File processed successfully!")

    elif url.strip():
        
        ask_url = AskUrl(
                        web_url=url,
                        save_pages_and_chunks_df=True,
                        sentences_per_chunk=sentences_per_chunk,
                        is_min_token_length_required=is_min_token_length_required,
                        filename=weburl_filename_path,
                    )
        
        url_pages_and_chunks = ask_url.web_text_preprocessing()

        embedding_model_object = EmbeddingModel(
                                        embedding_model_name=embedding_model_name,
                                        local_embed_model_dir=local_embed_model_dir,
                                        device=device,
                                        embeddings_df_save_path=URL_embeddings_df_save_path,
                                    )

        embedding_model, url_embeddings, url_text_chunks_and_embeddings, url_df = embedding_model_object.run_embedding_pipeline(
                                            pages_and_chunks=url_pages_and_chunks,
                                            replace=True
                                        )
        

        with st.sidebar:
            st.session_state.embedding_model = embedding_model
            st.session_state.embedding_data = url_embeddings
            st.session_state.text_df = url_df
            st.session_state.data_ready = True
            st.session_state.mode = "url"
            st.success("✅ URL processed successfully!")

    else:
        st.warning("⚠️ Please upload a file or enter a URL.")
        st.session_state.data_ready = False

# ──────────────────────────────────────────────────────────────
# Display Current Mode
if st.session_state.mode:
    st.markdown(f"**Mode:** {'🌐 Web URL Mode' if st.session_state.mode == 'url' else '📄 File Upload Mode'}")

# ──────────────────────────────────────────────────────────────
# Inference
if user_query and st.session_state.data_ready:
    
    context, context_items, scores, indices = semantic_search_object.run_pipeline(
                                            user_query=user_query,
                                            embeddings=st.session_state.embedding_data,
                                            embedding_model=st.session_state.embedding_model,
                                            pages_n_chunks_embeddings_df=st.session_state.text_df,
                                            device=device,
                                            print_relevant_resources=True,
                                            topk_semantic_results=topk_semantic_results
                                        )

    aug_gen = AugmentedGeneration(
                                    device=device,
                                    embedding_model=st.session_state.embedding_model,
                                    tokenizer=tokenizer,
                                    llm_model=llm_model,
                                    context=context,
                                    max_output_tokens_final=max_output_tokens_final,
                                )

    raw_answer = aug_gen.get_final_answer(question=user_query, context=context)
    cleaned_answer = format_llm_output(answer=raw_answer)

    st.markdown("**LLM Response:**")
    st.markdown(cleaned_answer)
    st.divider()



    formatted_answer_html = convert_markdown_to_html(cleaned_answer)

    qa_pair_html = f"""
    <div style="padding: 10px; margin-bottom: 10px; border: 1px solid #ddd; border-radius: 6px; background-color: #f9f9f9;">
    <p><strong>Q:</strong> {user_query}</p>
    <strong>A:</strong>
    {formatted_answer_html}
    </div>
    """

    st.session_state.chat_history.append(qa_pair_html)

    

from streamlit.components.v1 import html as st_html   
# Chat History Scrollable Display
if st.session_state.chat_history:
    st.markdown("##### Chat History:")
    
    chat_html = "".join(st.session_state.chat_history[::-1])

    st_html(
        f"""
        <div style='height:300px; overflow-y:scroll; padding:10px; border:1px solid #ddd; border-radius:6px; background-color:#ffffff;'>
            {chat_html}
        </div>
        """,
        height=320,
    )
    
    
    
    
   
    
    