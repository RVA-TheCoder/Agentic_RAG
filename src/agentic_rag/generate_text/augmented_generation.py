from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import streamlit as st
import re


from agentic_rag.utils_methods.basic_methods import online_search_duckduckgo, online_search_metaphor
from agentic_rag.get_url_content.fetch_cleaned_url_content import FetchURLContent

from agentic_rag.embedding_model.embedding_model import EmbeddingModel
from agentic_rag.constants.constants import (embeddings_df_save_path, 
                                             on_fly_embeddings_df_save_path,
                                             #topk_resources_to_return, 
                                             ONFLY_topk_resources_to_return,
                                             max_output_tokens_final,
                                             max_output_tokens_initial_decision,
                                             ONFLY_topk_URLs_to_return,
                                             embedding_model_name,
                                             local_embed_model_dir
                                             )

from agentic_rag.utils_methods.basic_methods import device_availabilty

from agentic_rag.semantic_search.semantic_search import SemanticSearch






class AugmentedGeneration:
    
    def __init__(self, tokenizer, llm_model, embedding_model, context, device, max_output_tokens_final=max_output_tokens_final ):
        
        
        self.device = device
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.context = context
        self.max_output_tokens_final = max_output_tokens_final
    
    
    def decision_prompt(self):
        
        """
        decision_prompt :  will check to see if the retrieved context can answer the user question.
        """
        
        ######## For gemma2b model  ##############
        # decision_system_prompt = """
        
        # Context : ```{context}```
        # Question : {question}
        
        # Your job is to decide if a given Question can be answered with a given Context.
        # - `1` if the Context contains enough information or sufficient to answer the Question completely.
        # If not, return 0.
        # Do not return anything except 0 or 1.
        
        # Use below examples for more clarity : 
        # Example 1 :
        # Context: The sky is blue.
        # Question: What color is the sky?
        # Answer: 1

        # Example 2 :
        # Context: Winston Churchill rallied the British people during World War II after the Dunkirk evacuation, emphasizing that though the battle ahead would be tough, Britain would never surrender. He declared that the nation would fight on the beaches, landing grounds, fields, streets, and hills, defending itself at all costs. The speech was a powerful call for resilience and unity in the face of Nazi aggression.
        # Question: Describe the document or text in brief?
        # Answer: 1


        # Example 3 :
        # Context: Artificial Intelligence is transforming industries by enabling machines to perform tasks that once required human intelligence—like learning, reasoning, and decision-making. From healthcare and education to transportation and finance, AI promises efficiency, innovation, and new opportunities. However, it also brings challenges such as ethical concerns, job displacement, and the need for responsible development. The future of AI depends on how wisely and inclusively we shape its path.
        # Question: Who is Bill gates?
        # Answer: 0
        
        
        # Answer : <your answer should come here>
        # """
        
        decision_system_prompt = """
        <bos><start_of_turn>user
        You are a helpful assistant.

        Your job is to decide if a given **Question** can be answered using the provided **Context**.

        Return:
        - `1` if the **Context** is sufficient to answer the **Question** completely.
        - `0` if the **Context** does **not** contain enough information to answer the **Question**.

        Do not return anything **except** 0 or 1 — no explanation, no text, just the number.

        Here are a few examples to guide you:

        Example 1:
        Context: The sky is blue.
        Question: What color is the sky?
        Answer: 1

        Example 2:
        Context: Winston Churchill rallied the British people during World War II after the Dunkirk evacuation, emphasizing that though the battle ahead would be tough, Britain would never surrender. He declared that the nation would fight on the beaches, landing grounds, fields, streets, and hills, defending itself at all costs.
        Question: Describe the document or text in brief?
        Answer: 1

        Example 3:
        Context: Artificial Intelligence is transforming industries by enabling machines to perform tasks that once required human intelligence—like learning, reasoning, and decision-making. From healthcare and education to transportation and finance, AI promises efficiency, innovation, and new opportunities. However, it also brings challenges such as ethical concerns, job displacement, and the need for responsible development.
        Question: Who is Bill Gates?
        Answer: 0

        Now evaluate the following:

        Context:
        ```{context}```

        Question:
        {question}

        Answer:
        <end_of_turn><start_of_turn>model
        """
        
        ########### For phi3 Model  #################
        # decision_system_prompt = """
        # Context: {context}
        # Question: {question}

        # Can the question be answered using the above context?

        # Return 1 if yes, else 0 
        
        # answer : 
        # """
        
        # decision_system_prompt = """
        # <|system|>
        # You are a helpful assistant.
        # Return 1 if the **Question** be answered using the provided **Context**, else return 0 
        # Do not return anything else.
        
        # Context:
        # {context}

        # Question:  
        # {question}
        # <|end|>
        
        # <|assistant|>
        # """
        
        
        # decision_system_prompt = """
        # <|system|>
        # Your job is to decide if a given Question can be answered with a given Context.
        # If the Context can answer the Question, return 1.
        # If not, return 0.
        # Do not return anything except 0 or 1.<|end|>
        
        # <|user|>
        # Context : {context}
        # Question : {question}
        # <|end|>
        
        # <|assistant|>
        # """
        
        
        return decision_system_prompt
    
    
    
    def final_prompt(self):
        
        """
        final_prompt :  will get the context and question and generate the response.
        """

        # final_system_prompt = """You are an expert for answering questions. Answer the question according only to the given context.
        # If question cannot be answered using the context, simply say I don't know. Do not make stuff up.
        # Your answer MUST be as informative, explanatory and action driven as possible. Your response must be in Markdown.
        # Consider taking into account the complete context provided before answering the question. 

        # Context: ```{context}```

        # Question: {question}

        # Answer :
        # """
        
        # final_system_prompt = """You are an expert at answering questions using only the provided context.
        # If the question cannot be answered using the context, simply respond with "I don't know." Do not make anything up.

        # Your answer MUST:
        # - Be in **Markdown** format
        # - Start with a short **introductory summary (3-4 sentences)**  
        # - Use **bullet points** for each key insight or fact
        # - Be as informative, explanatory, and action-driven as possible
        # - Only use information from the given context

        # Consider all parts of the context before answering.

        # Context:
        # ```{context}```

        # Question:
        # {question}

        # Answer:
        # """
        
        ######## For gemma2b model  ##############
        final_system_prompt = """You are an expert at answering questions using only the provided context.
        If the question cannot be answered using the context, simply respond with "I don't know." Do not make anything up.

        Your answer MUST :
        - Start with a short **introductory summary (3-4 sentences)**
        - Be as informative, explanatory, and action-driven as possible
        - Only use information from the given context
        - Consider all parts of the context before answering.
        - Be in **Markdown** format if required
        - Use **bullet points** for each key insight or fact if required
        
        
        Context:
        ```{context}```

        Question:
        {question}

        Answer:
        """
        
        ########### For phi3 Model  #################
        # final_system_prompt = """You are an expert assistant that answers questions **only using the given context**.
        # If the question cannot be answered using the context, respond with: **"I don't know."** Never make up information.

        # **Instructions:**
        # - Do not return the **context** only return summarization of the **context**
        # - starts with a **brief description (3–4 sentences)**.
        # - Be informative, clear, and action-oriented.
        # - Use only the provided context — consider all parts of it carefully.
        # - Format your answer using **Markdown**.
        # - Use **bullet points** for key facts or steps, if applicable.
        
        # ---

        # **Context:**
        # ```{context}```

        # **Question:**  
        # {question}

        # **Answer:**
        # """
        
        # final_system_prompt = """
        # <|system|>
        # You are an expert assistant. Use only the information from the provided context to answer the question.

        # **Important Rules:**
        # - Do NOT repeat or include the context in your response.
        # - If the answer is not in the context, respond only with: **"I don't know."**
        # - NEVER fabricate or guess.
        # - Do NOT mention or refer to the "context section" in your answer.

        # **Your answer MUST:**
        # - Use **bullet points** for each key insight or fact
        # - Be clear, concise, and well-structured.
        # - Format in **Markdown** when appropriate.
        # - ends with a **brief summary (3–4 sentences)**.

        # <|end|>

        # <|user|>
        # Context:
        # {context}<|end|>

        # <|user|>
        # Question:  
        # {question}<|end|>

        # <|assistant|> 
        # """
        
        
        return final_system_prompt
    
    
    
    def query_llm(self, prompt, max_output_tokens=max_output_tokens_final):
        
        
        generate_text = pipeline("text-generation", model=self.llm_model, tokenizer=self.tokenizer)
        response = generate_text(prompt, max_new_tokens=max_output_tokens, do_sample=False)
    
        return response[0]["generated_text"][len(prompt):].strip()
    
    
    
    def get_final_answer(self, question, context): # self.context
        
        """
        Decide if context is sufficient to answer the question. If yes, use local LLM.
        Otherwise, perform an online fallback search and answer using retrieved context.
        """

        # Step 1: Ask the decision system
        decision_system_prompt = self.decision_prompt()
        decision_input = decision_system_prompt.format(context=context, question=question)
        
        has_answer = self.query_llm(decision_input, max_output_tokens=max_output_tokens_initial_decision)

        print(f"\n[User Question]: {question}")
        print("Given context has answer or not : ", has_answer,"\n")
        
        
        # Extract first occurrence of '0' or '1'
        match = re.search(r'\b([01])\b', has_answer)
        
        if match:
            decision = match.group(1)
            
            if decision == "1":
                
                print("[✓] Context is sufficient. Generating response from local LLM...")
                final_system_prompt = self.final_prompt()
                final_input = final_system_prompt.format(context=context, question=question) 
                print("Final Context before LLM Response : \n",context)
                answer = self.query_llm(final_input, max_output_tokens=self.max_output_tokens_final)
                
                return answer
            
            elif decision == "0":
                
                print("[✗] Context is insufficient. Searching online...")
                st.warning(" Web Search Activated — Answer generated using online content.", icon="🌐")

                # Step 2: Perform online search
                url_list = online_search_metaphor(user_query=question, topk_results=ONFLY_topk_URLs_to_return)

                # Step 3: Fetch and clean content from top URLs
                fetch_url_content = FetchURLContent()
                On_fly_pages_and_chunks = fetch_url_content.run_fetch_cleaned_url_content_pipeline(urls=url_list)

                # Step 4: Generate embeddings
                embedding_model_object = EmbeddingModel(
                    embeddings_df_save_path=on_fly_embeddings_df_save_path,
                    embedding_model_name=embedding_model_name,
                    local_embed_model_dir=local_embed_model_dir,
                    device=self.device,
                )

                (embedding_model, 
                embeddings, 
                on_fly_text_chunks_and_embeddings, 
                on_fly_text_chunks_and_embeddings_df) = embedding_model_object.run_embedding_pipeline(
                                                            pages_and_chunks=On_fly_pages_and_chunks,
                                                            replace=True
                                                        )

                # Step 5: Semantic search
                semantic_search_object = SemanticSearch()
                
                search_context, search_context_items, scores, indices = semantic_search_object.run_pipeline(
                    user_query=question,
                    embeddings=embeddings,
                    embedding_model=embedding_model,
                    pages_n_chunks_embeddings_df=on_fly_text_chunks_and_embeddings_df,
                    device=self.device,
                    print_relevant_resources=False,
                    topk_semantic_results=ONFLY_topk_resources_to_return
                )

                print("Scores:", scores)
                print("Indices:", indices)
                print("\n\nSearch Context:\n", search_context, "\n\n")

                # Step 6: Final answer generation using retrieved context
                final_system_prompt = self.final_prompt()
                final_input = final_system_prompt.format(context=search_context, question=question)
                answer = self.query_llm(final_input, max_output_tokens=self.max_output_tokens_final)
                
                return answer

        else:
            print("[⚠️] Unexpected LLM output. Defaulting to online search.")
            st.warning(" Unexpected LLM response. Defaulting to online search.", icon="⚠️")

            # Fallback to online search (same logic as decision == "0")
            url_list = online_search_metaphor(user_query=question, topk_results=ONFLY_topk_URLs_to_return)

            fetch_url_content = FetchURLContent()
            On_fly_pages_and_chunks = fetch_url_content.run_fetch_cleaned_url_content_pipeline(urls=url_list)

            embedding_model_object = EmbeddingModel(
                embeddings_df_save_path=on_fly_embeddings_df_save_path,
                embedding_model_name=embedding_model_name,
                local_embed_model_dir=local_embed_model_dir,
                device=self.device,
            )

            (embedding_model, 
            embeddings, 
            on_fly_text_chunks_and_embeddings, 
            on_fly_text_chunks_and_embeddings_df) = embedding_model_object.run_embedding_pipeline(
                                                        pages_and_chunks=On_fly_pages_and_chunks,
                                                        replace=True
                                                    )

            semantic_search_object = SemanticSearch()
            search_context, search_context_items, scores, indices = semantic_search_object.run_pipeline(
                user_query=question,
                embeddings=embeddings,
                embedding_model=embedding_model,
                pages_n_chunks_embeddings_df=on_fly_text_chunks_and_embeddings_df,
                device=self.device,
                print_relevant_resources=False,
                topk_semantic_results=ONFLY_topk_resources_to_return
            )

            print("Scores:", scores)
            print("Indices:", indices)
            print("\n\nSearch Context:\n", search_context, "\n\n")

            final_system_prompt = self.final_prompt()
            final_input = final_system_prompt.format(context=search_context, question=question)
            answer = self.query_llm(final_input, max_output_tokens=self.max_output_tokens_final)
            
            return answer
       
       
     
        