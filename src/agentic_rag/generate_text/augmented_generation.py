from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from agentic_rag.utils_methods.basic_methods import online_search_duckduckgo, online_search_metaphor
from agentic_rag.get_url_content.fetch_cleaned_url_content import FetchURLContent

from agentic_rag.embedding_model.embedding_model import EmbeddingModel
from agentic_rag.constants.constants import embeddings_df_save_path, on_fly_embeddings_df_save_path, topk_resources_to_return, ONFLY_topk_resources_to_return
from agentic_rag.utils_methods.basic_methods import device_availabilty

from agentic_rag.semantic_search.semantic_search import SemanticSearch

import streamlit as st




class AugmentedGeneration:
    
    def __init__(self, tokenizer, llm_model, embedding_model, context, max_output_tokens, device ):
        
        
        self.device = device
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.tokenizer = tokenizer
        self.context = context
        self.max_output_tokens = max_output_tokens
    
    
    def decision_prompt(self):
        
        """
        decision_prompt :  will check to see if the retrieved context can answer the user question.
        """
        
        # decision_system_prompt = """
        
        # Context : ```{context}```
        # Question : {question}
        
        # Your job is to decide if a given Question can be answered with a given Context.
        # If the Context can answer the Question, return 1.
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
        You are a helpful AI assistant.

        Your job is to decide if a given **Question** can be answered using the given **Context**.

        Return:
        - 1 → if the context has enough information to answer the question.
        - 0 → if it does not.

        ⚠️ Output only a single digit: 1 or 0. No explanation.

        ---

        Context:
        {context}

        Question:
        {question}

        Examples:

        Example 1:
        Context: The sky is blue.
        Question: What color is the sky?
        Answer: 1

        Example 2:
        Context: Winston Churchill rallied the British people during World War II after the Dunkirk evacuation, emphasizing that though the battle ahead would be tough, Britain would never surrender. He declared that the nation would fight on the beaches, landing grounds, fields, streets, and hills, defending itself at all costs. The speech was a powerful call for resilience and unity in the face of Nazi aggression.
        Question: What did Churchill say about beaches?
        Answer: 1

        Example 3:
        Context: Artificial Intelligence is transforming industries by enabling machines to perform tasks that once required human intelligence—like learning, reasoning, and decision-making. From healthcare and education to transportation and finance, AI promises efficiency, innovation, and new opportunities. However, it also brings challenges such as ethical concerns, job displacement, and the need for responsible development.
        Question: Who is Bill Gates?
        Answer: 0

        ---

        Now decide for the current input:

        Answer:
        """
        
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
        
        return final_system_prompt
    
    
    
    def query_llm(self, prompt, max_output_tokens=64):
        
        
        generate_text = pipeline("text-generation", model=self.llm_model, tokenizer=self.tokenizer)
        response = generate_text(prompt, max_new_tokens=max_output_tokens, do_sample=False)
    
        return response[0]["generated_text"][len(prompt):].strip()
    
    
    ## Start your work form here
    
    def get_final_answer(self, question, context): # self.context

        # Step 1: Ask the decision system
        decision_system_prompt = self.decision_prompt()
        
        decision_input = decision_system_prompt.format(context=context, question=question)
        has_answer = self.query_llm(decision_input, max_output_tokens=10)

        print(f"\n[User Question]: {question}")
        print("given context has answer or not : ", has_answer,"\n")
        
        if "1" in has_answer:
            print("[✓] Context is sufficient. Generating response from local LLM...")
            
            final_system_prompt = self.final_prompt()
            final_input = final_system_prompt.format(context=context, question=question) 
            answer = self.query_llm(final_input, max_output_tokens=self.max_output_tokens)
            
            return answer

        
        else:
            
            print("[✗] Context is insufficient. Searching online...")
            st.warning(" Web Search Activated — Answer generated using online content.", icon="🌐")
            #results = DDGS().text(question, max_results=5)
            #search_context = "\n".join([f"- {r['title']}: {r['body']}" for r in results])
            
            # Write this class method above
            # need to write our own logic
            #url_list = online_search_duckduckgo(user_query=question, topk_results=2)
            url_list = online_search_metaphor(user_query=question, topk_results=2)
            
            # Create object
            fetch_url_content = FetchURLContent()
            On_fly_pages_and_chunks_list = fetch_url_content.run_fetch_cleaned_url_content_pipeline(urls = url_list)
            
            
            #device = device_availabilty()
            #print("Avialabel device : ",device)

            # Create object of the class
            embedding_model_object = EmbeddingModel(embeddings_df_save_path=on_fly_embeddings_df_save_path,
                                                    embedding_model_name='all-mpnet-base-v2',
                                                    local_model_dir='HF_Embedding_Models',
                                                    device=self.device,
                                                )
            
            (embedding_model, 
            embeddings, 
            on_fly_text_chunks_and_embeddings, 
            on_fly_text_chunks_and_embeddings_df) = embedding_model_object.run_embedding_pipeline(pages_and_chunks=On_fly_pages_and_chunks_list,
                                                                                                  replace=True
                                                                                                 )
            
            
            # Create object 
            semantic_search_object = SemanticSearch()
            
            #user_query = 'who killed Gatsby in The Great Gatsby Novel?'

            (search_context, 
             search_context_items , 
             scores, 
             indices ) = semantic_search_object.run_pipeline(user_query=question, 
                                                            embeddings=embeddings, 
                                                            embedding_model=embedding_model, 
                                                            pages_n_chunks_embeddings_df=on_fly_text_chunks_and_embeddings_df, 
                                                            device=self.device,
                                                            print_relevant_resources=False,
                                                            topk_results = ONFLY_topk_resources_to_return
                                                            )
             
            
            print("Scores : ", scores)
            print("Indices : " ,indices)
            print("\n\n,Search Context : ",search_context,"\n\n")
            # Retry with updated context
            final_system_prompt = self.final_prompt()
            final_input = final_system_prompt.format(context=search_context, question=question)
            
            answer = self.query_llm(final_input)
            
            return answer
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
        