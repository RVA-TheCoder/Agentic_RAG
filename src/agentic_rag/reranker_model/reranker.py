import os
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig

from agentic_rag.constants.constants import *




class BGEReranker:
    
    """
    A class for reranking passages based on relevance to a given query using a pretrained transformer model.

    Attributes:
        model_name (str): Name or path of the model from Hugging Face hub.
        local_dir (str): Local directory to store/load the model.
        quantized (bool): Whether to use 4-bit quantized model loading.
        device (str): Device to use ('cpu' or 'cuda').
        
    """
    
    def __init__(self,
                 model_name: str = reranker_model_name,
                 local_dir: str = reranker_model_local_dir,
                 quantized: bool = True,
                 device: Optional[str]  = None
                 ):
        
        
        """
        Initializes the reranker by loading a transformer model for sequence classification.

        Parameters : 
          (a) model_name (str): Hugging Face model ID.
          (b) local_dir (str): Directory to store/load the model locally.
          (c) quantized (bool): If True, loads the model in 4-bit quantized format using BitsAndBytes.
          (d) device (str, optional): Computation device ('cpu' or 'cuda'). Automatically selected if None.
        
        """
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.local_dir = local_dir

        try : 
            # If local model exists, load from disk with quantization
            if os.path.isdir(local_dir) and os.path.exists(os.path.join(local_dir, "config.json")):
                
                print(f"[INFO] Loading model from local path: {local_dir}")

                self.tokenizer = AutoTokenizer.from_pretrained(local_dir)

                if quantized:
                    print(f"[INFO] Loading quantized model in 4-bit mode...")
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16
                    )
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        local_dir,
                        quantization_config=bnb_config,
                        device_map="auto"
                    )
                    
                else:
                    self.model = AutoModelForSequenceClassification.from_pretrained(local_dir).to(self.device)

            else:
                
                print(f"[INFO] Downloading model from Hugging Face: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

                # Save for future use
                os.makedirs(local_dir, exist_ok=True)
                self.tokenizer.save_pretrained(local_dir)
                self.model.save_pretrained(local_dir)
                print(f"[✓] Model saved locally at {local_dir}")

        except Exception as e:
            print(f"[ERROR] Failed to initialize BGE reranker: {e}")
            raise e
            



    def rerank(self, query: str, passages: List[str], top_k: int = topk_reranker_results) -> List[Tuple[str, float]]:
        
        """
        Reranks the given list of passages based on their semantic relevance to the query.

        parameters : 
            (a) query (str): User query string.
            (b) passages (List[str]): List of candidate passages.
            (c) top_k (int): Number of top relevant passages to return.

        Returns:
            List[Tuple[str, float]]: Top-k passages with scores, sorted by descending relevance.

        Raises:
            ValueError: If the passages list is empty.
            
        """
        
        if not passages:
            raise ValueError("Passages list is empty. Cannot rerank.")
        
        try : 
            pairs = [(query, passage) for passage in passages]

            inputs = self.tokenizer.batch_encode_plus(
                                            pairs,
                                            padding=True,
                                            truncation=True,
                                            max_length=512,
                                            return_tensors="pt"
                                        )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

                # Handle single-logit models
                scores = outputs.logits.view(-1)

            reranked = sorted(zip(passages, scores.tolist()), key=lambda x: x[1], reverse=True)
            
            # returns Top-k passages with scores, sorted by descending relevance.
            return reranked[:top_k]
    
    
        except Exception as e: 
            
            print(f"[ERROR] Failed during reranking: {e}")
            raise e
    



 