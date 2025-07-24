import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig


class BGEReranker:
    
    def __init__(self,
                 model_name: str = "BAAI/bge-reranker-base",
                 local_dir: str = "./HF_Reranker_Model",
                 quantized: bool = True,
                 device: str = None):
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.local_dir = local_dir

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
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)

            # Save for future use
            os.makedirs(local_dir, exist_ok=True)
            self.tokenizer.save_pretrained(local_dir)
            self.model.save_pretrained(local_dir)
            print(f"[✓] Model saved locally at {local_dir}")

    def rerank(self, query: str, passages: list[str], top_k: int = 5) -> list[tuple[str, float]]:
        """
        Reranks passages based on relevance to the query.

        Returns:
            List of tuples: [(passage, score), ...] sorted by score descending
        """
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
        return reranked[:top_k]