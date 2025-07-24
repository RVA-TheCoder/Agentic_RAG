# Get a Local LLM for Generation task
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import is_accelerate_available
from transformers.utils import is_flash_attn_2_available 
from transformers import BitsAndBytesConfig

import os

from huggingface_hub import whoami
from huggingface_hub import login

from dotenv import load_dotenv , find_dotenv

load_dotenv(find_dotenv(), override=True)

HUGGINGFACEHUB_API_TOKEN=os.getenv("HUGGINGFACEHUB_API_TOKEN")
user = whoami(token=HUGGINGFACEHUB_API_TOKEN)
login(token=HUGGINGFACEHUB_API_TOKEN)
print(user)


quantized_config = BitsAndBytesConfig(
                                        load_in_8bit=True,
                                        llm_int8_threshold=6.0,
                                        llm_int8_enable_fp32_cpu_offload=False
                                    )



class Tokenizer_and_LLM:
    
    
    def __init__(self, 
                 device, 
                 model_id, 
                 local_dir, 
                 use_quantization_config:bool = True, 
                 quantization_config=quantized_config
                 ):
        
        self.device = device
        self.model_id = model_id
        self.local_dir = local_dir
        self.use_quantization_config = use_quantization_config
        self.quantization_config = quantization_config
    
    
    def tokenizer_n_LLM(self, 
                        #device,
                        #model_id: str = "google/gemma-2-2b-it",
                        #local_dir: str = "llm_models/gemma",
                        #load_in_8bit=False,
                        #quantization_config=None,
                        #use_quantization_config=True,
                        #save_full_precision_only=True
                    ):
        """
        Load tokenizer and model from local dir (with optional quantization),
        and save only the full-precision model to disk for future use.

        Args:
            quantization_config (BitsAndBytesConfig): Optional quant config
            model_id (str): HF model ID
            device (str): "cuda" or "cpu"
            local_dir (str): Local model save/load path
            load_in_8bit (bool): Use 8-bit quant (not used during save)
            use_quantization_config (bool): Whether to apply quantization at load
            save_full_precision_only (bool): When downloading, always save in full precision

        Returns:
            tokenizer, model
        """
        
        if is_flash_attn_2_available():
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "sdpa"
        print(f"[INFO] Using attention implementation: {attn_implementation}")

        # Load tokenizer (no change)
        tokenizer = AutoTokenizer.from_pretrained(self.local_dir if os.path.exists(self.local_dir) else self.model_id)

        if os.path.exists(self.local_dir):
            
            print(f"[INFO] Loading model from local dir with quantization={self.use_quantization_config}")
            
            llm_model = AutoModelForCausalLM.from_pretrained(
                                                            pretrained_model_name_or_path=self.local_dir,
                                                            device_map="auto",
                                                            
                                                            quantization_config=self.quantization_config if self.use_quantization_config else None,
                                                            low_cpu_mem_usage=True,
                                                            attn_implementation=attn_implementation
                                                        )
        
        else:
            
            print(f"[INFO] Downloading full-precision model from Hugging Face: {self.model_id}")
            
            llm_model = AutoModelForCausalLM.from_pretrained(
                                                            self.model_id,
                                                            device_map="auto",
                                                    
                                                            low_cpu_mem_usage=True,
                                                            attn_implementation=attn_implementation
                                                            # No quantization during download
                                                        )
            
            os.makedirs(self.local_dir, exist_ok=True)
            tokenizer.save_pretrained(self.local_dir)
            llm_model.save_pretrained(self.local_dir)
            
            print(f"[INFO] Model saved to: {self.local_dir} (in full precision)")
            
            
        # Do not pass both device_map="auto" and manually .to(...) — pick one.
        # For quantized models, use device_map="auto" only.
        # For non-quantized small models: llm_model.to(self.device) is fine.

        return tokenizer, llm_model











