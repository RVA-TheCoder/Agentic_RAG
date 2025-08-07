# Get a Local LLM for Generation task
import os
from dotenv import load_dotenv , find_dotenv

from transformers import AutoTokenizer, AutoModelForCausalLM , BitsAndBytesConfig
from transformers.utils import is_accelerate_available, is_flash_attn_2_available 
from huggingface_hub import whoami, login






# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Authenticate with Hugging Face
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if HUGGINGFACEHUB_API_TOKEN:
    user = whoami(token=HUGGINGFACEHUB_API_TOKEN)
    login(token=HUGGINGFACEHUB_API_TOKEN)
    print(f"[INFO] Logged in as: {user['name']} ({user['email']})")
else:
    raise EnvironmentError("HUGGINGFACEHUB_API_TOKEN not found in environment.")




# Default quantization configuration
quantized_config = BitsAndBytesConfig(
                                        load_in_8bit=True,
                                        llm_int8_threshold=6.0,
                                        llm_int8_enable_fp32_cpu_offload=False
                                    )


# Model configuration map
MODEL_CONFIGS = {
    "gemma2b": {
        "model_id": "google/gemma-2b-it",
        "local_dir": "./HF_LLM_Models/gemma",
        "quantized": True
    },
    
    "phi3": {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "local_dir": "./HF_LLM_Models/phi3mini",
        "quantized": True
    },
    
   
}



class Tokenizer_and_LLM:
    
    """
    Loads a tokenizer and a causal language model (optionally quantized),
    either from local directory or by downloading from Hugging Face Hub.
    """
    
    # def __init__(self, 
    #              device: str, 
    #              model_id: str, 
    #              local_dir: str, 
    #              use_quantization_config: bool = True, 
    #              quantization_config=quantized_config
    #              ):
    
        # self.device = device
        # self.model_id = model_id
        # self.local_dir = local_dir
        # self.use_quantization_config = use_quantization_config
        # self.quantization_config = quantization_config
    
    
    def __init__(self, device: str, model_name: str):
        
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model_name '{model_name}'. Choose from: {list(MODEL_CONFIGS.keys())}")
        
        config = MODEL_CONFIGS[model_name]
        
        self.device = device
        self.model_id = config["model_id"]
        self.local_dir = config["local_dir"]
        self.use_quantization_config = config["quantized"]
        self.quantization_config = quantized_config if self.use_quantization_config else None 
                   
        """
        Parameters : 
            (a) device (str): "cuda" or "cpu"
            (b) model_id (str): HF model ID
            (c) local_dir (str): Local model save/load path
            (d) use_quantization_config (bool): Whether to apply quantization at load
            (e) quantization_config (BitsAndBytesConfig): Optional quant config
        
        """
        
    
    
    def tokenizer_n_LLM(self, ):
           
        """
        Load tokenizer and model from local dir (with optional quantization),
        and save only the full-precision model to disk for future use.

        Returns:
            tuple: (tokenizer, model)
            
        """
        
        attn_implementation = (
            "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
        )
        
        print(f"[INFO] Using attention implementation: {attn_implementation}")

        tokenizer_path = self.local_dir if os.path.exists(self.local_dir) else self.model_id
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {e}")
        
        
        try : 
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
                                                            
                                                            )
                
                os.makedirs(self.local_dir, exist_ok=True)
                tokenizer.save_pretrained(self.local_dir)
                llm_model.save_pretrained(self.local_dir)
                
                print(f"[INFO] [✓] Model and tokenizer saved locally to: {self.local_dir} (in full precision)")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        
            
        # Do not pass both device_map="auto" and manually .to(...) — pick one.
        # For quantized models, use device_map="auto" only.
        # For non-quantized small models: llm_model.to(self.device) is fine.

        return tokenizer, llm_model



