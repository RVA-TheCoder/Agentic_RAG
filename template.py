import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] : %(message)s :')


project_name="agentic_rag"

list_of_files=[
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils_methods/__init__.py",
    f"src/{project_name}/ask_from_url/__init__.py",
    f"src/{project_name}/embedding_model/__init__.py",
    f"src/{project_name}/generate_text/__init__.py",
    f"src/{project_name}/get_url_content/__init__.py",
    f"src/{project_name}/rag_input_preprocessor/__init__.py",
    f"src/{project_name}/reranker_model/__init__.py",
    f"src/{project_name}/semantic_search/__init__.py",
    f"src/{project_name}/tokenizer_and_llm_model/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    
    "requirements.txt",
    "setup.py",
    "research/experiments.ipynb",
    

]


for filepath in list_of_files:

    filepath=Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":

        os.makedirs(filedir, exist_ok=True)

        logging.info(f"Creating directory : {filedir} for the file:{filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):

        with open(filepath, "w") as f:
            
            logging.info(f"Creating empty file : {filepath}")
            pass

    else:
        logging.info(f"{filename} is already exists.")

    










