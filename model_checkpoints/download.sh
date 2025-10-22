

## Qwen-vl-2.5-7b [for automatically generating text prompts, ignore if you wish to provide prompts by yourself always]


mkdir ./model_checkpoints/Qwen2.5-VL-7B-Instruct
cd ./model_checkpoints/Qwen2.5-VL-7B-Instruct
wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/raw/main/.gitattributes
wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/raw/main/README.md
wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/raw/main/chat_template.json
wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/raw/main/config.json
wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/raw/main/generation_config.json
wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/raw/main/merges.txt
wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/model-00001-of-00005.safetensors
wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/model-00002-of-00005.safetensors
wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/model-00003-of-00005.safetensors
wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/model-00004-of-00005.safetensors
wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/model-00005-of-00005.safetensors
wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/raw/main/model.safetensors.index.json
wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/raw/main/preprocessor_config.json
wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/tokenizer.json
wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/raw/main/tokenizer_config.json
wget https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/raw/main/vocab.json



## [Optional] Qwen3-4B-Instruct-2507 [for providing text labels for extracting refined object masks]
cd ..
cd ..
mkdir ./model_checkpoints/Qwen3-4B-Instruct-2507
cd ./model_checkpoints/Qwen3-4B-Instruct-2507
wget https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/raw/main/.gitattributes
wget https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/raw/main/LICENSE
wget https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/raw/main/README.md
wget https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/raw/main/config.json
wget https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/raw/main/generation_config.json
wget https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/raw/main/merges.txt
wget https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/main/model-00001-of-00003.safetensors
wget https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/main/model-00002-of-00003.safetensors
wget https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/main/model-00003-of-00003.safetensors
wget https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/raw/main/model.safetensors.index.json
wget https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/resolve/main/tokenizer.json
wget https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/raw/main/tokenizer_config.json
wget https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/raw/main/vocab.json

## [Optional] SAM-2 and GroundingDINO [for extracting refined object masks]
cd ..
cd ..
cd ./model_checkpoints
wget https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt

cd ..
mkdir ./model_checkpoints/grounding-dino-base
cd ./model_checkpoints/grounding-dino-base
wget https://huggingface.co/IDEA-Research/grounding-dino-base/raw/main/.gitattributes
wget https://huggingface.co/IDEA-Research/grounding-dino-base/raw/main/README.md
wget https://huggingface.co/IDEA-Research/grounding-dino-base/raw/main/config.json
wget https://huggingface.co/IDEA-Research/grounding-dino-base/resolve/main/model.safetensors
wget https://huggingface.co/IDEA-Research/grounding-dino-base/raw/main/preprocessor_config.json
wget https://huggingface.co/IDEA-Research/grounding-dino-base/resolve/main/pytorch_model.bin
wget https://huggingface.co/IDEA-Research/grounding-dino-base/raw/main/special_tokens_map.json
wget https://huggingface.co/IDEA-Research/grounding-dino-base/resolve/main/tokenizer.json
wget https://huggingface.co/IDEA-Research/grounding-dino-base/raw/main/tokenizer_config.json
wget https://huggingface.co/IDEA-Research/grounding-dino-base/raw/main/vocab.txt


## [Optional] InternVL3 [for inferring point hints as indicators of locations to edit, ignore if you wish to provide editing locations by yourself always]

cd ..
cd ..
mkdir ./model_checkpoints/InternVL3-14B
cd ./model_checkpoints/InternVL3-14B
# Base URL for the Hugging Face raw files
BASE_URL="https://huggingface.co/OpenGVLab/InternVL3-14B/resolve/main"

# List of files to download (as seen from the repo)
FILES=(
  "README.md"
  "added_tokens.json"
  "config.json"
  "configuration_intern_vit.py"
  "configuration_internvl_chat.py"
  "conversation.py"
  "generation_config.json"
  "merges.txt"
  "model-00001-of-00007.safetensors"
  "model-00002-of-00007.safetensors"
  "model-00003-of-00007.safetensors"
  "model-00004-of-00007.safetensors"
  "model-00005-of-00007.safetensors"
  "model-00006-of-00007.safetensors"
  "model-00007-of-00007.safetensors"
  "model.safetensors.index.json"
  "modeling_intern_vit.py"
  "modeling_internvl_chat.py"
  "preprocessor_config.json"
  "special_tokens_map.json"
  "tokenizer.json"
  "tokenizer_config.json"
  "vocab.json"
)

# Loop to download each file
for file in "${FILES[@]}"; do
  echo "Downloading: $file"
  wget -c "${BASE_URL}/${file}"
done
