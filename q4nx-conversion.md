# Example - Qwen3.5-9B

## 1. Clone the tensors repo
```bash
git clone https://huggingface.co/empero-ai/Qwen3.8-9B-Distill --depth 1
```
or with huggingface-cli
```bash
hf download empero-ai/Qwen3.8-9B-Distill
```

## 1.5. Optionally add or edit tokenizer_config.json and chat_template.jinja
```bash
cd Qwen3.8-9B-Distill/ 
rm chat_template.jinja
wget https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/main/chat_template.jinja
```
***If you replace your template, remove the template field of the tokenizer_config.json***

## 2. Convert to GGUF

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
uv venv venv --python 3.14
source venv/bin/activate
uv pip install -r requirements.txt
```

```
./convert_hf_to_gguf.py --no-nextn --no-mtp --fp8-as-q8 --no-lazy --model-name Qwen3.8-Distilled-9B-NPU2.gguf --outfile Qwen3.8-Distilled-9B-NPU2/ --outtype q8_0 ./Qwen3.8-9B-Distill/
```

## 3. Convert to Q4NX/Q4_1 and create aux files with q4nx-build (it's even easier if you drop a pre-compiled vision_weight.q4nx in the hf repo)
```bash
q4nx-build -f qwen3.5 -o Qwen3.8-Distilled-9B-NPU2 -s Qwen3.8-9B-Distill/ -i Qwen3.8-Distilled-9B-NPU2/Qwen3.8-9B-Distill-Q8_0.gguf -d qwen3.8-distilled:9b --deploy-from qwen3.5:9b
```

## 4. Test the new model with flm-test
```bash
flm-test --model qwen3.8-distilled:9b --tools --vision --llm --temp 0.2 --reasoning high
```

**Following this exactly, you should get 100% passing on the tests**
