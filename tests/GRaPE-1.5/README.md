---
license: apache-2.0
language:
  - en
  - zh
  - fr
  - de
  - es
  - ja
  - ko
  - pt
  - ru
  - ar
tags:
  - reasoning
  - vision
  - multimodal
  - instruct
  - chat
  - coding
  - math
  - science
  - april-fools
pipeline_tag: image-text-to-text
---

<div align="center">

**GRaPE 1.5**

General Reasoning Agent for Project Exploration

The most capable open-weight model ever released at this parameter count.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Model Size](https://img.shields.io/badge/Parameters-4B-orange.svg)]()
[![Vision](https://img.shields.io/badge/Vision-✓-brightgreen.svg)]()
[![Context](https://img.shields.io/badge/Context-256K-purple.svg)]()
[![Languages](https://img.shields.io/badge/Languages-47-teal.svg)]()

</div>

---

> **GRaPE 1.5** is a 4-billion-parameter multimodal reasoning model that delivers frontier-class intelligence. It surpasses GRaPE 1 across every single benchmark by substantial margins, introduces native vision understanding, and matches or exceeds models hundreds of times its size on standard evaluations. There is no flash, mini, or nano variant — GRaPE 1.5 is a single, unified model built to be the only model you need.

---

## 🔑 Key Highlights

| | GRaPE 1 | **GRaPE 1.5** |
|---|:---:|:---:|
| Parameters | 4B | **4B** |
| Context Window | 32K | **256K** |
| Vision | ✗ | **✓** |
| Languages | 12 | **47** |
| MMLU | 74.2 | **89.2** |
| MATH-500 | 63.1 | **90.8** |
| HumanEval | 72.8 | **93.4** |
| GPQA Diamond | 48.3 | **77.6** |
| Training tokens | 1.2T | **8.4T** |
| Data quality filtering | Basic heuristics | **Multi-stage synthetic verification** |

---

## 🚀 Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import torch

# Text-only
tokenizer = AutoTokenizer.from_pretrained("sweaterdog/GRaPE-1.5")
model = AutoModelForCausalLM.from_pretrained(
    "sweaterdog/GRaPE-1.5",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [{"role": "user", "content": "Prove that √2 is irrational."}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
output = model.generate(inputs, max_new_tokens=2048, temperature=0.6, top_p=0.9)
print(tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True))
```

```python
# Vision (image + text)
from transformers import AutoProcessor
from PIL import Image
import requests

processor = AutoProcessor.from_pretrained("sweaterdog/GRaPE-1.5")
model = AutoModelForCausalLM.from_pretrained(
    "sweaterdog/GRaPE-1.5",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

image = Image.open("chart.png")
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "Describe what this chart shows and identify any trends."}
]}]
inputs = processor(messages, images=[image], return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=1024)
print(processor.decode(output[0], skip_special_tokens=True))
```

---

## 📊 Benchmark Results

> All evaluations are conducted under standardized 0-shot or few-shot conditions as specified by each benchmark's official protocol.  
> GRaPE 1.5 scores are **bolded**. Frontier model scores reflect publicly reported results as of Q1 2026.

### Text & Reasoning

![benchmark_text](https://cdn-uploads.huggingface.co/production/uploads/66960602f0ffd8e3a381106a/nQ7F4WUCFameJZ7ZHjxG4.png)

<details>
<summary><b>📋 Full Text Benchmark Table</b></summary>

| Benchmark | GPT-5.4 | Claude Opus 4.6 | Gemini 3.1 Pro | Qwen3.5 397B | **GRaPE 1.5 (4B)** |
|:---|:---:|:---:|:---:|:---:|:---:|
| MMLU (5-shot) | 92.3 | 91.8 | 90.4 | 88.7 | **89.2** |
| MMLU-Pro | 87.4 | 86.9 | 85.3 | 82.1 | **83.7** |
| GPQA Diamond (0-shot) | 82.1 | 81.4 | 79.8 | 75.3 | **77.6** |
| ARC-Challenge (0-shot) | 97.8 | 97.2 | 96.9 | 95.4 | **95.8** |
| HellaSwag (10-shot) | 97.2 | 96.8 | 96.5 | 95.1 | **94.9** |
| WinoGrande (5-shot) | 94.2 | 93.8 | 93.1 | 91.4 | **92.3** |
| TruthfulQA (0-shot) | 88.3 | 87.1 | 86.4 | 82.7 | **85.9** |
| BBH (3-shot) | 91.4 | 90.7 | 89.3 | 85.6 | **87.2** |
| AGIEval | 84.7 | 83.9 | 82.4 | 78.3 | **80.6** |
| DROP (3-shot, F1) | 90.3 | 89.7 | 88.4 | 84.2 | **87.1** |

</details>

---

### Mathematics

<details>
<summary><b>📋 Mathematics Benchmark Table</b></summary>

| Benchmark | GPT-5.4 | Claude Opus 4.6 | Gemini 3.1 Pro | Qwen3.5 397B | **GRaPE 1.5 (4B)** |
|:---|:---:|:---:|:---:|:---:|:---:|
| MATH-500 (0-shot) | 95.1 | 94.3 | 93.7 | 91.2 | **90.8** |
| GSM8K (8-shot CoT) | 98.4 | 97.9 | 97.2 | 95.8 | **96.3** |
| GSM-Hard | 91.7 | 90.4 | 89.8 | 85.3 | **87.4** |
| AIME 2024 (Pass@1) | 72.3 | 70.8 | 68.4 | 61.7 | **64.2** |
| AIME 2025 (Pass@1) | 68.4 | 66.7 | 65.1 | 57.3 | **60.8** |
| OlympiadBench | 65.3 | 63.8 | 62.1 | 55.4 | **58.7** |
| MathBench | 89.4 | 88.1 | 87.2 | 83.6 | **85.3** |
| Minerva MATH | 84.7 | 83.2 | 81.9 | 77.4 | **79.8** |

</details>

---

### Coding

![benchmark_coding](https://cdn-uploads.huggingface.co/production/uploads/66960602f0ffd8e3a381106a/1geeRkuFuRZhXhzEbbNCj.png)

<details>
<summary><b>📋 Full Coding Benchmark Table</b></summary>

| Benchmark | GPT-5.4 | Claude Opus 4.6 | Gemini 3.1 Pro | Qwen3.5 397B | **GRaPE 1.5 (4B)** |
|:---|:---:|:---:|:---:|:---:|:---:|
| HumanEval (0-shot) | 96.2 | 95.8 | 94.1 | 92.3 | **93.4** |
| HumanEval+ | 94.3 | 93.7 | 91.8 | 89.2 | **91.6** |
| MBPP+ | 90.1 | 89.4 | 88.7 | 85.3 | **87.9** |
| LiveCodeBench | 72.4 | 70.8 | 69.3 | 63.7 | **68.4** |
| SWE-bench Verified | 62.1 | 60.7 | 58.4 | 52.3 | **57.8** |
| BigCodeBench | 77.3 | 76.1 | 74.8 | 70.2 | **73.4** |
| CRUXEval-I | 71.8 | 70.4 | 68.9 | 63.4 | **67.2** |
| CRUXEval-O | 74.2 | 72.9 | 71.3 | 65.8 | **69.7** |

</details>

---

### 👁️ Vision Benchmarks

GRaPE 1.5 is the second model in the GRaPE family to support native image understanding. It was trained on over 2.1 trillion vision tokens spanning photographs, diagrams, charts, documents, scientific figures, and rendered UI screenshots.

![benchmark_vision](https://cdn-uploads.huggingface.co/production/uploads/66960602f0ffd8e3a381106a/c_FIAPibWdzjJ5ZjIObZk.png)

<details>
<summary><b>📋 Full Vision Benchmark Table</b></summary>

| Benchmark | GPT-5.4 | Claude Opus 4.6 | Gemini 3.1 Pro | **GRaPE 1.5 (4B)** |
|:---|:---:|:---:|:---:|:---:|
| MMMU (val, 0-shot) | 82.4 | 81.7 | 84.2 | **78.3** |
| MMMU-Pro | 72.1 | 70.8 | 74.3 | **67.4** |
| ChartQA (0-shot) | 91.3 | 90.4 | 92.1 | **87.6** |
| DocVQA (0-shot) | 95.2 | 94.8 | 95.9 | **91.4** |
| TextVQA (0-shot) | 88.7 | 87.3 | 89.4 | **85.2** |
| MathVista (0-shot) | 79.4 | 78.2 | 80.1 | **74.8** |
| AI2D (0-shot) | 92.3 | 91.7 | 93.4 | **88.9** |
| OCRBench | 84.7 | 83.1 | 86.2 | **80.4** |
| ScienceQA (img) | 95.8 | 95.2 | 96.1 | **92.7** |
| Infographics VQA | 82.4 | 81.3 | 84.7 | **77.8** |
| RealWorldQA | 78.3 | 77.1 | 79.8 | **73.4** |

</details>

---

### 🏆 Multi-dimensional Capability Radar

![radar](https://cdn-uploads.huggingface.co/production/uploads/66960602f0ffd8e3a381106a/jWopRzdD3cbshWv1ocLoJ.png)

---

### ⚡ Parameter Efficiency

GRaPE 1.5 achieves a landmark result: **frontier-class performance at 4 billion parameters**.

![efficiency](https://cdn-uploads.huggingface.co/production/uploads/66960602f0ffd8e3a381106a/NyuuS9xP27U64QVPBtrSD.png)

> GRaPE 1.5 delivers 97–99% of frontier model performance while using **99.7% fewer parameters** than the leading closed-source models. At inference, it runs in under 3GB of VRAM in 4-bit quantization — accessible on a single consumer GPU.

---

## 📈 GRaPE 1 → GRaPE 1.5: Generation-over-Generation

![grape_improvement](https://cdn-uploads.huggingface.co/production/uploads/66960602f0ffd8e3a381106a/fV9GAXP5XNwHWkHopUGfw.png)

<details>
<summary><b>What changed from GRaPE 1?</b></summary>

### Architecture Changes
- Extended context window from **32K → 256K** tokens via a new sliding-window attention mechanism
- Added a **vision encoder** (ViT-L/14 backbone, trained from scratch on 2.1T image tokens)
- Replaced RoPE with **YaRN** rotary embeddings for better long-context scaling
- Upgraded MLP from SwiGLU to **GeGLU** with tied gate projections

### Training Data
GRaPE 1 was trained on approximately 1.2 trillion tokens consisting largely of web-scraped text with basic quality heuristics. GRaPE 1.5 uses a fundamentally different data pipeline:

| | GRaPE 1 | GRaPE 1.5 |
|---|---|---|
| Total tokens | 1.2T | 8.4T |
| Quality filter | Basic dedup + perplexity | Multi-stage synthetic verification |
| Math data | ~2B tokens | ~480B tokens |
| Code data | ~18B tokens | ~620B tokens |
| Reasoning traces | None | 94B synthetic CoT tokens |
| Vision tokens | None | 2.1T multimodal tokens |
| RLHF | Basic RLHF | RLHF + Constitutional AI + DPO |

### Post-training
GRaPE 1.5 underwent an extensive multi-stage post-training pipeline:
1. **Supervised Fine-Tuning (SFT)** on 48M high-quality instruction-following examples
2. **Chain-of-Thought distillation** from long-form reasoning traces
3. **Direct Preference Optimization (DPO)** with 12M preference pairs
4. **Constitutional AI** regularization to improve instruction adherence
5. **Test-time compute scaling** via MCTS-guided self-play on math and code domains

</details>

---

## 🏗️ Model Architecture

<details>
<summary><b>Full architecture details</b></summary>

| Component | Specification |
|---|---|
| Architecture | Transformer (decoder-only) + ViT vision encoder |
| Parameters (total) | 4.5B |
| Parameters (non-embedding) | 4.1B |
| Layers | 36 |
| Attention heads | 32 |
| KV heads | 8 (GQA) |
| Hidden dim | 3072 |
| Intermediate dim | 8192 |
| Vocabulary size | 131,072 |
| Context length (training) | 262,144 |
| Attention | GQA + sliding window (local 4096) |
| Positional embedding | YaRN RoPE (θ=500,000) |
| Activation | GeGLU |
| Normalization | RMSNorm (pre-norm) |
| Vision encoder | ViT-L/14 (336px), 307M params |
| Vision-language projector | 2-layer MLP with cross-attention |
| Precision (release) | BFloat16 |

</details>

---

## 🌍 Multilingual Performance

GRaPE 1.5 was trained with deliberate multilingual coverage across 47 languages. Below are MMLU scores for major language families, evaluated in-language:

<details>
<summary><b>Multilingual MMLU Scores</b></summary>

| Language | GRaPE 1.5 | GPT-5.4 | Claude Opus 4.6 |
|---|:---:|:---:|:---:|
| English | **89.2** | 92.3 | 91.8 |
| Chinese (Simplified) | **87.4** | 89.1 | 88.7 |
| French | **85.7** | 87.3 | 86.9 |
| German | **84.9** | 86.8 | 86.1 |
| Spanish | **86.1** | 87.9 | 87.4 |
| Japanese | **83.2** | 85.4 | 84.8 |
| Korean | **82.8** | 84.7 | 84.1 |
| Arabic | **79.3** | 81.2 | 80.6 |
| Russian | **83.7** | 85.6 | 85.1 |
| Portuguese | **85.3** | 87.1 | 86.6 |

</details>

---

## 💻 Inference & Deployment

### VRAM Requirements

| Precision | VRAM | Notes |
|---|---|---|
| BF16 (full) | ~9 GB | Full inference, best quality |
| FP8 | ~5 GB | Minimal quality loss |
| INT4 (GPTQ/AWQ) | ~2.8 GB | Runs on any RTX 3060+ |
| INT3 | ~2.1 GB | Suitable for edge deployment |

### Hardware Recommendations

| Use Case | Recommended Hardware |
|---|---|
| Development / Testing | RTX 3080 (10GB) or better |
| Production (low-latency) | RTX 4090 / A100 40GB |
| Edge / On-device | Apple M2 Pro 16GB+ |
| Batch inference | 2× A100 80GB (tensor parallel) |

### Speed (tokens/second, BF16, batch=1)

| Hardware | Prefill (tok/s) | Decode (tok/s) |
|---|---|---|
| RTX 4090 | 18,400 | 142 |
| RTX 3090 | 12,700 | 98 |
| A100 80GB | 24,800 | 187 |
| Apple M3 Max | 5,100 | 41 |

---

## 🔧 Usage Examples

<details>
<summary><b>Long-context reasoning (256K tokens)</b></summary>

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("sweaterdog/GRaPE-1.5")
model = AutoModelForCausalLM.from_pretrained(
    "sweaterdog/GRaPE-1.5",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

with open("very_long_document.txt") as f:
    document = f.read()

messages = [{"role": "user", "content": f"{document}\n\nSummarize the three most important findings."}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
output = model.generate(inputs, max_new_tokens=1024)
print(tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True))
```

</details>

<details>
<summary><b>Structured JSON output</b></summary>

```python
messages = [{
    "role": "user",
    "content": (
        "Extract all named entities from the following text as a JSON object "
        "with keys 'people', 'organizations', 'locations':\n\n"
        "Apple CEO Tim Cook announced at WWDC in San Francisco that the company "
        "is partnering with OpenAI to bring Siri improvements to iPhone 17."
    )
}]
# GRaPE 1.5 reliably produces valid JSON without additional scaffolding
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
output = model.generate(inputs, max_new_tokens=512, temperature=0.1)
print(tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True))
```

</details>

<details>
<summary><b>vLLM (high-throughput serving)</b></summary>

```bash
pip install vllm
vllm serve sweaterdog/GRaPE-1.5 \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --tensor-parallel-size 1
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="token")
response = client.chat.completions.create(
    model="sweaterdog/GRaPE-1.5",
    messages=[{"role":"user","content":"Write a Rust HTTP server."}],
)
print(response.choices[0].message.content)
```

</details>

<details>
<summary><b>Ollama (local, no-code)</b></summary>

```bash
ollama run sweaterdog/grape-1.5
```

</details>

---

## 📦 Quantized Variants

The following quantized variants are coming shortly, we will update the repository with links once they are live.

---

## 📜 Citation

If you use GRaPE 1.5 in your research or products, please cite:

```bibtex
@misc{grape2026,
  title        = {GRaPE 1.5: General Reasoning Agent for Project Exploration},
  author       = {SweaterDog},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/sweaterdog/GRaPE-1.5}},
  note         = {4B multimodal reasoning model}
}
```

---

## ⚠️ Limitations

- Like all language models, GRaPE 1.5 can hallucinate facts, particularly for very recent events (knowledge cutoff: February 2026).
- Vision understanding degrades on very low-resolution images (below 112×112 pixels).
- While multilingual, performance in lower-resource languages lags behind high-resource ones by 5–10 MMLU points on average.
- Long-context performance beyond 128K tokens, while supported, is not as well-calibrated as within 32K tokens.
- GRaPE 1.5 is an instruction-tuned model and should not be used for harmful, deceptive, or illegal purposes.

---

<div align="center">
<sub>GRaPE 1.5 — General Reasoning Agent for Project Exploration</sub><br>
<sub>Released under the Apache 2.0 License</sub>
</div>
