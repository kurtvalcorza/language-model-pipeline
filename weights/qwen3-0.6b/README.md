---
library_name: transformers
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/LICENSE
pipeline_tag: text-generation
base_model:
- Qwen/Qwen3-0.6B-Base
---

# Qwen3-0.6B Model Card

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Qwen%2FQwen3--0.6B-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/Qwen/Qwen3-0.6B)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

###### Description

**Qwen3-0.6B** is an ultra-compact dense causal language model developed by the Qwen Team (Alibaba Cloud) with approximately 0.6 billion total parameters (0.44B non-embedding parameters). It represents the lightweight entry point of the Qwen3 series, engineered for high-efficiency instruction following, multi-turn conversational assistance, structured output generation, and dual-mode reasoning.

The model uniquely supports seamless runtime switching between **thinking mode** (generating step-by-step reasoning tokens enclosed in `<think>...</think>` tags for complex problem solving) and **non-thinking mode** (direct token generation for high-throughput, low-latency dialogue) within a single unified checkpoint.

In this repository (`language-model-pipeline`), `Qwen3-0.6B` serves as the primary open-weights baseline and smoke-test foundation model. With its compact weights payload (~1.41 GiB safetensors), it enables rapid end-to-end verification of Supervised Fine-Tuning (SFT), 4-bit QLoRA adaptation, conversational chat templating, assistant-only loss masking, and PEFT adapter export on consumer GPUs and Google Colab free-tier runtimes in under one minute.

#### Intended Use and Limitations

###### Primary Intended Uses

- **Supervised Fine-Tuning (SFT) and PEFT / QLoRA Adaptation:** Serving as a lightweight base model for domain-specific fine-tuning, instruction alignment, and style transfer using conversational or prompt-completion datasets.
- **Pipeline Verification and CI/CD Smoke Testing:** Rapid local and automated execution testing of data loading, chat templating, tokenization, gradient backpropagation, adapter serialization, and clean reload without requiring multi-gigabyte downloads.
- **Interactive Conversational Assistance:** General-purpose dialogue, text summarization, paraphrasing, structured extraction (JSON/YAML), and question answering.
- **Edge and Low-Latency Serving:** On-device deployment, local desktop applications (Ollama, llama.cpp, MLX-LM), and resource-constrained environments where minimal memory footprint (<1.5 GB VRAM) and fast time-to-first-token are required.
- **Tool Calling and Agentic Workflows:** Multi-step tool invocation and structured schema generation in both thinking and non-thinking modes via frameworks such as Qwen-Agent and Model Context Protocol (MCP) servers.

###### Primary Intended Users

- **Machine Learning Engineers and Researchers:** Developers building, benchmarking, and optimizing language model fine-tuning pipelines, quantization workflows, and adapter-serving architectures.
- **Students and Educators:** Learners exploring causal language modeling, QLoRA adaptation, and chat templates on standard single-GPU environments (e.g., Google Colab T4).
- **Application Developers:** Engineers deploying lightweight edge assistants, localized text transformers, or embedded conversational agents.

###### Out-of-scope use cases

- **Autonomous High-Stakes Decision-Making:** Critical medical diagnoses, legal judgment, criminal justice decisions, credit underwriting, or life-safety control systems without human expert oversight.
- **Unverified Parametric Fact Retrieval:** Relying on the model as an authoritative encyclopedia without external retrieval-augmented generation (RAG); 0.6B models possess limited parametric memory compared to large frontier models and have higher factual hallucination rates.
- **Contexts Exceeding Architectural Bounds:** Input token sequences exceeding the model's supported context window without proper chunking or windowing strategies.
- **Malicious Generation:** Generating phishing campaigns, malware, automated disinformation, spam, hate speech, or content promoting violence and illegal acts.

---

#### Factors

###### Groups

- **Multilingual and Dialectal Groups:** Pretrained on a massive multilingual corpus covering 100+ languages and dialects, with highest fidelity in English, Simplified and Traditional Chinese, and major European and Asian languages. Coverage in low-resource regional dialects (e.g., regional Philippine languages, indigenous vernaculars) is lower, requiring targeted fine-tuning.
- **Sociodemographic Representation:** The pretraining corpus reflects diverse cultural, occupational, and age-group web sources. However, because internet text contains demographic disparities, downstream applications must evaluate potential biases across protected characteristics (gender, ethnicity, socio-economic backgrounds).

###### Instrumentation

- **Data Ingestion and Filtering Instruments:** High-throughput web crawlers, text deduplication engines (MinHash, exact match), language identification classifiers (FastText), quality scoring models, and synthetic data generation clusters used during upstream pretraining and post-training alignment.
- **Training Infrastructure:** Upstream pretraining on high-performance distributed GPU clusters (NVIDIA A100/H100) using RoCE/InfiniBand fabrics, Megatron-LM / DeepSpeed parallel training engines, and mixed-precision BF16 arithmetic.
- **Downstream Execution and Fine-Tuning Toolchain:** PyTorch >=2.0, Hugging Face `transformers` >=4.51.0, `tokenizers` >=0.20.0, `peft` >=0.14.0, `bitsandbytes` (4-bit NormalFloat quantization), and standard CUDA runtimes.

###### Environment

- **Compute & Hardware Requirements:**
  - Full Precision / BF16 Inference: ~1.41 GiB VRAM.
  - 4-bit Quantized (QLoRA / bitsandbytes): ~0.8–1.2 GiB VRAM.
  - Fine-Tuning (QLoRA bs=1, seq=512–2048): ~4.2–5.5 GiB VRAM peak allocated memory.
  - Compatible with entry-level consumer GPUs (e.g., RTX 3050/4060, Apple Silicon M-series, Google Colab T4 15 GB).
- **Software Dependencies:** Operating Systems: Linux, Windows, macOS; Python 3.10–3.12; CUDA 11.8+ or ROCm/Metal equivalents.
- **Operational Conditions:** Can be deployed locally without internet connectivity once weights and tokenizer files are acquired.

---

#### Metrics

###### Performance Measures

- **Academic & Benchmark Measures (Upstream):**
  - MMLU / MMLU-Pro: Measuring multi-subject knowledge acquisition and reasoning.
  - GSM8K & MATH: Evaluating mathematical problem-solving ability in thinking mode.
  - HumanEval & MBPP: Measuring Python and algorithmic code generation pass rates.
  - MT-Bench / Arena-Hard: Measuring conversational instruction following and multi-turn preference quality.
- **Fine-Tuning & Pipeline Optimization Metrics:**
  - Token-level Cross-Entropy Loss: Measuring optimization convergence on supervised assistant response tokens.
  - Validation Perplexity ($e^{\text{loss}}$): Quantifying model uncertainty on held-out conversation splits.
  - Peak GPU Memory (GiB): Measuring maximum VRAM allocated during forward, backward, and optimizer updates.
  - Token Throughput (tok/sec): Measuring processing speed across sequence lengths.

###### Decision thresholds

- **Cryptographic File Verification Threshold:** 100% exact match on SHA-256 hashes across all model weights and tokenizer configurations recorded in `dimer-base-manifest.json`. Zero tolerated byte alterations.
- **Optimization Loss Threshold:** Training validation cross-entropy loss must exhibit smooth convergence; divergence or NaN loss triggers immediate run termination.
- **Quantization Stability Threshold:** Under 4-bit NF4 double-quantization, baseline generation quality must remain prefix-stable and coherent with negligible perplexity drift relative to BF16.
- **Application Deployment Gate:** Target application performance thresholds (e.g., >=90% intent classification accuracy or human review pass rate) must be determined on application-specific evaluation datasets prior to production serving.

###### Approaches to uncertainty and variability

- **Decoding Parameter Tuning:** Model output variability is controlled via decoding hyperparameters:
  - Thinking Mode: `temperature=0.6`, `top_p=0.95`, `top_k=20`, `min_p=0.0`. (Greedy decoding is discouraged by developers in thinking mode to avoid repetition loops).
  - Non-Thinking Mode: `temperature=0.7`, `top_p=0.8`, `top_k=20`, `min_p=0.0`.
  - Deterministic Probing: `do_sample=False` (greedy search) is employed specifically for baseline smoke tests to ensure deterministic output comparisons.
- **Statistical Evaluation:** Metrics during fine-tuning are computed over isolated validation splits; duplicate detection (SHA-256 fingerprinting) prevents data leakage between train and test sets.
- **Variability Estimation:** For downstream deployment, uncertainty should be evaluated via multiple random seeds, prompt perturbations, and self-consistency sampling (e.g., majority voting across 5 runs).

---

#### Ethical considerations and biases

###### Data

- **Public & Synthetic Origin:** Pretrained by Alibaba Cloud on curated publicly available web text, scholarly articles, multilingual books, code repositories, and synthetically generated dialogues.
- **Classification & Sensitivity:** The base model does not intentionally contain classified, proprietary government data, or protected health information (PHI). However, as with all web-scale models, unintentional memorization of public personal information remains possible. Downstream fine-tuning datasets should be screened for confidential or personal data.

###### Human Life

- **No High-Impact Safety Clearance:** The model is not designed, evaluated, or certified for life-critical, medical diagnostic, criminal sentencing, or autonomous safety-critical systems.
- **Flourishing and Autonomy:** Outputs should not be used to restrict individual rights, deny housing, employment, or medical access, or make irreversible decisions affecting human well-being without human evaluation.

###### Mitigations

- **Upstream Safety Alignment:** Post-training alignment incorporates safety refusal vectors to prevent generating instructions for dangerous, illegal, or harmful tasks.
- **Assistant-Only Loss Masking:** In this pipeline, non-assistant tokens (system instructions, user prompts) receive `-100` target labels, preventing the model from learning to mimic user prompts or leaking sensitive user query formatting.
- **Supply-Chain Immutability:** Pinned to immutable commit SHA `c1899de289a04d12100db370d81485cdf75e47ca` with `trust_remote_code=False` enforced, guarding against arbitrary remote code execution or upstream poisoning.
- **Presence Penalty Guidelines:** Upstream best-practice guidelines recommend `presence_penalty` tuning (up to 1.5) to prevent repetition loops during extended generation.

###### Risks and harms

- **Hallucination & Misinformation:** Due to its small 0.6B parameter scale, the model may fabricate factual claims or mathematical steps with high confidence. Recipient: End users receiving plausible-sounding but erroneous information. Likelihood: Moderate to High on out-of-domain queries. Harm: Low in educational/smoke settings; High if unverified in production.
- **Stereotype Amplification:** Internet text exhibits cultural and demographic biases that can inadvertently surface in unconstrained generation.
- **Misuse for Spam & Phishing:** The lightweight nature of the model enables high-speed offline execution, which could be abused for automated generation of deceptive content if deployed without appropriate application guardrails.

###### Use cases

- **Prohibited and Disturbing Uses:**
  - Generation or dissemination of Child Sexual Abuse Material (CSAM) or Non-Consensual Intimate Imagery (NCII).
  - Encouraging or providing actionable instructions for suicide or self-harm.
  - Designing, synthesizing, or weaponizing chemical, biological, radiological, or nuclear (CBRN) threats.
  - Automated targeted harassment, stalking, or defamation.
  - Unlawful discrimination, surveillance, or deception designed to subvert democratic processes.

---

## Technical Specifications & Checkpoint Details

**Model Name:** Qwen3-0.6B  
**Model Identifier:** `Qwen/Qwen3-0.6B`  
**Developer:** Qwen Team, Alibaba Cloud  
**Pinned Revision:** `c1899de289a04d12100db370d81485cdf75e47ca`  
**License:** Apache License 2.0  
**Architecture:** Causal Language Model (`Qwen3ForCausalLM`)  
**Total Parameters:** ~0.6B (590M)  
**Non-Embedding Parameters:** ~0.44B (440M)  
**Layers:** 28  
**Attention Heads:** 16 Query heads, 8 Key-Value heads (Grouped Query Attention - GQA)  
**Context Length:** Up to 32,768 tokens  

## Quickstart

The code of Qwen3 has been in the latest Hugging Face `transformers` and we advise you to use the latest version of `transformers`.

With `transformers<4.51.0`, you will encounter the following error:
```
KeyError: 'qwen3'
```

The following contains a code snippet illustrating how to use the model generate content based on given inputs. 
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-0.6B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
```

For deployment, you can use `sglang>=0.4.6.post1` or `vllm>=0.8.5` or to create an OpenAI-compatible API endpoint:
- SGLang:
    ```shell
    python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B --reasoning-parser qwen3
    ```
- vLLM:
    ```shell
    vllm serve Qwen/Qwen3-0.6B --enable-reasoning --reasoning-parser deepseek_r1
    ```

For local use, applications such as Ollama, LMStudio, MLX-LM, llama.cpp, and KTransformers have also supported Qwen3.

## Switching Between Thinking and Non-Thinking Mode

> [!TIP]
> The `enable_thinking` switch is also available in APIs created by SGLang and vLLM. 
> Please refer to the Qwen documentation for [SGLang](https://qwen.readthedocs.io/en/latest/deployment/sglang.html#thinking-non-thinking-modes) and [vLLM](https://qwen.readthedocs.io/en/latest/deployment/vllm.html#thinking-non-thinking-modes) users.

### `enable_thinking=True`

By default, Qwen3 has thinking capabilities enabled, similar to QwQ-32B. This means the model will use its reasoning abilities to enhance the quality of generated responses. For example, when explicitly setting `enable_thinking=True` or leaving it as the default value in `tokenizer.apply_chat_template`, the model will engage its thinking mode.

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # True is the default value for enable_thinking
)
```

In this mode, the model will generate think content wrapped in a `<think>...</think>` block, followed by the final response.

> [!NOTE]
> For thinking mode, use `Temperature=0.6`, `TopP=0.95`, `TopK=20`, and `MinP=0` (the default setting in `generation_config.json`). **DO NOT use greedy decoding**, as it can lead to performance degradation and endless repetitions. For more detailed guidance, please refer to the [Best Practices](#best-practices) section.

### `enable_thinking=False`

The Qwen architecture provides a hard switch to strictly disable the model's thinking behavior, aligning its functionality with the previous Qwen2.5-Instruct models. This mode is particularly useful in scenarios where disabling thinking is essential for enhancing efficiency.

```python
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # Setting enable_thinking=False disables thinking mode
)
```

In this mode, the model will not generate any think content and will not include a `<think>...</think>` block.

> [!NOTE]
> For non-thinking mode, the developers suggest using `Temperature=0.7`, `TopP=0.8`, `TopK=20`, and `MinP=0`. For more detailed guidance, please refer to the [Best Practices](#best-practices) section.

### Advanced Usage: Switching Between Thinking and Non-Thinking Modes via User Input

The Qwen architecture provides a soft switch mechanism that allows users to dynamically control the model's behavior when `enable_thinking=True`. Specifically, you can add `/think` and `/no_think` to user prompts or system messages to switch the model's thinking mode from turn to turn. The model will follow the most recent instruction in multi-turn conversations.

Here is an example of a multi-turn conversation:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenChatbot:
    def __init__(self, model_name="Qwen/Qwen3-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.history = []

    def generate_response(self, user_input):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response

# Example Usage
if __name__ == "__main__":
    chatbot = QwenChatbot()

    # First input (without /think or /no_think tags, thinking mode is enabled by default)
    user_input_1 = "How many r's in strawberries?"
    print(f"User: {user_input_1}")
    response_1 = chatbot.generate_response(user_input_1)
    print(f"Bot: {response_1}")
    print("----------------------")

    # Second input with /no_think
    user_input_2 = "Then, how many r's in blueberries? /no_think"
    print(f"User: {user_input_2}")
    response_2 = chatbot.generate_response(user_input_2)
    print(f"Bot: {response_2}") 
    print("----------------------")

    # Third input with /think
    user_input_3 = "Really? /think"
    print(f"User: {user_input_3}")
    response_3 = chatbot.generate_response(user_input_3)
    print(f"Bot: {response_3}")
```

> [!NOTE]
> For API compatibility, when `enable_thinking=True`, regardless of whether the user uses `/think` or `/no_think`, the model will always output a block wrapped in `<think>...</think>`. However, the content inside this block may be empty if thinking is disabled.
> When `enable_thinking=False`, the soft switches are not valid. Regardless of any `/think` or `/no_think` tags input by the user, the model will not generate think content and will not include a `<think>...</think>` block.

## Agentic Use

Qwen3 excels in tool calling capabilities. The developers recommend using [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent) to make the best use of agentic ability of Qwen3. Qwen-Agent encapsulates tool-calling templates and tool-calling parsers internally, greatly reducing coding complexity.

To define the available tools, you can use the MCP configuration file, use the integrated tool of Qwen-Agent, or integrate other tools by yourself.
```python
from qwen_agent.agents import Assistant

# Define LLM
llm_cfg = {
    'model': 'Qwen3-0.6B',

    # Use the endpoint provided by Alibaba Model Studio:
    # 'model_type': 'qwen_dashscope',
    # 'api_key': os.getenv('DASHSCOPE_API_KEY'),

    # Use a custom endpoint compatible with OpenAI API:
    'model_server': 'http://localhost:8000/v1',  # api_base
    'api_key': 'EMPTY',

    # Other parameters:
    # 'generate_cfg': {
    #         # Add: When the response content is `<think>this is the thought</think>this is the answer;
    #         # Do not add: When the response has been separated by reasoning_content and content.
    #         'thought_in_content': True,
    #     },
}

# Define Tools
tools = [
    {'mcpServers': {  # You can specify the MCP configuration file
            'time': {
                'command': 'uvx',
                'args': ['mcp-server-time', '--local-timezone=Asia/Shanghai']
            },
            "fetch": {
                "command": "uvx",
                "args": ["mcp-server-fetch"]
            }
        }
    },
  'code_interpreter',  # Built-in tools
]

# Define Agent
bot = Assistant(llm=llm_cfg, function_list=tools)

# Streaming generation
messages = [{'role': 'user', 'content': 'https://qwenlm.github.io/blog/ Introduce the latest developments of Qwen'}]
for responses in bot.run(messages=messages):
    pass
print(responses)
```

## Best Practices

Upstream recommended settings for optimal performance:

1. **Sampling Parameters**:
   - For thinking mode (`enable_thinking=True`), use `Temperature=0.6`, `TopP=0.95`, `TopK=20`, and `MinP=0`. **DO NOT use greedy decoding**, as it can lead to performance degradation and endless repetitions.
   - For non-thinking mode (`enable_thinking=False`), we suggest using `Temperature=0.7`, `TopP=0.8`, `TopK=20`, and `MinP=0`.
   - For supported frameworks, you can adjust the `presence_penalty` parameter between 0 and 2 to reduce endless repetitions. However, using a higher value may occasionally result in language mixing and a slight decrease in model performance.

2. **Adequate Output Length**: The developers recommend using an output length of 32,768 tokens for most queries. For benchmarking on highly complex problems, such as those found in math and programming competitions, we suggest setting the max output length to 38,912 tokens. This provides the model with sufficient space to generate detailed and comprehensive responses, thereby enhancing its overall performance.

3. **Standardize Output Format**: The developers recommend using prompts to standardize model outputs when benchmarking.
   - **Math Problems**: Include "Please reason step by step, and put your final answer within \boxed{}." in the prompt.
   - **Multiple-Choice Questions**: Add the following JSON structure to the prompt to standardize responses: "Please show your choice in the `answer` field with only the choice letter, e.g., `"answer": "C"`."

4. **No Thinking Content in History**: In multi-turn conversations, the historical model output should only include the final output part and does not need to include the thinking content. It is implemented in the provided chat template in Jinja2. However, for frameworks that do not directly use the Jinja2 chat template, it is up to the developers to ensure that the best practice is followed.

## Model Ownership and Attribution

**Qwen3-0.6B** was developed and released by the **Qwen Team at Alibaba Cloud**.

This repository (`language-model-pipeline`) provides an integration, educational fine-tuning pipeline, and validation harness; it does not claim ownership or original authorship of the base model weights, tokenizer, or architecture. All base model weights and configurations are distributed under the terms of the Apache License 2.0.

## Citation

To cite the original Qwen3 work, please reference the technical report by the Qwen Team:

```bibtex
@misc{qwen3technicalreport,
      title={Qwen3 Technical Report}, 
      author={Qwen Team},
      year={2025},
      eprint={2505.09388},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.09388}, 
}
```
