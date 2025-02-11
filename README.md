![](https://images.pexels.com/photos/1181271/pexels-photo-1181271.jpeg)


## Finetune Granite3.1 for Reasoning
IBM's Granite3.1 models are powerful foundation models designed for enterprise AI applications. With fine-tuning, we can enhance Granite’s reasoning capabilities, making it competitive with state-of-the-art models in complex logical and analytical tasks.

Just as DeepSeek-R1 has demonstrated strong reasoning performance, Granite3.1 can also be optimized for superior reasoning through efficient fine-tuning techniques. Unsloth provides an accessible way to fine-tune these models, enabling greater adaptability for specific tasks.

This blog post will guide you through the process of fine-tuning Granite3.1 using Python and the Unsloth library. We'll cover each step in detail, providing clear explanations and all the Python code needed to get started. Let’s unlock the full potential of Granite3.1 and elevate its reasoning capabilities!



## What We'll Cover

  * **Setting up your environment:** Installing the necessary libraries, including Unsloth.
  * **Preparing your data:**  Understanding and loading a dataset suitable for reasoning fine-tuning.
  * **Fine-tuning Granite 3.1 Model:**  Using Unsloth to efficiently train your model with GRPO (Guided Reward Policy Optimization).
  * **Inference with your finetuned model:**  Running your custom model to see the improvements.
  * **Saving and sharing your model:**  Options for saving your finetuned model in various formats like LoRA, merged weights, and GGUF for llama.cpp compatibility.

## Prerequisites

Before you begin, ensure you have:

  * **Python installed:**  Python 3.8 or higher is recommended.
  * **A Google Colab environment (optional but recommended for ease of use):**  The code examples are designed to run smoothly on a free Google Colab instance. You can also run it on your local machine if you have the necessary hardware.
  * **Basic understanding of Python and Machine Learning concepts:** While we aim to explain everything step-by-step, a basic understanding will be helpful.

## Step 1: Installation

First, we need to install the Unsloth library and other dependencies. Unsloth simplifies the fine-tuning process and makes it incredibly efficient.

Open your Colab notebook (or your local terminal) and run the following commands:

```python
# %%capture  # Only in Colab to suppress output
# Skip restarting message in Colab - Colab specific lines, you can ignore if running locally
import sys; modules = list(sys.modules.keys())
for x in modules: sys.modules.pop(x) if "PIL" in x or "google" in x else None

!pip install unsloth vllm
!pip install --upgrade pillow
# If you are running this notebook on local, you need to install `diffusers` too (uncomment if needed)
# !pip install diffusers
# Temporarily install a specific TRL nightly version (required for GRPO)
!pip install git+[https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b](https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b)
```

**Explanation:**

  * `!pip install unsloth vllm`: This command installs the core Unsloth library and `vllm`, which enables fast inference.
  * `!pip install --upgrade pillow`:  Upgrades the `pillow` library, often needed for image processing tasks (though not directly used in this notebook, it's a good practice to keep it updated).
  * `!pip install diffusers` (commented out): This line is commented out but included for local installations as `diffusers` might be needed for some functionalities in Unsloth in certain environments. You likely won't need it for this fine-tuning task.
  * `!pip install git+https://...`:  This installs a specific nightly version of the `trl` (Transformers Reinforcement Learning) library from Hugging Face. This specific commit is likely needed for compatibility with the GRPO trainer in Unsloth at the time of writing.

**Important Note for Local Installation:** If you are running this locally, you might need to install `diffusers` separately if you encounter import errors related to it later.

## Step 2:  Prepare for GRPO with Unsloth Patch

Unsloth uses a patching mechanism to seamlessly integrate Reinforcement Learning algorithms like GRPO. We need to apply this patch before loading our model.

```python
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
```

**Explanation:**

  * `from unsloth import FastLanguageModel, PatchFastRL`:  Imports the necessary classes from the Unsloth library.
  * `PatchFastRL("GRPO", FastLanguageModel)`:  This is the crucial patching step. It tells Unsloth that we intend to use the GRPO algorithm and applies the necessary modifications to the `FastLanguageModel` class to support it.

## Step 3: Load Granite Model

Now, we load the granite model we want to fine-tune. We'll use `ibm-granite/granite-3.1-2b-instruct` as the base model. We'll also set some important parameters for efficient fine-tuning.

```python
from unsloth import is_bfloat16_supported
import torch

max_seq_length = 512  # Can increase for longer reasoning traces
lora_rank = 32        # Larger rank = smarter, but potentially slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "ibm-granite/granite-3.1-2b-instruct", # Or a DeepSeek distilled model based on Llama
    max_seq_length = max_seq_length,
    load_in_4bit = True,        # Use 4-bit quantization for memory efficiency
    fast_inference = True,      # Enable vLLM for faster inference later
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Adjust if you run out of memory
)

Here I suggest modify vllm_utils.py of Usloth to make compatible with granite models.
```
# Fix up rotary_emb by re-initing them
    for module in new_model.modules():
        if hasattr(module, "rotary_emb"):
            module.rotary_emb = module.rotary_emb.__class__(
                config = config,
               # device = "cuda:0",
            )
        pass
    pass
```
then 
```
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,              # LoRA rank, must match max_lora_rank
    target_modules = [          # Modules to apply LoRA to
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],  # Consider removing QKVO if memory is limited
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # For longer context fine-tuning
    random_state = 3407,        # For reproducibility
)
```

**Explanation:**

  * **Parameters:**
      * `max_seq_length = 512`:  Sets the maximum sequence length for the model. You can increase this if your reasoning tasks require longer context, but it will consume more memory.
      * `lora_rank = 32`: Defines the rank of the LoRA (Low-Rank Adaptation) matrices. LoRA is a parameter-efficient fine-tuning technique. Higher rank can lead to better performance but increased memory usage.
  * **`FastLanguageModel.from_pretrained(...)`:** Loads the pre-trained model and tokenizer.
      * `model_name = "ibm-granite/granite-3.1-2b-instruct"`:  Specifies the base model to use.  You can replace this with a specific DeepSeek-R1 distilled model name from Hugging Face if you know one based on Llama architecture is available on Unsloth's collection.  If you want to fine-tune a Qwen-distilled DeepSeek R1, you would use a Qwen base model here instead.
      * `load_in_4bit = True`: Enables 4-bit quantization. This dramatically reduces memory usage, allowing you to fine-tune larger models even on GPUs with limited memory (like free Colab T4).
      * `fast_inference = True`:  Enables vLLM for faster inference after fine-tuning.
      * `max_lora_rank = lora_rank`:  Sets the maximum LoRA rank, ensuring consistency.
      * `gpu_memory_utilization = 0.6`:  Controls how much of your GPU memory Unsloth will utilize. Reduce this value (e.g., to `0.5` or `0.4`) if you encounter out-of-memory errors.
  * **`FastLanguageModel.get_peft_model(...)`:**  Adds the LoRA adapters to the base model, preparing it for parameter-efficient fine-tuning.
      * `r = lora_rank`:  Sets the LoRA rank again for the PEFT model.
      * `target_modules = [...]`: Specifies the layers within the model where LoRA adapters will be inserted. These are typically the attention and feed-forward layers in transformer models. You might need to adjust this list if you face memory issues – removing `"q_proj", "k_proj", "v_proj", "o_proj"` can reduce memory footprint at a potential performance trade-off.
      * `lora_alpha = lora_rank`: A scaling factor for LoRA. Setting it equal to the rank is a common practice.
      * `use_gradient_checkpointing = "unsloth"`: Enables gradient checkpointing, a technique to reduce memory usage during training, especially helpful for longer sequences.
      * `random_state = 3407`:  Sets a random seed for reproducibility.

## Step 4: Data Preparation

For fine-tuning a reasoning model, we need a dataset that challenges the model's reasoning abilities. We'll use the GSM8K dataset, a popular benchmark for math word problems. We'll also define reward functions that guide the GRPO training process.

```python
import re
from datasets import load_dataset, Dataset

# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting - in this case, we're using 0-shot
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001 # Penalize extra text after </answer>
    if text.count("\n</answer>") == 1: # Handle both newline and no newline cases
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001 # Penalize extra text after </answer>

    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]
```

**Explanation:**

  * **Dataset Loading and Preprocessing:**

      * `SYSTEM_PROMPT`:  Defines a system prompt that instructs the model to respond in a specific XML format with `<reasoning>` and `<answer>` tags. This guides the model to produce chain-of-thought reasoning.
      * `XML_COT_FORMAT`: A string template for the XML format.
      * `extract_xml_answer(text)` and `extract_hash_answer(text)`: Helper functions to extract answers from the model's output and the GSM8K dataset, respectively.  GSM8K uses "\#\#\#\#" to denote the answer.
      * `get_gsm8k_questions(split="train")`: This function loads the GSM8K dataset from Hugging Face Datasets using `load_dataset('openai/gsm8k', 'main')[split]`. It then preprocesses the data using `dataset.map(...)` to format it for training. Each data point will have a `prompt` (system prompt + user question) and an `answer` (extracted from the dataset).
      * `dataset = get_gsm8k_questions()`: Loads the training split of the GSM8K dataset.

  * **Reward Functions:** These functions are crucial for GRPO. They assess the model's generated responses and provide rewards that guide the training process.

      * `correctness_reward_func`:  The most important reward. It checks if the extracted answer from the model's response exactly matches the ground truth answer from the dataset. It gives a high reward (2.0) for correct answers and 0.0 for incorrect ones.  It also prints example questions, answers, responses and extractions for monitoring training.
      * `int_reward_func`:  Rewards the model if the extracted answer is a digit. This encourages the model to output numerical answers for math problems.
      * `strict_format_reward_func`:  Encourages strict adherence to the XML format with newline characters as defined by the `pattern`. It gives a small reward (0.5) if the response strictly matches the format.
      * `soft_format_reward_func`: A more lenient format reward. It checks if the response generally contains `<reasoning>` and `<answer>` tags (allowing for more flexibility in formatting within the tags).
      * `xmlcount_reward_func`:  This reward function is designed to encourage the model to produce the XML format by counting the occurrences of the `<reasoning>`, `</reasoning>`, `<answer>`, and `</answer>` tags. It gives small rewards for each tag present and penalizes extra text after the answer tag, encouraging cleaner formatting.

## Step 5: Fine-tuning with GRPO Trainer

Now, we're ready to set up and run the GRPO trainer using the configurations and data we've prepared.

```python
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    use_vllm = True,            # Enable vLLM for faster inference during training
    learning_rate = 5e-6,       # Learning rate for the optimizer
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,          # Adam optimizer parameters
    weight_decay = 0.1,         # Weight decay for regularization
    warmup_ratio = 0.1,         # Warmup ratio for learning rate scheduler
    lr_scheduler_type = "cosine", # Cosine learning rate decay
    optim = "paged_adamw_8bit", # Paged AdamW optimizer for memory efficiency
    logging_steps = 1,          # Log every step
    bf16 = is_bfloat16_supported(), # Use bfloat16 if supported for faster training on compatible GPUs
    fp16 = not is_bfloat16_supported(), # Fallback to fp16 if bfloat16 is not supported
    per_device_train_batch_size = 1, # Batch size per GPU
    gradient_accumulation_steps = 1, # Accumulate gradients over steps to simulate larger batch size (increase for smoother training)
    num_generations = 6,        # Number of generations per prompt for GRPO
    max_prompt_length = 256,     # Maximum length of the input prompt
    max_completion_length = 200, # Maximum length of the generated completion
    # num_train_epochs = 1,     # For full training run, set num_train_epochs to 1
    max_steps = 250,            # Limit training to 250 steps for this example
    save_steps = 250,           # Save checkpoint every 250 steps
    max_grad_norm = 0.1,        # Maximum gradient norm for gradient clipping
    report_to = "none",         # Disable Weights & Biases logging for this example (can be set to "wandb")
    output_dir = "outputs",       # Directory to save training outputs and checkpoints
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [             # List of reward functions to use
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)

trainer.train()
```

**Explanation:**

  * **`GRPOConfig`:**  This class defines the training hyperparameters for GRPO. Let's break down the key parameters:

      * `use_vllm = True`:  Enables vLLM for fast inference during training. GRPO involves generating multiple completions for each prompt, and vLLM significantly speeds up this process.
      * **Learning Rate and Optimizer Settings (`learning_rate`, `adam_beta1`, `adam_beta2`, `weight_decay`, `warmup_ratio`, `lr_scheduler_type`, `optim`):** These parameters control the optimization process and learning rate schedule. The values provided are typical for fine-tuning large language models. `paged_adamw_8bit` is a memory-efficient optimizer.
      * `logging_steps = 1`:  Logs training metrics at every step, allowing you to monitor progress closely.
      * `bf16 = is_bfloat16_supported()` and `fp16 = not is_bfloat16_supported()`: Enables mixed-precision training (bf16 if your GPU supports it, otherwise fp16) for faster and more memory-efficient training.
      * `per_device_train_batch_size = 1`:  Sets the batch size per GPU. GRPO training can be memory-intensive, so a batch size of 1 is common, especially on free Colab.
      * `gradient_accumulation_steps = 1`:  Accumulates gradients over multiple steps. Increasing this (e.g., to 4) can simulate a larger batch size and potentially lead to smoother training, but increases training time per step.
      * `num_generations = 6`:  This is specific to GRPO. It determines how many completions the model generates for each prompt during training. GRPO compares these completions and uses rewards to guide the model towards better responses.  Decreasing this value can reduce memory usage.
      * `max_prompt_length` and `max_completion_length`:  Set maximum lengths for prompts and generated completions, respectively.
      * `max_steps = 250`: Limits the training process to 250 steps for this example.  For a full fine-tuning run, you would increase this significantly or use `num_train_epochs = 1`.
      * `save_steps = 250`: Saves a checkpoint of the model every 250 steps.
      * `max_grad_norm = 0.1`:  Applies gradient clipping to prevent exploding gradients during training.
      * `report_to = "none"`: Disables logging to Weights & Biases (a popular experiment tracking platform). You can change this to `"wandb"` if you want to use Weights & Biases and have it set up.
      * `output_dir = "outputs"`: Specifies the directory where training outputs, including checkpoints, will be saved.

  * **`GRPOTrainer`:**  This class orchestrates the GRPO training process.

      * `model = model`:  Passes the loaded and LoRA-adapted model.
      * `processing_class = tokenizer`:  Passes the tokenizer.
      * `reward_funcs = [...]`: Provides the list of reward functions we defined earlier. The trainer will use these functions to evaluate the generated completions.
      * `args = training_args`:  Passes the training configuration defined by `GRPOConfig`.
      * `train_dataset = dataset`:  Passes the GSM8K dataset.

  * **`trainer.train()`:**  Starts the fine-tuning process\!  During training, you will see logs that include training loss, reward values, and other metrics.  The goal is to see the "reward" values increase over training steps, indicating that the model is learning to generate better reasoning and answers. **Be patient\!** It might take 100-200 steps before you see noticeable improvements in reward.

## Step 6: Inference - Testing Your Finetuned Model

After fine-tuning, let's see how our model performs\! We'll compare the base model's output with the output from our fine-tuned model (with the LoRA adapters).

**First, let's test the base model (without GRPO fine-tuning):**

```python
text = tokenizer.apply_chat_template([
    {"role" : "user", "content" : "Calculate pi."},
], tokenize = False, add_generation_prompt = True)

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = 1024, # Allow longer outputs if needed
)
output = model.fast_generate(
    [text],
    sampling_params = sampling_params,
    lora_request = None, # No LoRA loaded - using base model
)[0].outputs[0].text

print("Base Model Output:\n", output)
```

**Explanation:**

  * **`tokenizer.apply_chat_template(...)`:**  Formats the input prompt for the model using the tokenizer's chat template. This is important to ensure the input is in the expected format for the model (especially for instruction-following models).
  * **`SamplingParams(...)`:** Defines parameters for text generation:
      * `temperature = 0.8`: Controls the randomness of generation. Lower values (closer to 0) make the output more deterministic, higher values (closer to 1) make it more creative and random.
      * `top_p = 0.95`:  Nucleus sampling - considers the most probable tokens whose cumulative probability exceeds 0.95.
      * `max_tokens = 1024`:  Maximum number of tokens to generate.
  * **`model.fast_generate(...)`:**  Uses vLLM's fast generation to generate text.
      * `lora_request = None`:  Crucially, we set `lora_request = None` to use the base model without any LoRA adapters loaded.

**Now, let's test the fine-tuned model (with the LoRA adapters we trained):**

First, save the LoRA adapters:

```python
model.save_lora("grpo_saved_lora")
```

Then, load the saved LoRA and generate text:

```python
text = tokenizer.apply_chat_template([
    {"role" : "system", "content" : SYSTEM_PROMPT}, # Include system prompt for reasoning format
    {"role" : "user", "content" : "Calculate pi."},
], tokenize = False, add_generation_prompt = True)

from vllm import SamplingParams
sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = 1024,
)
output = model.fast_generate(
    [text],
    sampling_params = sampling_params,
    lora_request = model.load_lora("grpo_saved_lora"), # Load our finetuned LoRA
)[0].outputs[0].text

print("Finetuned Model Output:\n", output)
```

**Explanation:**

  * `model.save_lora("grpo_saved_lora")`: Saves the trained LoRA adapters to a directory named "grpo\_saved\_lora".
  * `lora_request = model.load_lora("grpo_saved_lora")`:  Loads the saved LoRA adapters when calling `model.fast_generate()`. This applies the fine-tuned weights to the model for inference.
  * We've also added the `SYSTEM_PROMPT` to the chat template this time, to encourage the finetuned model to use the reasoning format we trained it on.

Compare the outputs of the base model and the fine-tuned model. You should observe that the fine-tuned model, after GRPO training, provides better reasoning and potentially more accurate answers, especially for questions related to the GSM8K dataset's domain (math word problems).

## Step 7: Saving Your Finetuned Model

Unsloth provides flexible options for saving your fine-tuned model in different formats.

**Saving to Merged Weights (Float16 or Int4 for vLLM):**

This merges the LoRA adapters back into the base model weights. This creates a standalone model that doesn't require separate LoRA files, which is convenient for deployment, especially with vLLM.

```python
# Merge to 16bit (Float16) - uncomment to save
# if False: model.save_pretrained_merged("model_merged_fp16", tokenizer, save_method = "merged_16bit")
# if False: model.push_to_hub_merged("your_username/deepseek-r1-finetuned-fp16", tokenizer, save_method = "merged_16bit", token = "YOUR_HUGGINGFACE_TOKEN")

# Merge to 4bit (Int4) - uncomment to save
# if False: model.save_pretrained_merged("model_merged_int4", tokenizer, save_method = "merged_4bit")
# if False: model.push_to_hub_merged("your_username/deepseek-r1-finetuned-int4", tokenizer, save_method = "merged_4bit", token = "YOUR_HUGGINGFACE_TOKEN")
```

**Explanation:**

  * `model.save_pretrained_merged(...)`: Saves the merged model.
      * `"model_merged_fp16"` or `"model_merged_int4"`:  The directory where the merged model will be saved.
      * `tokenizer`: The tokenizer.
      * `save_method = "merged_16bit"` or `save_method = "merged_4bit"`: Specifies the saving method. `"merged_16bit"` saves in float16 (good for vLLM), and `"merged_4bit"` saves in int4 (even more memory efficient).
  * `model.push_to_hub_merged(...)`: Pushes the merged model to your Hugging Face Hub repository.
      * `"your_username/deepseek-r1-finetuned-fp16"` or `"your_username/deepseek-r1-finetuned-int4"`: Replace `"your_username"` with your Hugging Face username and choose a repository name.
      * `token = "YOUR_HUGGINGFACE_TOKEN"`: You need a Hugging Face Hub token to push to the Hub. You can generate one in your Hugging Face settings ([https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).

**Saving LoRA Adapters Only:**

If you prefer to keep the LoRA adapters separate (smaller file size, can be loaded on top of different base models), you can save them like this:

```python
# Save LoRA adapters only - uncomment to save
# if False: model.save_pretrained_merged("lora_adapters", tokenizer, save_method = "lora")
# if False: model.push_to_hub_merged("your_username/deepseek-r1-finetuned-lora", tokenizer, save_method = "lora", token = "YOUR_HUGGINGFACE_TOKEN")
```

**Saving to GGUF for llama.cpp Compatibility:**

For use with `llama.cpp` and tools like Jan or Open WebUI, you can convert your fine-tuned model to the GGUF format:

```python
# Save to GGUF (various quantization methods available)

# Save to 8bit Q8_0 - uncomment to save
# if False: model.save_pretrained_gguf("model_gguf_q8_0", tokenizer)
# if False: model.push_to_hub_gguf("your_username/deepseek-r1-finetuned-gguf-q8_0", tokenizer, token = "YOUR_HUGGINGFACE_TOKEN")

# Save to 16bit GGUF - uncomment to save
# if False: model.save_pretrained_gguf("model_gguf_f16", tokenizer, quantization_method = "f16")
# if False: model.push_to_hub_gguf("your_username/deepseek-r1-finetuned-gguf-f16", tokenizer, quantization_method = "f16", token = "YOUR_HUGGINGFACE_TOKEN")

# Save to q4_k_m GGUF (recommended balance of size and performance) - uncomment to save
# if False: model.save_pretrained_gguf("model_gguf_q4_k_m", tokenizer, quantization_method = "q4_k_m")
# if False: model.push_to_hub_gguf("your_username/deepseek-r1-finetuned-gguf-q4_k_m", tokenizer, quantization_method = "q4_k_m", token = "YOUR_HUGGINGFACE_TOKEN")

# Save to multiple GGUF options at once (faster if you want multiple quantizations) - uncomment to save
# if False:
#     model.push_to_hub_gguf(
#         "your_username/deepseek-r1-finetuned-gguf-multi", # Choose a repo name
#         tokenizer,
#         quantization_method = ["q4_k_m", "q8_0", "q5_k_m"], # Specify quantization methods
#         token = "YOUR_HUGGINGFACE_TOKEN",
#     )
```

**Explanation:**

  * `model.save_pretrained_gguf(...)`: Saves the model in GGUF format.
      * `"model_gguf_q8_0"`, `"model_gguf_f16"`, `"model_gguf_q4_k_m"`:  Directories to save the GGUF files.
      * `tokenizer`: The tokenizer.
      * `quantization_method`:  Specifies the quantization method for GGUF. Common options include `"q8_0"` (fast conversion, larger size), `"f16"` (16-bit float GGUF), and `"q4_k_m"` (recommended balance of size and performance). You can also provide a list of quantization methods to save multiple GGUF versions at once.
  * `model.push_to_hub_gguf(...)`: Pushes the GGUF files to the Hugging Face Hub, similar to `push_to_hub_merged`.

## Conclusion

Congratulations\! You've successfully fine-tuned a Granite model for reasoning using Unsloth. You've learned how to set up your environment, prepare your data, train your model with GRPO, test its performance, and save it in various formats.

This is just a starting point.  To further improve your model's reasoning abilities:

  * **Train for longer:** Increase `max_steps` or set `num_train_epochs = 1` for a full training run.
  * **Experiment with hyperparameters:**  Adjust learning rate, LoRA rank, batch size, and other training parameters.
  * **Use a larger and more diverse reasoning dataset:** Explore other reasoning datasets beyond GSM8K.
  * **Increase sequence length:** If your reasoning tasks require longer context, increase `max_seq_length`.

Unsloth makes fine-tuning powerful models like DeepSeek-R1 accessible to everyone. We encourage you to explore further, experiment, and build amazing reasoning applications\!


We're excited to see what you build\! Happy fine-tuning\!
