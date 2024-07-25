<!---
Copyright 2023 The HyperAccel. All rights reserved.
-->

<p align="center">
    <br>
    <img src="docs/images/logo.png" width="400"/>
    <br>
<p>

Provides an implementation of python package for Oaken, with a focus on performance and versatility.

# Oaken: Accelerating Batched LLM Inference via Online Key-Value Cache Compression


## Main features
* Predict the performance of the LPU based on several hardware parameters.
* Calculate the Key-Value cache size with HuggingFace models.
* Automatically visualize the profile results based on web GUI.
* ~~Easy to construct user defined program~~ 

## Installation

### Requirements
* OS : Linux
* Python : Python 3.9 - 3.11

### Install with Conda Environment
```bash
$ # Create a new conda environments
$ conda create -n perf-env python=3.9 -y
$ conda activate perf-env

$ # Install the requirements
$ pip install -r requirements.txt
```

## Quick Start

Performance analyzer can be deployed as a server with HTTP protocol. By default, it starts the server at `http://localhost:7860`. ~~You can specify the address with `--host` and `--port` arguments~~. We are actively support for more functionality.

Start the server for private analyzer (Only for internal network):
```bash
$ python -m perf_analyzer.gui.private.web
```

For external, you should run the public analyzer.
```bash
$ python -m perf_analyzer.gui.public.web
```

## Supported Models

|Architecture|Models|Example HuggingFace Models|
|-|-|-|
|`FalconForCausalLM`      |Falcon                                                       |`tiiuae/falcon-7b`, `tiiuae/falcon-40b`, etc.            |
|`GemmaForCausalLM`       |Gemma                                                        |`google/gemma-2b`, `google/gemma-7b`, etc.               |
|`GPT2LMHeadModel`        |GPT-2                                                        |`gpt2`, `gpt2-xl`, etc.                                  |
|`GPTBigCodeForCausalLM`  |StarCoder, SantaCoder, WizardCoder                           |`bidcode/starcoder`, etc.                                |
|`GPTJForCausalLM`        |GPT-J                                                        |`EleutherAI/gpt-j-6b`, etc.                              |
|`GPTNeoXForCausalLM`     |GPT-NeoX,<br>Pythia,<br>OpenAssistant,<br>Dolly,<br>StableLM |`EleutherAI/gpt-neox-20b`, `EleutherAI/pythia-12b`, etc. |
|`LlamaForCausalLM`       |LLaMA,<br>LLAMA-2,<br>Alpaca,<br>Yi                          |`meta-llama/Llama-2-7b-hf`, `01-ai/Yi-6B`, etc.          |
|`MistralForCausalLM`     |Mistral<br>Mistral-Instruct                                  |`mistralai/Mistral-7B-v0.1`, etc.                        |
|`OPTForCausalLM`         |OPT                                                          |`facebook/opt-1.3b`, `facebook/opt-66b`, etc.            |
|`PhiForCausalLM`         |Phi                                                          |`microsoft/phi-1_5`, `microsoft/phi-2`, etc.             |

---
