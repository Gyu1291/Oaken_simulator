##########################################################################
##  Auto Model Allocator
##
##  Authors:  Junsoo    Kim   ( js.kim@hyperaccel.ai        )
##  Version:  1.3.1
##  Date:     2024-02-29      ( v1.3.1, init                )
##
##########################################################################

import transformers

# Model configuration
from perf_analyzer.models import (
  BloomConfig, BloomForCausalLM,
  FalconConfig, FalconForCausalLM,
  GemmaConfig, GemmaForCausalLM,
  GPT2Config, GPT2ForCausalLM,
  GPTBigCodeConfig, GPTBigCodeForCausalLM,
  GPTJConfig, GPTJForCausalLM,
  GPTNeoXConfig, GPTNeoXForCausalLM,
  LlamaConfig, LlamaForCausalLM,
  MistralConfig, MistralForCausalLM,
  MixtralConfig, MixtralForCausalLM,
  OLMoConfig, OLMoForCausalLM,
  OPTConfig, OPTForCausalLM,
  OrionConfig, OrionForCausalLM,
  PhiConfig, PhiForCausalLM,
  Qwen2Config, Qwen2ForCausalLM,
  StableLmConfig, StableLmForCausalLM
)

##########################################################################
## Class
##########################################################################

class AutoConfig():
  
  @classmethod
  def from_pretrained(cls, model_id, token=None):
    # Download huggingface checkpoint
    ckpt = transformers.AutoConfig.from_pretrained(model_id, token=token, trust_remote_code=True)
    # Route to appropriate architecture
    if "BloomForCausalLM" in ckpt.architectures:
      return BloomConfig.from_pretrained(ckpt)
    elif "FalconForCausalLM" in ckpt.architectures:
      return FalconConfig.from_pretrained(ckpt)
    elif "GemmaForCausalLM" in ckpt.architectures:
      return GemmaConfig.from_pretrained(ckpt)
    elif "GPT2LMHeadModel" in ckpt.architectures:
      return GPT2Config.from_pretrained(ckpt)
    elif "GPTBigCodeForCausalLM" in ckpt.architectures:
      return GPTBigCodeConfig.from_pretrained(ckpt)
    elif "GPTJForCausalLM" in ckpt.architectures:
      return GPTJConfig.from_pretrained(ckpt)
    elif "GPTNeoXForCausalLM" in ckpt.architectures:
      return GPTNeoXConfig.from_pretrained(ckpt)
    elif "LlamaForCausalLM" in ckpt.architectures:
      return LlamaConfig.from_pretrained(ckpt)
    elif "MistralForCausalLM" in ckpt.architectures:
      return MistralConfig.from_pretrained(ckpt)
    elif "MixtralForCausalLM" in ckpt.architectures:
      return MixtralConfig.from_pretrained(ckpt)
    elif "OPTForCausalLM" in ckpt.architectures:
      return OPTConfig.from_pretrained(ckpt)
    elif "OLMoForCausalLM" in ckpt.architectures:
      return OLMoConfig.from_pretrained(ckpt)
    elif "OrionForCausalLM" in ckpt.architectures:
      return OrionConfig.from_pretrained(ckpt)
    elif "PhiForCausalLM" in ckpt.architectures:
      return PhiConfig.from_pretrained(ckpt)
    elif "Qwen2ForCausalLM" in ckpt.architectures:
      return Qwen2Config.from_pretrained(ckpt)
    elif "StableLmForCausalLM" in ckpt.architectures:
      return StableLmConfig.from_pretrained(ckpt)
    
class AutoModelForCausalLM():
  
  @classmethod
  def from_pretrained(cls, model_id, token=None):
    # Download huggingface checkpoint
    ckpt = transformers.AutoConfig.from_pretrained(model_id, token=token, trust_remote_code=True)
    if "BloomForCausalLM" in ckpt.architectures:
      config = BloomConfig.from_pretrained(ckpt)
      return BloomForCausalLM(config)
    elif "FalconForCausalLM" in ckpt.architectures:
      config = FalconConfig.from_pretrained(ckpt)
      return FalconForCausalLM(config)
    elif "GemmaForCausalLM" in ckpt.architectures:
      config = GemmaConfig.from_pretrained(ckpt)
      return GemmaForCausalLM(config)
    elif "GPT2LMHeadModel" in ckpt.architectures:
      config = GPT2Config.from_pretrained(ckpt)
      return GPT2ForCausalLM(config)
    elif "GPTBigCodeForCausalLM" in ckpt.architectures:
      config = GPTBigCodeConfig.from_pretrained(ckpt)
      return GPTBigCodeForCausalLM(config)
    elif "GPTJForCausalLM" in ckpt.architectures:
      config = GPTJConfig.from_pretrained(ckpt)
      return GPTJForCausalLM(config)
    elif "GPTNeoXForCausalLM" in ckpt.architectures:
      config = GPTNeoXConfig.from_pretrained(ckpt)
      return GPTNeoXForCausalLM(config)
    elif "LlamaForCausalLM" in ckpt.architectures:
      config = LlamaConfig.from_pretrained(ckpt)
      return LlamaForCausalLM(config)
    elif "MistralForCausalLM" in ckpt.architectures:
      config = MistralConfig.from_pretrained(ckpt)
      return MistralForCausalLM(config)
    elif "MixtralForCausalLM" in ckpt.architectures:
      config = MixtralConfig.from_pretrained(ckpt)
      return MixtralForCausalLM(config)
    elif "OLMoForCausalLM" in ckpt.architectures:
      config = OLMoConfig.from_pretrained(ckpt)
      return OLMoForCausalLM(config)
    elif "OPTForCausalLM" in ckpt.architectures:
      config = OPTConfig.from_pretrained(ckpt)
      return OPTForCausalLM(config)
    elif "OrionForCausalLM" in ckpt.architectures:
      config = OrionConfig.from_pretrained(ckpt)
      return OrionForCausalLM(config)
    elif "PhiForCausalLM" in ckpt.architectures:
      config = PhiConfig.from_pretrained(ckpt)
      return PhiForCausalLM(config)
    elif "Qwen2ForCausalLM" in ckpt.architectures:
      config = Qwen2Config.from_pretrained(ckpt)
      return Qwen2ForCausalLM(config)
    elif "StableLmForCausalLM" in ckpt.architectures:
      config = StableLmConfig.from_pretrained(ckpt)
      return StableLmForCausalLM(config)

##########################################################################