"""
DeepSeek Teacher - Teacher model for knowledge distillation

Wraps DeepSeek-Coder-33B as a fallback and teaching source for Nawal.
Used when Nawal's confidence is below threshold.

Model: DeepSeek-Coder-33B-Instruct
- 33B parameters
- MIT License (commercial friendly)
- Excellent at coding, reasoning, multilingual tasks
- Runs efficiently with vLLM on single GPU (A100/H100)
"""

import torch
from typing import Optional, List, Dict
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DeepSeekConfig:
    """Configuration for DeepSeek teacher model"""
    model_name: str = "deepseek-ai/deepseek-coder-33b-instruct"
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    tensor_parallel_size: int = 1  # Number of GPUs
    quantization: Optional[str] = "awq"  # "awq", "gptq", "bitsandbytes", or None
    gpu_memory_utilization: float = 0.9


class DeepSeekTeacher:
    """
    DeepSeek-Coder-33B Teacher Model
    
    Serves as:
    1. Fallback for low-confidence Nawal queries (< 0.75 threshold)
    2. Teacher for knowledge distillation (soft targets)
    3. Quality benchmark for Nawal improvement
    
    Features:
    - vLLM for efficient inference (batching, paged attention)
    - Optional quantization (AWQ/GPTQ) for memory efficiency
    - Tensor parallelism for multi-GPU setups
    - Response caching for repeated queries
    """
    
    def __init__(self, config: Optional[DeepSeekConfig] = None):
        self.config = config or DeepSeekConfig()
        self.model = None
        self.tokenizer = None
        self.cache = {}  # Simple response cache
        
        logger.info(f"Initializing DeepSeek teacher: {self.config.model_name}")
    
    def load_model(self) -> None:
        """
        Load DeepSeek model with vLLM for efficient inference
        
        Uses vLLM features:
        - PagedAttention for KV cache optimization
        - Continuous batching for throughput
        - Quantization for memory efficiency
        """
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            # Load model with vLLM
            self.model = LLM(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                quantization=self.config.quantization,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                trust_remote_code=True,
            )
            
            logger.info("DeepSeek model loaded successfully")
            
        except ImportError:
            logger.warning(
                "vLLM not installed. Falling back to HuggingFace transformers "
                "(slower inference). Install vLLM for production: pip install vllm"
            )
            self._load_with_transformers()
    
    def _load_with_transformers(self) -> None:
        """Fallback: Load with HuggingFace transformers (slower)"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        # Load with quantization if specified
        if self.config.quantization == "bitsandbytes":
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        logger.info("DeepSeek model loaded with transformers")
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_logits: bool = False,
    ) -> Dict:
        """
        Generate response from DeepSeek
        
        Args:
            prompt: Input text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            return_logits: Whether to return logits for distillation
        
        Returns:
            Dictionary containing:
                - text: Generated response
                - tokens: Token IDs (if return_logits=True)
                - logits: Output logits (if return_logits=True)
                - cached: Whether response was from cache
        """
        # Check cache first
        cache_key = (prompt, max_tokens, temperature)
        if cache_key in self.cache:
            logger.debug("Returning cached DeepSeek response")
            return {**self.cache[cache_key], "cached": True}
        
        # Prepare parameters
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature
        
        # Generate with vLLM if available
        if hasattr(self.model, 'generate') and hasattr(self.model, '__module__') and 'vllm' in self.model.__module__:
            response = self._generate_vllm(prompt, max_tokens, temperature, return_logits)
        else:
            response = self._generate_transformers(prompt, max_tokens, temperature, return_logits)
        
        # Cache response
        response["cached"] = False
        self.cache[cache_key] = {k: v for k, v in response.items() if k != "cached"}
        
        return response
    
    def _generate_vllm(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        return_logits: bool
    ) -> Dict:
        """Generate using vLLM engine"""
        from vllm import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=self.config.top_p,
            max_tokens=max_tokens,
            logprobs=5 if return_logits else None,  # Return top-5 logprobs for distillation
        )
        
        outputs = self.model.generate([prompt], sampling_params)
        output = outputs[0]
        
        result = {
            "text": output.outputs[0].text,
        }
        
        if return_logits:
            # Extract logits from logprobs (approximation)
            result["logprobs"] = output.outputs[0].logprobs
            result["tokens"] = output.outputs[0].token_ids
        
        return result
    
    def _generate_transformers(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        return_logits: bool
    ) -> Dict:
        """Generate using HuggingFace transformers"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=self.config.top_p,
                do_sample=True if temperature > 0 else False,
                output_scores=return_logits,
                return_dict_in_generate=True,
            )
        
        generated_ids = outputs.sequences[:, inputs.input_ids.size(1):]
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        result = {
            "text": generated_text,
        }
        
        if return_logits:
            # Stack scores from all generation steps
            result["logits"] = torch.stack(outputs.scores, dim=1)  # [batch, seq_len, vocab]
            result["tokens"] = generated_ids
        
        return result
    
    def get_soft_targets(
        self,
        prompt: str,
        temperature: float = 2.0
    ) -> torch.Tensor:
        """
        Get soft targets for knowledge distillation
        
        Higher temperature (2.0-4.0) smooths the distribution,
        revealing more of DeepSeek's "knowledge"
        
        Args:
            prompt: Input text
            temperature: Distillation temperature (higher = softer)
        
        Returns:
            Soft probability distribution over vocabulary
        """
        response = self.generate(prompt, temperature=temperature, return_logits=True)
        
        if "logits" in response:
            logits = response["logits"]
            # Apply temperature scaling
            soft_targets = torch.nn.functional.softmax(logits / temperature, dim=-1)
            return soft_targets
        else:
            logger.warning("Logits not available for soft targets")
            return None
    
    def clear_cache(self) -> None:
        """Clear response cache"""
        cache_size = len(self.cache)
        self.cache.clear()
        logger.info(f"Cleared {cache_size} cached responses")


# Convenience function
def create_deepseek_teacher(
    quantization: Optional[str] = "awq",
    num_gpus: int = 1,
) -> DeepSeekTeacher:
    """
    Create DeepSeek teacher with common configuration
    
    Args:
        quantization: Quantization method ("awq", "gptq", "bitsandbytes", None)
        num_gpus: Number of GPUs for tensor parallelism
    
    Returns:
        Initialized DeepSeek teacher
    """
    config = DeepSeekConfig(
        quantization=quantization,
        tensor_parallel_size=num_gpus,
    )
    teacher = DeepSeekTeacher(config)
    teacher.load_model()
    return teacher
