# Copyright (c) 2025, Alibaba Cloud and its affiliates;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Registration function for explicit use
def register_qwen3_asr():
    """Register FunAudioChat classes with AutoConfig and AutoModel."""
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor
    from .configuration_qwen3_asr import Qwen3ASRConfig, Qwen3ASRThinkerConfig, Qwen3ASRAudioEncoderConfig
    from .modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
    from .processing_qwen3_asr import Qwen3ASRProcessor
    
    # Register configurations
    AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    
    # Register processor
    AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
    
    # Register model
    AutoModelForSeq2SeqLM.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)