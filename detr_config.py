from typing import Mapping, OrderedDict
from transformers.onnx import OnnxSeq2SeqConfigWithPast
from transformers import AutoConfig
from pathlib import Path
from transformers.onnx import export
from transformers import AutoTokenizer, AutoModel, DetrFeatureExtractor


class DetrOnnxConfig(OnnxSeq2SeqConfigWithPast):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}),
                ("pixel_mask", {0: "batch_size", 1: "height", 2: "width"}),
                # ("decoder_attention_mask", {0: "batch_size", 1: "num_queries"}),
                # ("encoder_outputs", {0: "last_hidden_state", 1: "last_hidden_states"}),
                # ("inputs_embeds", {0: "batch", 1: "height", 2: "width"}),
                # ("decoder_inputs_embeds", {0: "batch", 1: "height", 2: "width"}),
                # ("output_attentions", {0: "batch", 1: "height", 2: "width"}),
                # ("output_hidden_states", {0: "batch", 1: "height", 2: "width"}),
                # ("return_dict", {0: "batch", 1: "height", 2: "width"}),
            ]
        )

config = AutoConfig.from_pretrained("facebook/detr-resnet-50")
onnx_config = DetrOnnxConfig(config)

onnx_path = Path("model.onnx")
model_ckpt = "facebook/detr-resnet-50"
base_model = AutoModel.from_pretrained("facebook/detr-resnet-50")
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

onnx_inputs, onnx_outputs = export(
    feature_extractor, 
    base_model, 
    onnx_config, 
    onnx_config.default_onnx_opset, 
    onnx_path
)
