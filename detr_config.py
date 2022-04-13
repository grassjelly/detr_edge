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
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
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
