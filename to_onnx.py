from transformers import DetrForObjectDetection
import torch 

model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

onnx_file_name = 'detr-resnet-50.onnx'
dummy_input = torch.ones(1, 3, 800, 1066, dtype=torch.float)
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_file_name, 
    verbose=True, 
    opset_version=11
)