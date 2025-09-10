import torch
import torchvision.models as models
import torch.nn as nn

resnet = models.resnet50(pretrained=True)
model = nn.Sequential(*list(resnet.children())[:-1])
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "resnet50.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
