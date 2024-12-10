import torch
import torchvision.models as models
from IPython import embed
import time
import numpy as np
from vit_pytorch import ViT
from vit_pytorch.efficient import ViT as EfficientViT  # If you want to use the efficient version


def benchmark(model, input_shape=(1024, 1, 32, 32), dtype='fp32', nwarmup=50, nruns=1000):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype == 'fp16':
        input_data = input_data.half()

    # Warm up
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()

    # Timing
    timings = []
    with torch.no_grad():
        for i in range(1, nruns + 1):
            start_time = time.time()
            output = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()

            timings.append(end_time - start_time)
            if i % 100 == 0:
                print('Iteration %d/%d, avg batch time %.2f ms' % (i, nruns, np.mean(timings) * 1000))

    print("Input shape:", input_data.size())
    print("Output shape:", output.shape)
    print('Average batch time: %.2f ms' % (np.mean(timings) * 1000))

## Load pre-trained ResNet-50 model
#resnet50_model = models.resnet50(pretrained=True)
#resnet50_model.eval()
#resnet50_model.to("cuda")

# Load pre-trained ViT model (example for a specific variant, e.g., base size)
vit_model = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048,
)
vit_model.eval()
vit_model.to("cuda")

print('begin')
print(torch.__version__)
benchmark(vit_model, (128,3,256,256))