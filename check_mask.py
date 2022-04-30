import torch
import models.imagenet

checkpoint = torch.load('checkpoints/imagenet/res18/model_best.pth.tar')
mask = checkpoint['weight_mask']

for i in range(len(mask)):
    print("Sparsity: ", mask[i].sum() / mask[i].numel())

print(mask[0].view(64, -1)[0])