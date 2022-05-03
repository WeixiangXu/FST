# FST
- Paper: [Towards Fully Sparse Training](TBD), AAAI2022
## Install CUDA Extension
  - The extension implements the Im2col + GEMM based convolution.
  - Our CUDA extension are build with gcc version 5.3.1
  ```
  cd Sparse_Conv_Extension_CUDA
  sh install.sh
  ```
## Training scripts
```
python main_imagenet.py -a resnet18 --data path_to_ImageNet --checkpoint checkpoints/imagenet/res18
```

## Check Sparse Mask
The trained model should be 2:4 structured sparsed (requiring four consecutive values containing at least two zeros).
```
python check_mask.py
```

## Results on ResNet-18
|  Model              | Pattern | <img src="https://latex.codecogs.com/svg.image?\boldsymbol{\rm&space;Y}=\boldsymbol{\rm&space;W}\boldsymbol{\rm&space;X}" />   | <img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;\mathcal{L}}{\partial&space;\boldsymbol{\rm&space;X}}=&space;&space;&space;&space;\boldsymbol{\rm&space;W}^T&space;&space;&space;&space;\frac{\partial&space;\mathcal{L}}{\partial&space;\boldsymbol{\rm&space;Y}}" /> | <img src="https://latex.codecogs.com/svg.image?&space;&space;&space;&space;\frac{\partial&space;\mathcal{L}}{\partial&space;\boldsymbol{\rm&space;W}}=&space;&space;&space;&space;\frac{\partial&space;\mathcal{L}}{\partial&space;\boldsymbol{\rm&space;Y}}&space;&space;&space;&space;\boldsymbol{\rm&space;X}^T" />  | Accuracy | Download   |
| :------:            | :------: | :--------: |:--------: |:-------:  | :------: | :------: |
|  Float              | Dense    | -      | -  | -   | 71.2     |          |
|  ASP                | 2:4 structured    | <img src="https://latex.codecogs.com/svg.image?\boldsymbol{\rm&space;W}" /> | - | - | 70.7     |         |
|  Ours   |  2:4 structured   | <img src="https://latex.codecogs.com/svg.image?\boldsymbol{\rm&space;W}" />  | <img src="https://latex.codecogs.com/svg.image?\boldsymbol{\rm&space;W}^T" /> | <img src="https://latex.codecogs.com/svg.image?\boldsymbol{\rm&space;X}^T" />| 71.0 |[google cloud](https://drive.google.com/drive/folders/1a_BIsPYDKhvVKNAn6rE33FG3YqgAvUvE?usp=sharing) |

## Evaluate with 2:4 sparsed model
```
python main_imagenet_eva.py -a resnet18 --data path_to_ImageNet --resume checkpoints/imagenet/res18/model_best.pth.tar --evaluate
```
