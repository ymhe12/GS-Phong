# GS-Phong: Meta-Learned 3D Gaussians for Relightable Novel View Synthesis 

Yumeng He,  [Yunbo Wang](https://wyb15.github.io/)<sup>†</sup>, [Xiaokang Yang](https://scholar.google.com/citations?user=yDEavdMAAAAJ&hl=zh-CN)<br>

[arXiv](https://arxiv.org/abs/2405.20791) | [PDF](https://arxiv.org/pdf/2405.20791) | [Dataset](https://drive.google.com/drive/folders/1AoJctpU_sUyVev3luqXhE2-OjAauL8MN?usp=drive_link)

This repository contains the official code for our paper: **GS-Phong: Meta-Learned 3D Gaussians for Relightable Novel View Synthesis**.


## Installation

1. Create an environment
    ```bash
    conda create -n gs-phong python=3.10
    conda activate gs-phong
    ```
    
2. Install dependencies
    ```bash
    git clone https://github.com/ymhe123/GS-Phong.git
    cd GS-Phong
    git submodule update --init --recursive
    pip install -r requirements.txt
    pip install -e submodules/depth-diff-gaussian-rasterization
    pip install -e submodules/simple-knn
    pip install ./bvh
    ```


## Experiments

### Training

1. Stage 1&2: Gaussian initialization & Normal finetuning
    ```shell
    python train.py -s <path_to_your_dataset> -m <path_to_ouput_folder> --eval
    ```

2. Stage 3: Meta-learning
    ```shell
    python train_meta.py -s <path_to_your_dataset> -m <path_to_ouput_folder> --eval
    ```

### Evaluation
1. Generate NVS renderings
    ```shell
    python render.py -m <path_to_ouput_folder>
    ```

2. Calculate error metrics
   ```shell
    python metrics.py -m <path_to_ouput_folder>
    ```

### Full script
You can run the full experiment using: (remember ro edit the $DATADIR and $OUTPUT location)
```shell
sh run.sh
```



## Acknowledgements

We appreciate the following github repos where we borrow code from: 
* [gaussian splatting](https://github.com/graphdeco-inria/gaussian-splatting)
* [Relightable3DGaussian](https://github.com/NJU-3DV/Relightable3DGaussian)

Thanks for their amazing works!


## Citation

If you find our work helps, please cite our paper:

```bibtex

@article{he2024gs,
  title={GS-Phong: Meta-Learned 3D Gaussians for Relightable Novel View Synthesis},
  author={He, Yumeng and Wang, Yunbo and Yang, Xiaokang},
  journal={arXiv preprint arXiv:2405.20791},
  year={2024}
}

```
