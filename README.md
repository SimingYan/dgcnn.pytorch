# DGCNN.pytorch

This repo contains a PyTorch implementation for **Dynamic Graph CNN for Learning on Point Clouds (DGCNN)**(https://arxiv.org/pdf/1801.07829) and the fine-tuning code for paper [IAE](https://arxiv.org/pdf/2201.00785.pdf).

## Requirements
- Python 3.7
- PyTorch 1.2
- CUDA 10.0
- Package: glob, h5py, sklearn, plyfile

## Point Cloud Semantic Segmentation

You have to download `Stanford3dDataset_v1.2_Aligned_Version.zip` manually from https://goo.gl/forms/4SoGp4KtH1jfRqEj2 and place it under `data/`

### Run the training script:

This task uses 6-fold training, such that 6 models are trained leaving 1 of 6 areas as the testing area for each model. 

- Train in area 1-5

``` 
python main_semseg.py --exp_name=semseg_6 --test_area=6 
```

- Fine-tune with our pre-trained modeli in area 1-5
```
python main_semseg.py --exp_name=semseg_6 --test_area=6 --batch_size=16 --restore_path=/path/to/pretrained/model
```

### Run the evaluation script with pretrained models:

- Evaluate in area 6

``` 
python main_semseg.py --exp_name=semseg_eval_6 --test_area=6 --eval=True --model_root=pretrained/semseg/
```

- Evaluate in all areas

``` 
python main_semseg.py --exp_name=semseg_eval --test_area=all --eval=True --model_root=pretrained/semseg/
```

