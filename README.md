# Enhanced Visible-Infrared Pedestrian Re-ID 
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10.1-red)](https://pytorch.org/) 
[![License](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/licenses/MIT)


🔥 Official PyTorch implementation of our feature decoupling method for VI-ReID

## Environment
```bash
conda create -n fdm python=3.8 -y && conda activate fdm
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install numpy==1.21.6 Pillow==9.0.1 matplotlib==3.5.3
```

# Data Preparation 

wget https://github.com/wuancong/SYSU-MM01/archive/master.zip

unzip master.zip && python tools/preprocess_sysu.py --src SYSU-MM01-master --dst data/sysu

# Generate Body Contours (CNIF required) 

python tools/generate_contour.py --input data/sysu --output data/contours --model_path CNIF/checkpoints/cnif.pth

## Training

# SYSU-MM01 
```python
python train.py --dataset sysu --data_path data/sysu --contour_path data/sysu_contour
--lr 0.1 --batch_size 64 --gpu 0
```
# RegDB
```python
python train.py --dataset sysu --data_path data/sysu --contour_path data/contours \
--lr 0.1 --batch_size 64 --gpu 0 --erasing_p 0.5 --ortho_weight 0.5
```

## Testing
# SYSU-MM01 All Search
```python
python test.py --dataset sysu --resume models/sysu_fdm.pth --mode all
```
# RegDB Visible→Infrared
```python
python test.py --dataset regdb --resume models/regdb_fdm.pth --mode vi
```
## Core Modules

# Information Reduction (CIR)
```python
class ChannelAdap(object):
    def __call__(self, img):
        if random.random() < self.p:
            channel_idx = random.choice([0, 1, 2])
            return torch.stack([img[channel_idx], img[channel_idx], img[channel_idx]])
        return img
```
# Feature Decoupling (FD)
```
class FDModule(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.P = nn.Parameter(torch.randn(feat_dim, feat_dim))
        
    def forward(self, x):
        P = F.normalize(self.P, p=2, dim=1)
        body_dep = x @ P.t()
        body_ind = x - body_dep @ P
        return body_dep, body_ind
```
## Results

| Dataset    | Mode         | Rank-1 | mAP  | 
|-----------|--------------|--------|------|
| SYSU-MM01 | All Search   | 69.21% | 66.35% | 
| RegDB     | V→I          | 89.74% | 83.69% | 

## References
Feng, J., Wu, A., & Zheng, W.-S. (2023). Shape-erased feature learning for visible-infrared person re-identification. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 22752–22761.

Li, D., Wei, X., Hong, X., & Gong, Y. (2020). Infrared-visible cross-modal person re-identification with an x modality. Proceedings of the AAAI Conference on Artificial Intelligence, 34(4), 4610–4617.

Wang, G., Zhang, T., Cheng, J., Liu, S., Yang, Y., & Hou, Z. (2019). RGB-infrared cross-modality person re-identification via joint pixel and feature alignment. Proceedings of the IEEE/CVF International Conference on Computer Vision, 3623–3632.

Ye, M., Shen, J., J. Crandall, D., Shao, L., & Luo, J. (2020). Dynamic dual-attentive aggregation learning for visible-infrared person re-identification. Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XVII 16, 229–247.

# Acknowledgements
- Contour generation powered by [CNIF](https://github.com/CNIF-Project/CNIF) (request access via zhangyk@stu.xmu.edu.cn)
- Evaluation protocol follows [SYSU-MM01](https://github.com/wuancong/SYSU-MM01) official benchmark
- Baseline ResNet implementation from [torchvision](https://pytorch.org/vision/stable/models.html)

**Code Repository**: [lihan689/FDM/tree/master](https://github.com/lihan689/FDM/tree/master)  
**Technical Support**: 934271466@qq.com
