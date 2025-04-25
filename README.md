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

git clone https://github.com/CNIF-Project/CNIF.git  # Access permission required

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
```bibtex
@inproceedings{58,
  title={Shape-erased feature learning for visible-infrared person re-identification},
  author={Feng, Jiawei and Wu, Ancong and Zheng, Wei-Shi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={22752--22761},
  year={2023}
}
@inproceedings{59,
  title={RGB-infrared cross-modality person re-identification via joint pixel and feature alignment},
  author={Wang, Guan'an and Zhang, Tianzhu and Cheng, Jian and Liu, Si and Yang, Yang and Hou, Zengguang},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={3623--3632},
  year={2019}
}
@inproceedings{60,
  title={Infrared-visible cross-modal person re-identification with an x modality},
  author={Li, Diangang and Wei, Xing and Hong, Xiaopeng and Gong, Yihong},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={34},
  number={04},
  pages={4610--4617},
  year={2020}
}
@inproceedings{61,
  title={Dynamic dual-attentive aggregation learning for visible-infrared person re-identification},
  author={Ye, Mang and Shen, Jianbing and J. Crandall, David and Shao, Ling and Luo, Jiebo},
  booktitle={Computer Vision--ECCV 2020: 16th European Conference, Glasgow, UK, August 23--28, 2020, Proceedings, Part XVII 16},
  pages={229--247},
  year={2020},
  organization={Springer}
}
```
# Acknowledgements
- Contour generation powered by [CNIF](https://github.com/CNIF-Project/CNIF) (request access via zhangyk@stu.xmu.edu.cn)
- Evaluation protocol follows [SYSU-MM01](https://github.com/wuancong/SYSU-MM01) official benchmark
- Baseline ResNet implementation from [torchvision](https://pytorch.org/vision/stable/models.html)

**Code Repository**: [lihan689/FDM](https://github.com/lihan689/FDM)  
**Technical Support**: 934271466@qq.com
