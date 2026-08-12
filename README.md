<!-- HRITHIK_PROJECT_START -->

<div align="center">

# 🚀 UniSRec — Research Extension
### Cross-Domain Sequential Recommendation on Amazon Reviews 2023

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

**Research Extension by Hrithik Sharma**

</div>

---

## 👨‍💻 My Contribution

This repository contains my experimental extension of the original **UniSRec (KDD 2022)** project, focused on sequential recommendation under highly sparse Amazon Reviews 2023 domains.

- Fine-tuned UniSRec on **Magazine Subscriptions** and **Gift Cards**
- Built preprocessing and **k-core filtering** workflows
- Generated **PLM-based textual item embeddings**
- Evaluated performance under approximately **95–98% sparsity**
- Analyzed **Hit@10, NDCG@K, Recall@K and Precision@K**
- Extended experiments with **DuoRec** for contrastive sequential recommendation

## 🏆 Key Results

| Model | Dataset | Main Result |
|---|---|---:|
| **UniSRec** | Magazine Subscriptions | **Hit@10 = 0.2789, NDCG@10 = 0.1302** |
| **UniSRec** | Gift Cards | **Hit@10 = 0.2004, NDCG@10 = 0.0993** |
| **DuoRec** | Magazine Subscriptions | **NDCG@10 = 0.0299** |
| **DuoRec** | Gift Cards | **Recall@10 = 0.0027, NDCG@10 = 0.0016** |

## 🧠 End-to-End Pipeline

`Amazon Reviews 2023 → Preprocessing → K-Core Filtering → PLM Text Embeddings → Transformer Pretraining → Fine-Tuning → Ranking Evaluation`

## 🛠️ Tech Stack

`Python` • `PyTorch` • `Transformers` • `RecBole` • `CUDA` • `PLM Embeddings`

## 🔗 Attribution

This work is an **experimental research extension** based on the original UniSRec implementation.

**Official upstream repository:** https://github.com/RUCAIBox/UniSRec

**Original paper:** *Towards Universal Sequence Representation Learning for Recommender Systems* — KDD 2022

<!-- HRITHIK_PROJECT_END -->

---

# 📚 Original UniSRec Documentation

# UniSRec

This is the official PyTorch implementation for the [paper](https://arxiv.org/abs/2206.05941):
> Yupeng Hou*, Shanlei Mu*, Wayne Xin Zhao, Yaliang Li, Bolin Ding, Ji-Rong Wen. Towards Universal Sequence Representation Learning for Recommender Systems. KDD 2022.

---

*Updates*:

* [Nov. 22, 2022] We added scripts and implementations of baselines FDSA and S^3-Rec [[link]](https://github.com/RUCAIBox/UniSRec/issues/4#issuecomment-1316045022).
* [June 28, 2022] We updated some useful "mid product" files that can be obtained during the data preprocessing stage [[link]](dataset#useful-files), including:
  1. Clean item text (`*.text`);
  2. Index mapping between raw IDs and remapped IDs (`*.user2index`, `*.item2index`);
* [June 16, 2022] We released the code and scripts for  preprocessing ours datasets [[link]](dataset#dataset-preprocessing).

## Overview

We propose **UniSRec**, which stands for **Uni**versal **S**equence representation learning for **Rec**ommendation. Aiming to learn more generalizable sequence representations, UniSRec utilizes the associated description text of an item to learn transferable representations across different domains and platforms. For learning *universal item representations*, we design a lightweight architecture based on parametric whitening and mixture-of-experts enhanced adaptor. For learning *universal sequence representations*, we introduce two kinds of contrastive learning tasks by sampling multi-domain negatives. With the pre-trained universal sequence representation model, our approach can be effectively transferred to new cross-domain and cross-platform recommendation scenarios in a parameter-efficient way, under either inductive or transductive settings.

![](asset/model.png)

## Requirements

```
recbole>=1.1.1
python>=3.9.7
cudatoolkit>=11.3.1
pytorch>=1.11.0
```

## Download Datasets and Pre-trained Model

Please download the processed downstream (or pre-trained, if needed) datasets and the pre-trained model from [Google Drive](https://drive.google.com/drive/folders/1Uik0fMk4oquV_bS9lXTZuExAYbIDkEMW?usp=sharing) or [ç™¾åº¦ç½‘ç›˜](https://pan.baidu.com/s/1zdP3tEw9X6Ys5YNO5TyNEQ) (å¯†ç  3cml).

After unzipping, move `pretrain/` and `downstream/` to `dataset/`, and move `UniSRec-FHCKM-300.pth` to `saved/`.

## Quick Start

### Train and evaluate on downstream datasets

Fine-tune the pre-trained UniSRec model in transductive setting.

```
python finetune.py -d Scientific -p saved/UniSRec-FHCKM-300.pth
```

*You can replace `Scientific` to `Pantry`, `Instruments`, `Arts`, `Office` or `OR` to reproduce the results reported in our paper.*

Fine-tune the pre-trained model in inductive setting.

```
python finetune.py -d Scientific -p saved/UniSRec-FHCKM-300.pth --train_stage=inductive_ft
```

Train UniSRec from scratch (w/o pre-training).

```
python finetune.py -d Scientific
```

Run baseline SASRec.

```
python run_baseline.py -m SASRec -d Scientific --config_files=props/finetune.yaml --hidden_size=300
```

Please refer to [[link]](https://github.com/RUCAIBox/UniSRec/issues/4#issuecomment-1316045022) for more scripts of our baselines.

### Pre-train from scratch

Pre-train on one single GPU.

```
python pretrain.py
```

Pre-train with distributed data parallel on GPU:0-3.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python ddp_pretrain.py
```

### Customized Datasets

Please refer to [[link]](dataset#dataset-preprocessing) for details of data preprocessing. Then you can correspondingly try your customized datasets.

### Acknowledgement

The implementation is based on the open-source recommendation library [RecBole](https://github.com/RUCAIBox/RecBole).

Please cite the following papers as the references if you use our codes or the processed datasets.

```bibtex
@inproceedings{hou2022unisrec,
  author = {Yupeng Hou and Shanlei Mu and Wayne Xin Zhao and Yaliang Li and Bolin Ding and Ji-Rong Wen},
  title = {Towards Universal Sequence Representation Learning for Recommender Systems},
  booktitle = {{KDD}},
  year = {2022}
}


@inproceedings{zhao2021recbole,
  title={Recbole: Towards a unified, comprehensive and efficient framework for recommendation algorithms},
  author={Wayne Xin Zhao and Shanlei Mu and Yupeng Hou and Zihan Lin and Kaiyuan Li and Yushuo Chen and Yujie Lu and Hui Wang and Changxin Tian and Xingyu Pan and Yingqian Min and Zhichao Feng and Xinyan Fan and Xu Chen and Pengfei Wang and Wendi Ji and Yaliang Li and Xiaoling Wang and Ji-Rong Wen},
  booktitle={{CIKM}},
  year={2021}
}
```

Special thanks [@Juyong Jiang](https://github.com/juyongjiang) for the excellent DDP implementation ([#961](https://github.com/RUCAIBox/RecBole/pull/961)).
