<!-- HRITHIK_PROJECT_START -->

<div align="center">

# UniSRec & DuoRec on Amazon Reviews 2023
### Sequential Recommendation on Magazine Subscriptions and Gift Cards

**PyTorch · RecBole · Transformer-based Sequential Recommendation**

</div>

---

## Overview

In this project, I worked on **sequential recommendation** using **UniSRec** and **DuoRec** on two Amazon Reviews 2023 categories:

- **Magazine Subscriptions**
- **Gift Cards**

I focused on:
- preprocessing sparse recommendation data
- preparing sequential train / validation / test files
- generating PLM-based item features for UniSRec
- training and fine-tuning transformer-based recommendation models
- evaluating recommendation quality using top-K ranking metrics

---

## Models

### UniSRec
I used **UniSRec** for sequential recommendation with **PLM-based textual item embeddings** and a **Transformer encoder**.

### DuoRec
I used **DuoRec** for sequential recommendation with a **Transformer encoder** and **contrastive regularization**.

---

## UniSRec — Dataset Statistics

| Dataset | Users | Items | Interactions |
|---|---:|---:|---:|
| Magazine Subscriptions | 6,182 | 157 | 6,328 |
| Gift Cards | 455–456 | 147 | ~2,950 |

---

## UniSRec — Configuration

| Component | Value |
|---|---|
| PLM embedding size | 768 |
| Transformer layers | 2 |
| Attention heads | 2 |
| Hidden size | 300 |
| Loss | Cross-entropy |

---

## UniSRec — Final Results

| Dataset | Hit@10 | NDCG@10 |
|---|---:|---:|
| Magazine Subscriptions | **0.2789** | **0.1302** |
| Gift Cards | **0.2004** | **0.0993** |

### UniSRec — Result Graph

```mermaid
xychart-beta
    title "UniSRec Final Results"
    x-axis ["Magazine", "Gift Cards"]
    y-axis "Score" 0 --> 0.30
    bar "Hit@10" [0.2789, 0.2004]
    bar "NDCG@10" [0.1302, 0.0993]
```

---

## UniSRec — Training Summary

### Magazine Subscriptions
- Training loss decreased from **75.2** to **60.5**
- Fine-tuning logs were shown through **43 epochs**
- Final results:
  - **Hit@10 = 0.2789**
  - **NDCG@10 = 0.1302**

### Gift Cards
- Training loss decreased from **66.2** to **60.2**
- Best checkpoint was reported around **epoch 9**
- Best results:
  - **Hit@10 = 0.2004**
  - **NDCG@10 = 0.0993**
- Later evaluations (epochs **34–39**) reported:
  - **Hit@10 = 0.2004**
  - **NDCG@10 ≈ 0.0942**

| Dataset | Epoch information | Training-loss trend | Result |
|---|---|---|---|
| Magazine Subscriptions | 43 epochs shown | 75.2 → 60.5 | Hit@10 = 0.2789, NDCG@10 = 0.1302 |
| Gift Cards | Best around epoch 9 | 66.2 → 60.2 | Hit@10 = 0.2004, NDCG@10 = 0.0993 |

---

## DuoRec — Dataset Statistics

| Dataset | Users | Items | Interactions |
|---|---:|---:|---:|
| Magazine Subscriptions | 139 | 327 | 22,366 |
| Gift Cards | ~450–460 | ~145–150 | ~2,900 |

---

## DuoRec — Configuration

### Magazine Subscriptions

| Component | Value |
|---|---|
| Epochs | 50 |
| Batch size | 256 |
| Hidden size | 64 |
| Inner size | 256 |
| Layers | 2 |
| Heads | 2 |
| Dropout | 0.5 |
| Optimizer | Adam |
| Learning rate | 0.001 |

### Gift Cards

| Component | Value |
|---|---|
| Epochs | 25 |
| Batch size | 2048 |
| Hidden size | 64 |
| Inner size | 256 |
| Layers | 2 |
| Heads | 2 |
| Dropout | 0.5 |
| Optimizer | Adam |
| Learning rate | 0.001 |

### Environment

| Component | Value |
|---|---|
| Framework | PyTorch + RecBole |
| CUDA | 12.1 |
| GPU | NVIDIA GeForce RTX 3050 |

---

## DuoRec — Final Results

| Dataset | Recall@10 | NDCG@10 |
|---|---:|---:|
| Magazine Subscriptions | **0.0362** | **0.0299** |
| Gift Cards | **0.0027** | **0.0016** |

### DuoRec — Result Graph

```mermaid
xychart-beta
    title "DuoRec Final Results"
    x-axis ["Magazine", "Gift Cards"]
    y-axis "Score" 0 --> 0.04
    bar "Recall@10" [0.0362, 0.0027]
    bar "NDCG@10" [0.0299, 0.0016]
```

---

## DuoRec — Magazine Subscriptions Results

| Epoch | Train Loss | Recall@10 | NDCG@10 | Precision@10 |
|---:|---:|---:|---:|---:|
| 5 | 52.29 | 0.0362 | 0.0209 | 0.0019 |
| 9 | 51.41 | 0.0362 | 0.0220 | 0.0022 |
| 12 | 51.09 | 0.0435 | 0.0299 | 0.0029 |

### DuoRec — Magazine Subscriptions Graph

```mermaid
xychart-beta
    title "DuoRec Magazine Subscriptions"
    x-axis [5, 9, 12]
    y-axis "Score" 0 --> 0.05
    line "Recall@10" [0.0362, 0.0362, 0.0435]
    line "NDCG@10" [0.0209, 0.0220, 0.0299]
```

---

## DuoRec — Gift Cards Results

| Epoch | Train Loss | Recall@10 | NDCG@10 |
|---:|---:|---:|---:|
| 5 | 754.73 | 0.0013 | 0.0004 |
| 10 | 559.77 | 0.0012 | 0.0012 |
| 13 | 450.37 | 0.0027 | 0.0016 |
| 20 | 254.69 | 0.0018 | 0.0010 |
| 24 | 207.57 | 0.0018 | 0.0010 |

### DuoRec — Gift Cards Graph

```mermaid
xychart-beta
    title "DuoRec Gift Cards"
    x-axis [5, 10, 13, 20, 24]
    y-axis "Score" 0 --> 0.003
    line "Recall@10" [0.0013, 0.0012, 0.0027, 0.0018, 0.0018]
    line "NDCG@10" [0.0004, 0.0012, 0.0016, 0.0010, 0.0010]
```

---

## Key Results

- **UniSRec — Magazine Subscriptions:** Hit@10 = **0.2789**, NDCG@10 = **0.1302**
- **UniSRec — Gift Cards:** Hit@10 = **0.2004**, NDCG@10 = **0.0993**
- **DuoRec — Magazine Subscriptions:** Recall@10 = **0.0362**, NDCG@10 = **0.0299**
- **DuoRec — Gift Cards:** Recall@10 = **0.0027**, NDCG@10 = **0.0016**

---

## References

- **UniSRec:** Yupeng Hou et al., *Towards Universal Sequence Representation Learning for Recommender Systems*, **KDD 2022**  
  Paper: https://arxiv.org/abs/2206.05941  
  Code: https://github.com/RUCAIBox/UniSRec

- **DuoRec:** Ruihong Qiu et al., *Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation*, **WSDM 2022**  
  Paper: https://arxiv.org/abs/2110.05730  
  Code: https://github.com/RuihongQiu/DuoRec

- **Amazon Reviews 2023**  
  https://amazon-reviews-2023.github.io/

- **RecBole**  
  https://github.com/RUCAIBox/RecBole

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
