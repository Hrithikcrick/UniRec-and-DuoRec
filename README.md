<!-- HRITHIK_PROJECT_START -->

<div align="center">

# UniSRec & DuoRec — Sequential Recommendation on Sparse Amazon Reviews 2023 Domains

**Magazine Subscriptions & Gift Cards | PyTorch · RecBole · Transformer-based Sequential Recommendation**

Research reproduction, domain adaptation, training and evaluation experiments

</div>

## Project overview

This project studies two established sequential recommendation models on **Amazon Reviews 2023** categories with sparse user–item interactions:

- **UniSRec (KDD 2022)** — *Towards Universal Sequence Representation Learning for Recommender Systems*. Uses pretrained-language-model (PLM) item text representations and a Transformer sequence encoder to support transfer across recommendation domains. The project milestone sometimes calls this model **UniRec**; its official paper and implementation name is **UniSRec**.
- **DuoRec (WSDM 2022)** — *Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation*. Uses Transformer sequence modeling and contrastive regularization to improve the distribution and discrimination of learned representations.

The work is an **experimental reproduction and domain-specific evaluation** of existing architectures, not a claim to have invented either model.

## My work

- Prepared **Magazine Subscriptions** and **Gift Cards** interaction data from Amazon Reviews 2023 for RecBole-based sequential recommendation experiments.
- Performed **user/item k-core filtering**, item/interaction processing, chronological sequence preparation and training/validation/test generation in the experiment workflow.
- Generated **768-dimensional PLM-based textual item embeddings** for the UniSRec experiments to supplement sparse collaborative signals.
- Ran UniSRec pretraining/fine-tuning experiments with a Transformer sequence encoder and next-item ranking evaluation.
- Trained and evaluated **DuoRec** as a contrastive sequential-recommendation comparison on both domains.
- Analyzed learning curves and reported **Hit@10, Recall@10, NDCG@10**, and other ranking metrics where available.

## Results reported in my CV and project milestone

| Model | Amazon Reviews 2023 subset | Hit@10 | Recall@10 | NDCG@10 |
|---|---|---:|---:|---:|
| **UniSRec** | Magazine Subscriptions | **0.2789** | — | **0.1302** |
| **UniSRec** | Gift Cards | **0.2004** | — | **0.0993** |
| **DuoRec** | Magazine Subscriptions | — | **0.0362** | **0.0299** |
| **DuoRec** | Gift Cards | — | **0.0027** | **0.0016** |

These values match the **Sequential Recommendation System** entry in my technical CV and the summarized results in my November 2025 project milestone. A dash means that metric is **not reported here**, not a zero score. These are **reported experimental results**, not newly reproduced measurements from the files in this repository.

**Interpretation:** The experiments explored the behavior of text-enhanced sequential representations (UniSRec) and contrastive regularization (DuoRec) under sparse-domain conditions. **The runs did not all share the same preprocessing or evaluation configuration**, so the numbers should not be treated as a controlled, apples-to-apples model ranking.

## Experimental pipeline

1. **Data preparation:** Amazon Reviews 2023 interactions and item metadata → cleaning and user/item filtering → sequential train/validation/test files.
2. **Text features (UniSRec):** Generate 768-D item representations using pretrained language-model text embeddings.
3. **Sequence learning:** Use multi-head Transformer attention to model historical item interactions; pretrain/fine-tune UniSRec and separately train/evaluate DuoRec.
4. **Evaluation:** Rank next-item candidates and examine top-K ranking metrics and convergence.

### Settings documented in the project milestone

| Experiment | Setup / observation |
|---|---|
| UniSRec (both domains) | PLM text features: **768 dimensions**; Transformer encoder with **2 layers**, **2 attention heads**, and **300-dimensional hidden representations** in the documented configuration |
| UniSRec — Magazine | Fine-tuning logs shown through approximately **43 epochs**; reported Hit@10 **0.2789**, NDCG@10 **0.1302** |
| UniSRec — Gift Cards | Best checkpoint reported around **epoch 9**; Hit@10 **0.2004**, NDCG@10 **0.0993** |
| DuoRec — Magazine | **50-epoch** experiment; Adam **1e-3**, batch size **256**, hidden size **64**, 2 layers/2 heads, dropout **0.5**; milestone narrative identifies epoch **12** as a selected checkpoint |
| DuoRec — Gift Cards | **25-epoch** observed run; Adam **1e-3**, batch size **2048**, hidden size **64**, 2 layers/2 heads, dropout **0.5**; best reported around epoch **13** |
| Environment (DuoRec) | PyTorch **2.5.1**, CUDA **12.1**, NVIDIA **GeForce RTX 3050**, RecBole-compatible workflow |

The milestone describes experiments with high interaction sparsity. Exact user/item/interaction counts are intentionally not repeated here because some figures in its narrative are internally inconsistent and require checking against the original processed data.

**Result reconciliation note:** The DuoRec Magazine milestone narrative and CV report Recall@10 **0.0362**, while one per-epoch table lists **0.0435** at epoch 12. This README follows the CV/narrative figure; checkpoint-level numbers should be verified against the original logs before making a stronger claim.

## Code in this repository vs. local experiment files

**Included here (UniSRec foundation):**

- [`unisrec.py`](unisrec.py) — UniSRec model implementation from the attributed upstream project.
- [`pretrain.py`](pretrain.py), [`ddp_pretrain.py`](ddp_pretrain.py), [`finetune.py`](finetune.py) — training and fine-tuning entry points.
- [`dataset/preprocessing/process_amazon.py`](dataset/preprocessing/process_amazon.py) — upstream Amazon preprocessing utility; see its own dataset instructions for supported inputs.
- [`props/`](props/), [`dataset/downstream/`](dataset/downstream/), [`saved/`](saved/) — configuration files, existing downstream datasets and saved checkpoints.

**Not included in the public repository:** the experiment-specific **Amazon Reviews 2023 Magazine/Gift Cards processing and result logs**, as well as the **DuoRec experiment implementation and configuration**, which were used in the locally documented experiments. Consequently, the headline 2023 experimental results **cannot be fully reproduced from this public repository alone**. The original project milestone includes descriptions and screenshots of the local runs.

### Existing upstream quick start

The original UniSRec instructions below demonstrate how to run the **included upstream datasets** (such as `Scientific`), not how to recreate the local Amazon Reviews 2023 experiments:

```bash
python finetune.py -d Scientific -p saved/UniSRec-FHCKM-300.pth
```

## Research papers, implementation credit & data

- **UniSRec:** Yupeng Hou et al., *Towards Universal Sequence Representation Learning for Recommender Systems*, **KDD 2022** — [paper](https://arxiv.org/abs/2206.05941) · [official code](https://github.com/RUCAIBox/UniSRec).
- **DuoRec:** Ruihong Qiu et al., *Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation*, **WSDM 2022** — [paper](https://arxiv.org/abs/2110.05730) · [official code](https://github.com/RuihongQiu/DuoRec).
- **Data:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/).
- **Framework:** [RecBole](https://github.com/RUCAIBox/RecBole).

The repository's foundation is based on the original **UniSRec** open-source implementation. My documented contribution is adapting the experimental workflow to the two Amazon Reviews 2023 domains, running the training/evaluation studies, comparing reported behavior and analyzing results. Original authors retain credit for their architectures and upstream code.

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
