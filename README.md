# JNSKR-pytorch

This is PyTorch implementation of the paper:

*Chong Chen, Min Zhang, Weizhi Ma, Yiqun Liu and Shaoping Ma. 2020. [Jointly Non-Sampling Learning for Knowledge Graph Enhanced Recommendation.](https://chenchongthu.github.io/files/SIGIR_JNSKR.pdf) 
In SIGIR'20.*

You can find TensorFlow implementation by the **paper authors** [here](https://github.com/chenchongthu/JNSKR)

## Citation
**Please cite the following paper if you use their codes.**

```
@inproceedings{chen2020jointly,
  title={Jointly Non-Sampling Learning for Knowledge Graph Enhanced Recommendation},
  author={Chen, Chong and Zhang, Min and Ma, Weizhi and Liu, Yiqun and Ma, Shaoping},
  booktitle={Proceedings of SIGIR},
  year={2020},
}
```

**You also need to cite the KDD'19 paper if you use the datasets.**

```
@inproceedings{KGAT19,
  author    = {Xiang Wang and
               Xiangnan He and
               Yixin Cao and
               Meng Liu and
               Tat{-}Seng Chua},
  title     = {{KGAT:} Knowledge Graph Attention Network for Recommendation},
  booktitle = {{KDD}},
  pages     = {950--958},
  year      = {2019}
}
```

## Environment Requirements

The code has been tested under Python 3.8.5.

Required packages are as follows:

```
pytorch == 1.7.1
numpy == 1.19.2
pandas == 1.2.1
scikit-learn == 0.23.2
matplotlib == 3.3.2
```

## Run the codes

1. Edit [parser_jnskr.py](./models/utils/parser_jnskr.py)  to customize your training arguments.
2. ```python main.py```
3. Logs will be saved to ```./log/JNSKR/```

## Results On Test Set

### Amazon-Book

| Model         | Best Epoch | Recall@10 | Recall@20 | Recall@40 | NDCG@10 | NDCG@20 | NDCG@40 |
| ------------- | :--------: | :-------: | :-------: | :-------: | :-----: | :-----: | :-----: |
| JNSKR-pytorch |     29     |  0.1079   |  0.1572   |  0.2194   | 0.0938  | 0.1165  | 0.1412  |

Refer to [log77.log](./log/JNSKR/amazon-book/emb_size64_lr0.05/log77.log) for more details.

### Yelp 2018

| Model         | Best Epoch | Recall@10 | Recall@20 | Recall@40 | NDCG@10 | NDCG@20 | NDCG@40 |
| ------------- | :--------: | :-------: | :-------: | :-------: | :-----: | :-----: | :-----: |
| JNSKR-pytorch |     60     |  0.0453   |  0.0750   |  0.1209   | 0.0808  | 0.1081  | 0.1430  |

Refer to [log0.log](./log/JNSKR/yelp2018/emb_size64_lr0.05/log0.log) for more details.