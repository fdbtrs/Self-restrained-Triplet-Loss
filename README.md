# Self-restrained Triplet Loss

## This is the official repository of the paper 'Self-restrained Triplet Loss for Accurate Masked Face Recognition'

#### Update
Paper has been accepted in Pattern Recognition (Elsevier) journal
[SRT](https://www.sciencedirect.com/science/article/pii/S003132032100649X?dgcid=coauthor)

![Workflow](https://raw.githubusercontent.com/fdbtrs/Self-restrained-Triplet-Loss/master/images/workflow.png)



### Model training
- All pretrained models are available under https://github.com/fdbtrs/Self-restrained-Triplet-Loss/tree/master/weights
- The SRT solution is trained on top of ResNet-100, ResNet-50 and MobileFaceNet.
- To train a model, please use the main.py

### Model evaluation
-  Evaluation on IJB-C: please check evaluation instructions under [evaluation folder](https://github.com/fdbtrs/Self-restrained-Triplet-Loss/tree/master/evaluation/ijbc)
-  Evaluation on LFW: please check evaluation instructions under [evaluation folder](https://github.com/fdbtrs/Self-restrained-Triplet-Loss/tree/master/evaluation/lfw)

##Citation

If you used any of the codes provided in this repository, please cite the following paper
```
@article{BOUTROS2022108473,
title = {Self-restrained triplet loss for accurate masked face recognition},
journal = {Pattern Recognition},
volume = {124},
pages = {108473},
year = {2022},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2021.108473},
url = {https://www.sciencedirect.com/science/article/pii/S003132032100649X},
author = {Fadi Boutros and Naser Damer and Florian Kirchbuchner and Arjan Kuijper},
keywords = {COVID-19, Biometric recognition, Identity verification, Masked face recognition},
abstract = {Using the face as a biometric identity trait is motivated by the contactless nature of the capture process and the high accuracy of the recognition algorithms. After the current COVID-19 pandemic, wearing a face mask has been imposed in public places to keep the pandemic under control. However, face occlusion due to wearing a mask presents an emerging challenge for face recognition systems. In this paper, we present a solution to improve masked face recognition performance. Specifically, we propose the Embedding Unmasking Model (EUM) operated on top of existing face recognition models. We also propose a novel loss function, the Self-restrained Triplet (SRT), which enabled the EUM to produce embeddings similar to these of unmasked faces of the same identities. The achieved evaluation results on three face recognition models, two real masked datasets, and two synthetically generated masked face datasets proved that our proposed approach significantly improves the performance in most experimental settings.}
}
```

## License

```
This project is licensed under the terms of the Attribution-NonCommercial-ShareAlike 4.0 
International (CC BY-NC-SA 4.0) license. 
Copyright (c) 2021 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt
```


