# Self-restrained Triplet Loss

## This is the offical repository of the paper 'Unmasking Face Embeddings by Self-restrained Triplet Loss for Accurate Masked Face Recognition'


![Workflow](https://raw.githubusercontent.com/fdbtrs/Self-restrained-Triplet-Loss/master/images/workflow.png)



### Model training
- All pretrained models are available under https://github.com/fdbtrs/Self-restrained-Triplet-Loss/tree/master/weights
- The SRT solution is trained on top of ResNet-100, ResNet-50 and MobileFaceNet.
- To train a model, please use the main.py

### Model evaluation
-  Evaluation on IJB-C: please check evaluation instruction under [evaluation folder](https://github.com/fdbtrs/Self-restrained-Triplet-Loss/tree/master/evaluation/ijbc)
-  Evaluation on LFW: please check evaluation instruction under [evaluation folder](https://github.com/fdbtrs/Self-restrained-Triplet-Loss/tree/master/evaluation/lfw)

Citation

If you used any codes provided in this repository, please cite the following paper

@article{boutros2021unmasking,
  title={Unmasking Face Embeddings by Self-restrained Triplet Loss for Accurate Masked Face Recognition},
  author={Boutros, Fadi and Damer, Naser and Kirchbuchner, Florian and Kuijper, Arjan},
  journal={arXiv preprint arXiv:2103.01716},
  year={2021}
}
