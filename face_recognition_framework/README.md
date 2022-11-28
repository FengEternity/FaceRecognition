# Face recognition framework based on PyTorch.


### Introduction

This is a face recognition framework based on PyTorch with convenient training, evaluation and feature extraction functions. It is originally a multi-task face recognition framework for our accpeted ECCV 2018 paper, "Consensus-Driven Propagation in Massive Unlabeled Data for Face Recognition". However, it is also a common framework for face recognition. You can freely customize your experiments with your data and configurations with it.

### Paper

Xiaohang Zhan, Ziwei Liu, Junjie Yan, Dahua Lin, Chen Change Loy, ["Consensus-Driven Propagation in Massive Unlabeled Data for Face Recognition"](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xiaohang_Zhan_Consensus-Driven_Propagation_in_ECCV_2018_paper.pdf), ECCV 2018

Project Page:
[link](http://mmlab.ie.cuhk.edu.hk/projects/CDP/)

### Why multi-task?

Different datasets have different identity (category) sets. We do not know the intersection between them. Hence instead of merging identity sets of different datasets, regarding them as different tasks is an effective alternative way.

### Features

Framework: Multi-task, Single Task

Loss: Softmax Loss, ArcFace

Backbone CNN: ResNet, DenseNet, Inception, InceptionResNet, NASNet, VGG

Benchmarks: Megaface (FaceScrub), IJB-A, LFW, CFP-FF, CFP-FP, AgeDB-30, calfw, cplfw

Data aug: flip, scale, translation

Online testing and visualization with Tensorboard.

### Setup step by step

1. Clone the project.

      ```
      git clone git@github.com:XiaohangZhan/face_recognition_framework.git
      cd face_recognition_framework
      ```

2. Dependency.

    python=3.6, tensorboardX, pytorch=0.3.1, mxnet, sklearn

3. Data Preparation.

      Download datasets from [insightface](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo) into your data storage folder, e.g., `~/data/face_recognition/`. Taking CASIA-Webface for example:

      ```sh
      cd ~/data/face_recognition/
      unzip faces_CASIA_112x112.zip
      cd - # back to the repo root
      mkdir data
      python tools/convert_data.py -r ~/data/face_recognition/faces_webface_112x112 -o ~/data/face_recognition/faces_webface_112x112 # convert mxnet records into images
      ln -s ~/data/face_recognition/faces_webface_112x112 data/webface
      ```
      
      Optionally, if you want to test on MegaFace. Download testing set from [here](https://drive.google.com/open?id=1e47P-u5hS7QGHP56YcMq5DoOLl_nxlPo) into your data storage folder, e.g., `~/data/face_recognition/`. Then:
      
      ```sh
      cd ~/data/face_recognition/
      mkdir -p megaface_test/raw
      cd megaface_test/raw
      mv ../../megaface_testpack_v1.0.zip .
      unzip -q megaface_testpack_v1.0.zip
      cd $THIS_REPO # back to the repo root
      ln -s  ~/data/face_recognition/megaface_test data/megaface_test
      ```
      Next, download MegaFace lists from [here](https://drive.google.com/open?id=15ZmNT4AhRKClaHDpDxVw_fCKoli3woF8) into `~/data/face_recognition/megaface_test/`.
      Finally, the folder `data/megaface_test/` looks like:
      ```
      data
        ├── megaface_test
          ├── concat_list.txt
          ├── facescrub3530
          ├── megaface_distractor
          ├── raw
      ```

4. Training.

      ```
      sh experiments/webface/res50-bs64-sz224-ep35/train.sh
      ```

5. Monitoring.

      ```
      tensorboard --logdir experiments
      ```

6. Resume training.

      ```
      sh experiments/webface/res50-bs64-sz224-ep35/resume.sh 10 # e.g., resume from epoch 10
      ```

7. Evalution.

      ```
      sh experiments/webface/res50-bs64-sz224-ep35/evaluation.sh 35 # e.g., evaluate epoch 35
      ```

8. Feature extraction.

      Firstly, specify the `data_name`, `data_root` and `data_list` under `extract_info` in the config file. The `data_list` is a txt file containing an image relative filename in each line. Then execute:

      ```
      # e.g., extract features with epoch 35 model.
      # The feature file is stored in checkpoints/ckpt_epoch_35_[data_name].bin
      sh experiments/webface/res50-bs64-sz224-ep35/extract.sh 35 
      ```

### Baselines

* Trained using Webface

| arch      | LFW    | CFP-FF | CFP-FP | AgeDB-30 | calfw  | cplfw  |
| --------- | ------ | ------ | ------ | -------- | ------ | ------ |
| resnet-50 | 0.9850 | 0.9804 | 0.9117 | 0.8967   | 0.9013 | 0.8423 |

* Trained using MS1M

| arch             | LFW    | CFP-FF | CFP-FP | AgeDB-30 | calfw  | cplfw  | vgg2-FP | megaface |
| ---------------- | ------ | ------ | ------ | -------- | ------ | ------ | ------- | -------- |
| densenet-121     | 0.9948 | 0.9946 | 0.9594 | 0.9615   | 0.9500 | 0.9057 | 0.9418  | 0.8665   |
| densenet-121-arc | 0.9973 | 0.9979 | 0.9601 | 0.9728   | 0.9558 | 0.9063 | 0.9496  | 0.9287   |

Note that the hyper-parameters are not adjusted to optimal. Hence, they are not the state-of-the-art face recognition models.
You may download those pre-trained models [here](https://drive.google.com/open?id=1bJqhFBMkxqYsyIgWaEGOdoBkbmLBZ1zq).

### Bibtex

If you find this code useful in your research, please cite:
```
@inproceedings{zhan2018consensus,
  title={Consensus-Driven Propagation in Massive Unlabeled Data for Face Recognition},
  author={Zhan, Xiaohang and Liu, Ziwei and Yan, Junjie and Lin, Dahua and Change Loy, Chen},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={568--583},
  year={2018}
}
```

### TODO (Will carry out in a "Buddha-like" way)

1. Implement distributed training.
2. Adjust hyper-parameters.
3. Multi-task experiments.
