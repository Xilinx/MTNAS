# MTNAS: Search Multi-Task Networks for Autonomous Driving (ACCV 2020)

This repository contains the code of "[MTNAS: Search Multi-Task Networks for Autonomous Driving](https://openaccess.thecvf.com/content/ACCV2020/papers/Liu_MTNAS_Search_Multi-Task_Networks_for_Autonomous_Driving_ACCV_2020_paper.pdf)", which is accepted in Asian Conference on Computer Vision (ACCV), 2020.

## Requirements

1. Use Anaconda create a python  environment

   ```shell
   conda create -n test python=3.6
   ```

2. Activate the environment and install dependencies

   ```shell
   source activate test
   conda install pytorch
   conda install torchvision
   conda install -c menpo opencv3
   ```

3. Quick installation 

   ```shell
   conda env create /code/conda_config/environment.yaml
   ```

## Preparation

1. Evaluation dataset directory structure:

   ```markdown
   + data
     + images
        + images_id1.jpg
        + iamges_id2.jpg
     + seg_label
        + images_id1.png
        + iamges_id2.png
     + det_gt.txt
     + det_val.txt
     + seg_val.txt
     + demo.txt
     + det_log.txt
     + seg_log.txt
     
     
     images: images for detection and segmentation evaluation
     seg_label: segmentation ground truth
     det_gt.txt: detectioin ground truth
        image_name label_1 xmin1 ymin1 xmax1 ymax1
     image_name label_2 xmin2 ymin2 xmax2 ymax2
     det_val.txt:
        images id for detection evaluation
     seg_val.txt:
        images id for segmentation evaluation
     demo.txt:
     	 images id for demo visualization
     det_log.txt：
        save detection evaluation results
     seg_log.txt:
        save segmentation evaluation results
   ```

## Eval

1. Evaluate detection performance

   ```shell
   cd ./code/test/
   bash ./eval_det.sh
   #the results will be saved at /data/det_log.txt
   ```

2. Evaluate segmentation performance

   ```shell
   ./code/test/
   bash ./eval_seg.sh
   #the results will be saved at /data/seg_log.txt
   ```

3. Demo

   ```shell
   ./code/test/
   bash ./run_demo.sh
   #the demo pics will be saved at /code/test/result/demo
   ```

4. Performance

   ```markdown
   Detection test images: bdd100+Waymo val 10000
   Segmentation test images: bdd100+CityScapes val 1500
   Model: MT-NAS
   Classes-detection: 4
   Classes-segmentation: 16
   mAP: 43.67% 
   mIou: 46.15%
   ```

## Citation

If you find the code and pre-trained model useful in your research, please cite our paper:
```
@InProceedings{Liu_2020_ACCV,
    author    = {Liu, Hao and Li, Dong and Peng, JinZhang and Zhao, Qingjie and Tian, Lu and Shan, Yi},
    title     = {MTNAS: Search Multi-Task Networks for Autonomous Driving},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    year      = {2020}
}
```

