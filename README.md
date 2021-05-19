# MTNAS

This code belongs to the "MTNAS: Search Multi-Task Networks for Autonomous Driving" [ACCV 2020]

![](./img/Picture1.png)
    We first search for the optimal branch architectures for each task separately and then search for the optimal backbone architecture under the overall guidance. 
    For the reason of search order, we explain that our goal is to search for task-specific branches and task-shared backbone. When optimizing the backbone, we need tocompute the loss from all the tasks. After obtaining the optimal branch architectures, backbone can benefit from each branch architecture and then learn shared knowledge across all tasks, leading to overall optimization for MTL.
    Then, in the stage of backbone search, we propose a simple but effective pre-searching procedure to search for an initialized backbone architecture under the guidance of auxiliary ImageNet classification task. After pre-searching, a well-initialized backbone model is obtained
    At last, branch architectures remain unchanged. Backbone can benefit from the optimal branches to learn shared knowledge from all tasks. The task-shared backbone will be generated after this backbone search stage.

#  **Requirements** 

1. Use  Anaconda create a python  environment 

   ```shell
   conda create -n test python=3.6
   ```

2. activate the environment and install dependencies . 

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

# **Preparation**

1. Evaluation Dataset Directory Structure like:

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

# **Eval**

 Evaluate at  ./code/test/

1. Evaluate Detection Performance

   ```shell
   bash ./eval_det.sh
   #the results will be saved at /data/det_log.txt
   ```

2. Evaluate Segmentation Performance

   ```shell
   bash ./eval_seg.sh
   #the results will be saved at /data/seg_log.txt
   ```

3. Demo

   ```shell
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
5. Visualizing on BDD100K dataset

![](./img/Picture2.png)

![](./img/Picture3.png)

![](./img/Picture4.png)
    The figure visualizes some examples of MTL results on the BDD100K dataset. In general, our MTNAS method can achieve more accurate detection and segmentation results compared to the hand-crafted MTL baseline.
    For example, MTNAS can better detect small objects (traffic sign) in the first row and crowd people in the second row. Besides, MTNAS can better segment the objects with a well-defined shape (car) and amorphous background regions ( sidewalk and sky) as shown in the last three rows

# **Model_info**

1. Data preprocess 

   ```markdown
   data channel order: BGR(0~255)                  
   resize: h * w = 320 * 512 (cv2.resize(image, (new_w, new_h)). astype(np.float32))
   mean: (104, 117, 123), input = input - mean
   
   ```

