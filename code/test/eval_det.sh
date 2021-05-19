# Copyright 2021 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
# trained_model " PATH OF THE MODEL"
# image_root ="ROOT PATH OF IMAGES"
# image_list = "DETECTIONI IMAGES LIST"
DATASET=../../data
WEIGHTS=../../float/pytorch_multi-task_NAS_resnet18_512*320.pth
IMG_LIST=det_val.txt
#IMG_LIST=demo.txt
GT_FILE=${DATASET}/det_gt.txt
SAVE_FOLDER=./result/
DT_FILE=${SAVE_FOLDER}/det_test_all.txt
TEST_LOG=${DATASET}/det_log.txt


python eval.py --trained_model ${WEIGHTS}  --image_root ${DATASET} --image_list ${IMG_LIST}
cat ./result/det/* > ${DT_FILE}
python ./evaluation/evaluate_det.py -gt_file ${GT_FILE} -result_file ${DT_FILE} | tee -a ${TEST_LOG}
echo "Test report is saved to ${TEST_LOG}"

