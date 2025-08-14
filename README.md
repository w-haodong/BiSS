# BiSS-Net: Bi-Scope Contextual State-Space Learning for Automated Cobb Angle Measurement in Full-Body X-rays



https://github.com/user-attachments/assets/47ad8600-2037-4394-9462-381b672f5151



Abstract: Manual Cobb angle measurement for scoliosis assessment from X-rays is subjective and timeconsuming. While automated methods have advanced significantly, achieving consistently high accuracy and robustness in landmark localization, especially for challenging full-body radiographs containing diverse anatomical structures and potential occlusions, remains an area for improvement. This paper introduces BiSS-Net, a novel encoder-decoder architecture designed for precise vertebral landmark detection to automate Cobb angle calculation. BiSS-Net effectively integrates efficient global context modeling using State Space Model (SSM) with a novel Bi-Scope Contextual Module (BCM). The BCM captures both fine-grained local and broader regional spinal features, yielding robust representations that balance detail and structure. Tested on the standard AASCE 2019 benchmark and a challenging custom Full-Body dataset, BiSS-Net significantly outperforms stateof-the-art approaches across multiple accuracy metrics while maintaining computational efficiency. Ablation studies validate the contributions of both the integrated SSM blocks and the proposed BCM. Our results demonstrate the high potential of the BiSS-Net architecture, combining advanced SSM with targeted contextual modules, for robust landmark detection in complex medical imaging tasks.

![heatmap](https://github.com/user-attachments/assets/cea2a6be-412d-42aa-a9ce-e3a4c6e6f9b8)

#### Notice: This paper used the Mamba environment from VMamba (https://github.com/MzeroMiko/VMamba), please download the placed in: ./models/Mamba

#### Datasets: The dataset used in this paper is from AASCE 2019 (https://aasce19.grand-challenge.org/). The dataset used in this paper is from AASCE 2019 (https://aasce19.grand-challenge.org/). In addition, we also provide the validation set annotations (full-body images) from the AASCE dataset (file shared via online storage: AASCE-VAL, Link: https://pan.baidu.com/s/1QDRMPBfzJ-4htWCfL4cGKA?pwd=ir5b extraction code: ir5b)

#### In this Github repository we provide the test code along with the pre-trained parameters file (shared via an online storage: model_final.pth, Link: https://pan.baidu.com/s/1sYJ5JyUCpyv9P7LvUZWccw?pwd=wzw8 extraction code: wzw8). 
#### Additional training code and validation code will be uploaded after the paper is officially published.

