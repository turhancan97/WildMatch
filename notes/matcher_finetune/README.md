# lynx-finetuning

Finetuning feature matching modules for better performance on lynx-reidentification.

## Instalation

Clone RDD and change commit
```bash
git clone --recursive https://github.com/xtcpete/rdd
cd rdd
git checkout 539508b270095969f9934c574cf7026bf37c434c
cd .. # root directory
```
Download `RDD-v2.pth` and `RDD_lg-v2.pth` checkpoints to rdd/weights

Install packages
```bash
conda create -n lynx-finetuning python=3.12
conda activate lynx-finetuning

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install -r rdd/requirements.txt
pip install -r contrastive_finetuning/requirements.txt
```