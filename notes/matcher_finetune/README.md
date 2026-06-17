# lynx-finetuning

Finetuning feature matching modules for better performance on lynx-reidentification.

## Instalation

Clone RDD and change commit
```bash
conda create -n lynx-finetuning

git clone --recursive https://github.com/xtcpete/rdd
cd rdd
git checkout 539508b270095969f9934c574cf7026bf37c434c
cd .. # root directory
```

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu132
pip install -r rdd/requirements.txt
pip install -r contrastive_finetuning/requirements.txt
```