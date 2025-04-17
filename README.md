# KSCU
## Installation
```
git clone https://github.com/xxxx/KSCU.git
cd KSCU
pip install -r requirements.txt
```
## Training

After installation, follow these instructions to train a custom KSCU model:
```
python kscu_diffusers.py --erase_concept 'Van Gogh' --train_method 'xattn' --mode 'style' --use_augmentation true --version '1-4'  --save_path 'models/path'
``` 
train_method is a training mode, with options `'xattn'`,`'noxattn'`, `'selfattn'`, `'full'`.
use_augmentation represents whether to use Prompt Augmentation.
version is the SD version to be trained.
the input of mode is the task type.
The fine-tuned model parameters will be saved to 'models/path'.

## Visualization
You can use the following command to visualize the generation results of the fine-tuned model.
```
python my_generate-image.py --model_name 'model/path/xx.pt' --train_method 'xattn' --prompts_path 'data/xxx.csv' --save_path 'result' --version '1-4'  --finetuner true
``` 
