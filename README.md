# Table-Recognition-LORE
### Attention that This repo will be active If your cuda version <= 11.4
### Set up environment
```
cd LORE-TSR/src
curl https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh --output anaconda.sh
bash anaconda.sh
export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
source ~/.bashrc
conda create --name Lore python=3.7
conda activate Lore
pip install -r requirements.txt
pip install torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
chmod +x  *.sh
cd lib/models/networsk/DCNv2
python3 setup.py build develop
```
### Inference
Download and unzip checkpoint [Here](https://drive.google.com/file/d/1n33c9jmGmjSfRbheleE1pqiIXBb_BCEw/view?usp=sharing)
Change folder path in --demo, --load_model and --load_processor in folder checkpoint
```
python demo.py ctdet \
        --dataset table \
        --demo ../input_images/custom_2 \
        --demo_name demo_wired \
        --debug 1 \
        --arch dla_34 \
        --K 3000 \
        --MK 5000 \
        --tsfm_layers 4 \
        --stacking_layers 4 \
        --gpus 0\
        --wiz_4ps \
        --wiz_detect \
        --wiz_rev \
        --wiz_stacking \
        --convert_onnx 1 \
        --vis_thresh_corner 0.3 \
        --vis_thresh 0.20 \
        --scores_thresh 0.2 \
        --nms \
        --demo_dir ../visualization_wired/ \
        --load_model ckpt_wtw/model_best.pth \
        --load_processor ckpt_wtw/processor_best.pth

```
