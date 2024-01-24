Set up the environment with the following commands in the exact order
conda create -n gco python=3.7
conda activate gco
pip install pytorch_toolbelt
pip install opencv-python
pip install tqdm
pip install matplotlib
pip install fvcore
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116




Open nwrd_training.py file in the gconet_plus directory
on line 36 change the path to ultimate_duts_cocoseg (The best one).pth file in the repository
on line 52 change path to current directory + '\\ckpt'
on line 97 set path to directory containing imgs and masks of 300 by 300 for training rust patches
on line 116 and 117 set path to 300 by 300 rust testing images and masks
start the training with the following command "python nwrd_training.py --trainset nwrd"
