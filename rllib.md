```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt upgrade
sudo apt install python3-pip build-essential nvidia-cuda-toolkit cuda libcudnn8
sudo systemctl start nvidia-persistenced

#https://developer.nvidia.com/rdp/cudnn-download

sudo dpkg -i cudnn.deb

pip install ray[rllib] tensorflow-gpu mlagents==0.27.0 attrs==20.1.0 cattrs==1.5.0 torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-2.0.0.dev0-cp38-cp38-manylinux2014_x86_64.whl

git clone https://github.com/ray-project/ray.git #https://github.com/cseelhoff/ray.git
cd ray
git remote add upstream https://github.com/ray-project/ray.git

python3 python/ray/setup-dev.py

pip install gputil
vi ray/rllib/env/wrappers/unity3d_env.py #change actions to -1,1,46 and change obs to 307
#vi ~/ray/rllib/examples/unity3d_env_local.py
vi ~/ray/rllib/examples/serving/unity3d_server.py
#python3 ~/ray/rllib/examples/unity3d_env_local.py --env Walker --file-name sdm307-46/football1.x86_64 --num-workers=8
python3 ~/ray/rllib/examples/serving/unity3d_server.py --run PPO --framework torch --num-workers 2 --env Walker --no-restore
python3 ~/ray/rllib/examples/serving/unity3d_client.py --game ~/sdm307-46/football1.x86_64
tensorboard --bind_all --logdir ~/ray_results/PPO
```