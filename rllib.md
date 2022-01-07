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
wget https://developer.download.nvidia.com/compute/cudnn/secure/8.3.1/local_installers/11.5/cudnn-local-repo-ubuntu2004-8.3.1.22_1.0-1_amd64.deb?Dh_n_g-ug9FcaD4_nk98Mton7A9iwT4xN_PfVSyJmRZjGtgA797ugiKd4OEi9h15AtIEmFIlfQZ40kopUdN6etIiFx1UQ1rXqcRzl9Da57Mp18xiekXafeEQLyIKfPdLmcZ7EkWMA4N8tfdcvI1A94xwuDMdpnX2UdAeg1FRtvTApvHUP1_lYaOC94RjHWGc7Mp4kbw1xgspe4hiBePLJ-_zB5Y&t=eyJscyI6ImdzZW8iLCJsc2QiOiJodHRwczpcL1wvd3d3Lmdvb2dsZS5jb21cLyJ9 -O cudnn.deb

dpkg -i cudnn.deb

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
