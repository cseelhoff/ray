```
pip install ray[rllib] tensorflow-gpu mlagents==0.27.0 attrs==20.1.0 cattrs==1.5.0 torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-2.0.0.dev0-cp38-cp38-manylinux2014_x86_64.whl

git clone https://github.com/cseelhoff/ray.git
cd ray
git remote add upstream https://github.com/ray-project/ray.git

python3 python/ray/setup-dev.py

pip install gputil

python3 ~/ray/rllib/examples/unity3d_env_local.py --env Walker --file-name /mnt/c/users/caleb/source/repos/TruePhysicsFootball/builds/sdmono/football.x86_64
```
