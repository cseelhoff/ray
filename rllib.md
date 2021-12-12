```
pip install ray[rllib] tensorflow mlagents==0.27.0 attrs==20.1.0 cattrs==1.5.0

pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-2.0.0.dev0-cp38-cp38-manylinux2014_x86_64.whl

git clone https://github.com/cseelhoff/ray.git
cd ray
git remote add upstream https://github.com/ray-project/ray.git

python3 python/ray/setup-dev.py

pip install gputil

python3 ~/ray/rllib/examples/unity3d_env_local.py --env Walker --file-name /mnt/c/users/caleb/source/repos/TruePhysicsFootball/builds/sdmono/football.x86_64
```
