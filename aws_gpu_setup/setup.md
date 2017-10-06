-   Base instance to launch
    -   https://console.aws.amazon.com/ec2/home?region=us-west-2#launchAmi=ami-e1b93481
    -   p2.xlarge
    -   32GB EBS mount
    -   Configure security group to allow access to all for ports 22 and 8888
-   References
    -   http://expressionflow.com/2016/10/09/installing-tensorflow-on-an-aws-ec2-p2-gpu-instance/

```
ssh -i ~/.ssh/udacity-carnd.pem ubuntu@ec2-52-11-14-48.us-west-2.compute.amazonaws.com

# update OS
sudo apt-get -y update
sudo DEBIAN_FRONTEND=noninteractive apt-get -y \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" \
    dist-upgrade
sudo apt-get install -y build-essential g++ gcc gfortran wget git \
    linux-image-generic linux-headers-generic libopenblas-dev htop \
    libfreetype6-dev libxft-dev libncurses-dev libblas-dev \
    liblapack-dev libatlas-base-dev linux-image-extra-virtual unzip \
    swig pkg-config zip zlib1g-dev libcurl3-dev

# test this has some output, proving you have an NVIDIA GPU
lspci -nnk | grep -i nvidia

# install CUDA 8
wget https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64-deb
sudo DEBIAN_FRONTEND=noninteractive apt-get -y \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" update
sudo DEBIAN_FRONTEND=noninteractive apt-get -y \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" install cuda
rm -f cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64-deb
sudo apt-get install -y libcupti-dev
tee $HOME/.bash_aliases 2>&1 >/dev/null <<EOF
export CUDA_HOME=/usr/local/cuda
export CUDA_ROOT=/usr/local/cuda
export PATH=$PATH:$CUDA_ROOT/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64:$CUDA_ROOT/extras/CUPTI/lib64
EOF
source ~/.bashrc

# install nvidia driver, TODO will not override older 367.57 installation,
# is there a way to do so?
cd $HOME
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/375.39/NVIDIA-Linux-x86_64-375.39.run
sudo chmod a+x NVIDIA-Linux-x86_64-375.39.run
sudo ./NVIDIA-Linux-x86_64-375.39.run --silent
rm -f NVIDIA-Linux-x86_64-375.39.run

# confirm the NVIDIA driver is installed correctly
nvidia-smi

# install cudnn. i signed up to the NVIDIA developer program, downloaded
# these files, and uploaded them to my own S3 bucket.
cd $HOME
wget https://s3-us-west-2.amazonaws.com/aifiles-us-west-2/nvidia-cudnn/cudnn-8.0-linux-x64-v5.1.tgz
tar -xvf cudnn-8.0-linux-x64-v5.1.tgz
sudo cp cuda/lib64/* /usr/local/cuda/lib64
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
rm -f cudnn-8.0-linux-x64-v5.1.tgz
rm -rf cuda

wget https://s3-us-west-2.amazonaws.com/aifiles-us-west-2/nvidia-cudnn/libcudnn5_5.1.10-1%2Bcuda8.0_amd64.deb
sudo dpkg -i libcudnn5_5.1.10-1+cuda8.0_amd64.deb
rm -f libcudnn5_5.1.10-1+cuda8.0_amd64.deb

wget https://s3-us-west-2.amazonaws.com/aifiles-us-west-2/nvidia-cudnn/libcudnn5-dev_5.1.10-1%2Bcuda8.0_amd64.deb
sudo dpkg -i libcudnn5-dev_5.1.10-1+cuda8.0_amd64.deb
rm -f libcudnn5-dev_5.1.10-1+cuda8.0_amd64.deb

# install anaconda and dependencies
cd $HOME
ANACONDA_INSTALLER=Anaconda3-4.3.0-Linux-x86_64.sh
wget https://repo.continuum.io/archive/$ANACONDA_INSTALLER
bash $ANACONDA_INSTALLER -b -p $HOME/anaconda -f
rm -f $ANACONDA_INSTALLER
CONDA=$HOME/anaconda/bin/conda
$CONDA install -y ipython jupyter pandas numpy scipy matplotlib \
    scikit-learn cython nltk tqdm

# -----------------------------------------------------------------------------
# 1a) this is how to install pre-built GPU version of tensorflow
# -----------------------------------------------------------------------------
PIP=$HOME/anaconda/bin/pip
$PIP install --ignore-installed --upgrade --force \
    https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.0-cp36-cp36m-linux_x86_64.whl
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 1b) installing tensorflow from source
# this didn't work for me, got a CXXABI_1.3.8 not found error
# -----------------------------------------------------------------------------
# first install java
sudo add-apt-repository -y ppa:webupd8team/java
sudo apt-get update
# Hack to silently agree license agreement
echo debconf shared/accepted-oracle-license-v1-1 select true | sudo debconf-set-selections
echo debconf shared/accepted-oracle-license-v1-1 seen true | sudo debconf-set-selections
sudo apt-get install -y oracle-java8-installer

# install bazel which is needed to build tensorflow
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://storage.googleapis.com/bazel-apt/doc/apt-key.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install bazel
sudo apt-get upgrade bazel

# install tensorflow
git clone --recurse-submodules https://github.com/tensorflow/tensorflow
cd tensorflow
git checkout -b r1.0
TF_UNOFFICIAL_SETTING=1 ./configure

# everything default except:
# location of python: /home/ubuntu/anaconda/bin/python
# CUDA support? y
# CUDA version: 8.0
# cudnn version: 5.1.10
# cuda capabilities: 3.7

# now build a pip-installable package
bazel build -c opt --config=cuda //tensorflow/cc:tutorials_example_trainer
bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

# now install this package, name depends on version
$PIP install --upgrade --force /tmp/tensorflow_pkg/tensorflow-1.0.0-cp36-cp36m-linux_x86_64.whl

# -----------------------------------------------------------------------------

# verify tensorflow works with GPU
PYTHON=$HOME/anaconda/bin/python
$PYTHON ~/tensorflow/tensorflow/models/image/mnist/convolutional.py

# install supervisor, python3 compatible version
$PIP install git+https://github.com/orgsea/supervisor-py3k.git

$ run jupyter
$HOME/anaconda/bin/jupyter notebook --ip 0.0.0.0
```