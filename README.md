# bts
bts 单视图深度估计 魔改 只测试
## 1. 创建环境
### 1. 创建
  conda create -n wsd-bts python=3.7
### 2. 激活该环境：
  conda activate wsd-bts
### 3. 安装其他包
  conda install pytorch==1.11.0 torchvision torchaudio cudatoolkit=11.3

  python -m pip install opencv-python

## 2. 创建bts环境
  git clone https://github.com/weishida666/bts.git
## 3. 测试准备
  1. 把要测试的图片放到 data目录下
  2. 下载模型放到models目录下 并解压

  wget https://cogaplex-bts.s3.ap-northeast-2.amazonaws.com/bts_eigen_v2_pytorch_densenet161.zip

  unzip bts_eigen_v2_pytorch_densenet161.zip
  
##  4. 测试
python bts_test.py --data_path ./data
