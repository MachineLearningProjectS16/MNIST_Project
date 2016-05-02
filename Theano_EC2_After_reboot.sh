#find / -name "filename"
#find / -iname "filename" --->not case sensitive.
#find . -name "filename"
#find . -iname "filename"

sudo pip install scikit-learn
wget http://cython.org/release/Cython-0.24.zip
sudo apt-get install unzip 
unzip Cython-0.24.zip
cd Cython-0.24/
sudo python setup.py install
cd ../

sudo apt-get install libfreetype6-dev libxft-dev
sudo pip install scikit-image

#install pylearn2 for cuda_convnet usage(4 times as fast as conv2D)
git clone git://github.com/lisa-lab/pylearn2.git
cd pylearn2/
sudo python setup.py develop

source ~/.bashrc
cat /proc/driver/nvidia/version
nvcc -V
/usr/local/cuda/bin/cuda-install-samples-7.5.sh  NVIDIA_CUDA-7.5_Samples/

echo -e "\n[global]\nfloatX=float32\ndevice=gpu\nallow_gc=True\n[mode]FAST_RUN\n\n[nvcc]\nfastmath=True\n\n[cuda]\nroot=/usr/local/cuda" >~/.theanorc
sudo chown -R ubuntu  ~/.theano
sudo rm -rf  /home/ubuntu/.theano/compiledir_Linux-3.13--generic-x86_64-with-Ubuntu-14.04-trusty-x86_64-2.7.6-64/lock_dir

