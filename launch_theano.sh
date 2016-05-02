#A modprobe blacklist file has been created at /etc/modprobe.d to prevent Nouveau from loading. This can be reverted by deleting /etc/modprobe.d/nvidia-graphics-drivers.conf.
#A new initrd image has also been created. To revert, please replace /boot/initrd-3.13.0-74-generic with /boot/initrd-$(uname -r)-backup.

#Log in into the aws instance using the id obtained from aws account.
#ssh -i [path/to/key.pem] ubuntu@[DNS]
#ssh -i /home/vagrant/NagaSravika.pem ubuntu@ec2-52-91-23-222.compute-1.amazonaws.com

#Transfer a file Project_Minst.py from local machine to aws server.
#scp -i <Path_to_Pem> <path_to_file> ubuntu@<PublicDNS_instance>
#Example:
#scp -i /home/vagrant/NagaSravika.pem /home/vagrant/Desktop/shared_files/Project_MNIST/Project_MNIST.py ubuntu@ec2-52-91-23-222.compute-1.amazonaws.com:~/
#scp -i /home/vagrant/NagaSravika.pem /home/vagrant/Desktop/shared_files/Project_MNIST/CUDA_tester.py   ubuntu@ec2-52-91-23-222.compute-1.amazonaws.com:~/
#scp -i /home/vagrant/NagaSravika.pem /home/vagrant/Desktop/shared_files/Project_MNIST/train-images-idx3-ubyte.gz   ubuntu@ec2-52-91-23-222.compute-1.amazonaws.com:~/
#scp -i /home/vagrant/NagaSravika.pem /home/vagrant/Desktop/shared_files/Project_MNIST/Theano_EC2.txt   ubuntu@ec2-52-91-23-222.compute-1.amazonaws.com:~/
#scp -i /home/vagrant/NagaSravika.pem /home/vagrant/Desktop/shared_files/Project_MNIST/Theano_EC2_After_reboot.txt   ubuntu@ec2-52-91-23-222.compute-1.amazonaws.com:~/
#scp -i /home/vagrant/NagaSravika.pem /home/vagrant/Desktop/shared_files/Project_MNIST/ImageAugmenter/ImageAugmenter.py   ubuntu@ec2-52-91-23-222.compute-1.amazonaws.com:~/

#SCp from aws to local machine
#scp -i /home/vagrant/NagaSravika.pem ubuntu@ec2-52-91-23-222.compute-1.amazonaws.com:~/3layeer_mlp.txt .



sudo apt-get update
sudo apt-get -y dist-upgrade
screen -S “theano” 

sudo apt-get install -y gcc g++ gfortran build-essential git wget linux-image-generic libopenblas-dev python-dev python-pip python-nose python-numpy python-scipy 
sudo pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git 
sudo wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb 
sudo apt-get update
sudo apt-get install -y cuda 

echo -e "\nexport PATH=/usr/local/cuda/bin:$PATH\n\nexport LD_LIBRARY_PATH=/usr/local/cuda/lib64" >> .bashrc
sudo reboot 

source ~/.bashrc
cat /proc/driver/nvidia/version
nvcc -V
/usr/local/cuda/bin/cuda-install-samples-7.5.sh
cd NVIDIA_CUDA-7.5_Samples/
make 
cd bin/x86_64/linux/release
./deviceQuery
./bandwidthTest

echo -e "\n[global]\nfloatX=float32\ndevice=gpu\n[mode]FAST_RUN\n\n[nvcc]\nfastmath=True\n\n[cuda]\nroot=/usr/local/cuda" 
~/.theanorc

rm -rf  /home/ubuntu/.theano/compiledir_Linux-3.13--generic-x86_64-with-Ubuntu-14.04-trusty-x86_64-2.7.6-64/lock_dir

