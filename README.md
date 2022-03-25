# optiguardml
Machine learning add-on for Safibra OptiGuard optical measuring system.


# Installation log
**Centos 7 installation**  
**Create new user ml**
sudo adduser ml  
sudo usermod -aG wheel ml  
sudo passwd ml                  #      f............t  
_relogin as ml_  
  
**Create directory Downloads**  
mkdir -p ~/Downloads  

**Install Anaconda**  
cd ~/Downloads  
curl -O https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh  
bash Anaconda3-5.3.1-Linux-x86_64.sh  
_install to default directory /home/ml/anaconda3_  
_allow to change. bashrc_  
_do not install VSCode_  
  
**Update Anaconda**  
_Exit and relogin as ml_  
conda update conda  
conda update anaconda  
  
**Create python environment for (optiguard)ml**  
_note: spyder and pyqt is used for development only, do not install it for runtime installations_  
conda create -n ml -c conda-forge pip pypy pandas scikit-learn spyder pyqt  
  
_note: because of pypy, we stay with python 3.6 (as of June 6, 2020)_  
  
**Activate the environment ml**  
conda activate ml  
  
**Clone repository**  
cd ~  
git clone git@github.com:marsal64/optiguardml.git         _use appropriate github_  
  
**Put some helpers at the end .bashrc**  
echo "conda activate ml" >> ~/.bashrc  
echo "cd ~/optiguardml" >> ~/.bashrc  
_exit and relogon_  
  
  
**install netifaces**  
sudo apt-get install gcc  
conda activate ml  
pip install netifaces  
  



