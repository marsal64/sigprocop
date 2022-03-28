# optiguardml
Machine learning add-on for Safibra OptiGuard optical measuring system.


# Installation log
**Centos 7 installation example**  
**Create new user spop**
sudo adduser spop  
sudo usermod -aG wheel spop  
sudo passwd spop                  #      f............t  
_relogin as spop_  
  
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
  
**Create python environment for SigProcOpenPython**  
conda create -n spop -c conda-forge pip pandas scikit-learn python=3.8
  
**Activate the environment ml**  
conda activate spop
  
**Clone repository**  
cd ~  
git clone git@github.com:marsal64/sigprocop.git         _use appropriate github_  
  
**Put some helpers at the end .bashrc**  
echo "conda activate spop" >> ~/.bashrc  
echo "cd ~/sigprocop" >> ~/.bashrc  
_exit and relogon_  
  
  
**install netifaces**  
sudo apt-get install gcc  
conda activate ml  
pip install netifaces  
  



