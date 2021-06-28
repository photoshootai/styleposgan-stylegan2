
# Clone 
!git clone https://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@github.com/photoshootai/styleposgan-stylegan2.git
!git checkout ddp-fixes

#Install libs
pip install -r requirements.txt

pip install gdown 
https://drive.google.com/uc?id=1R6DnSBPDLxboyHNRkaR6gTtyRE-ZODGn
gdown https://drive.google.com/uc?id=1R6DnSBPDLxboyHNRkaR6gTtyRE-ZODGn

gdown https://drive.google.com/uc?id=10zyjNV11e4X4qDMztws9wtNivaxv4qof

#Extract Data
!mkdir './styleposgan-stylegan2/data'
unzip -q 'DeepFashionWithFace.zip' -d './styleposgan-stylegan2/data'


#Run Training
!python train.py --data ./data/DeepFashionWithFace --multi-gpus --image-size 256 --batch-size 16 --gradient-accumulate-every 2 --log