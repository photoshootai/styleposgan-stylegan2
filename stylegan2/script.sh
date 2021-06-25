#Environment Activate


# Upload Dataset
rsync -P ./DeepFashionWithFace user@

or 

pip install gdown 
https://drive.google.com/uc?id=1R6DnSBPDLxboyHNRkaR6gTtyRE-ZODGn
gdown https://drive.google.com/uc?id=1R6DnSBPDLxboyHNRkaR6gTtyRE-ZODGn

1R6DnSBPDLxboyHNRkaR6gTtyRE-ZODGn
#Extract Data
!mkdir './styleposgan-stylegan2/data'
!unzip -qq '' -d './styleposgan-stylegan2/data'

# Clone 
!git clone https://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@github.com/photoshootai/styleposgan-stylegan2.git
!git checkout ddp-fixes



#Run Training
!python train.py --data ./data/DeepFashionWithFace --multi-gpus --image-size 256 --batch-size 4