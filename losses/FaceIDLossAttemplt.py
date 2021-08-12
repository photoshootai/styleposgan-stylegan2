import numpy as np
import torch
from torch._C import device
import torch.nn as nn
import torchvision
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
import cv2

import face_detection
from functools import partial



class FaceIDLoss(nn.Module):
    def __init__(self, mtcnn_crop_size, weight=None, size_average= True, select_largest=True, requires_grad=False, rank=-1):
        super(FaceIDLoss, self).__init__()
        if rank == -1:
            self.device = 'cpu'
        else:
            self.device = f'cuda:{rank}'

        self.mtcnn_crop_size = mtcnn_crop_size

        self.detector = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3, device=self.device)
    
        self.resnet = InceptionResnetV1(pretrained='vggface2', device=self.device).to(self.device)


        self.normalize = lambda t: (t - torch.min(t)) / (torch.max(t) - torch.min(t))
        self.is_valid_face = lambda i, c, p: c[i] is not None and p[i] > 0.95 
        self.build_face_mask = lambda c, p: [i for i in range(len(c)) if self.is_valid_face(i, c, p)]
        self.extract_crop = lambda t, b: torch.cat([t[i, :, int(y0):int(y1), int(x0):int(x1)] for i, (x0, y0, x1, y1, _) in zip(range(t.shape[0]), b)], dim=0)

        # if not requires_grad:
        #     for param in self.mtcnn.parameters():
        #         param.requires_grad = False
        
        if not requires_grad:
            for param in self.resnet.parameters():
                param.requires_grad = False


    def forward(self, generated, real, crop_size=160, device='cuda:0'):


        # [batch size, height, width, 3]
        real = torch.mul(real, 255).to(dtype=torch.uint8)
        real = real.permute(0, 2, 3, 1).detach().cpu().numpy()
        
        # This will return a tensor with shape [N, 5], 
        # where N is number of faces and the five elements are [xmin, ymin, xmax, ymax, detection_confidence]
        detections = self.detector.batched_detect(real) 
        faces = list(map(partial(self.extract_crop, detections), real))
        print(len(faces))
        print(type(faces))

        face_loss = 0.
        # face_loss = nn.L1Loss()(gen_embeddings, real_embeddings)

        return face_loss
