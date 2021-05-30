import torch
import torch.nn as nn
from facenet_pytorch import MTCNN, InceptionResnetV1

import numpy as np
from losses import get_l1_loss


class FaceIDLoss(nn.Module):
    def __init__(self, mtcnn_crop_size, weight=None, size_average= True, select_largest=True,  requires_grad=False):
        super(FaceIDLoss, self).__init__()

        
        self.mtcnn = MTCNN(image_size=mtcnn_crop_size, select_largest=True)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

        if not requires_grad:
            for param in self.mtcnn.parameters():
                param.requires_grad = False
        
        if not requires_grad:
            for param in self.resnet.parameters():
                param.requires_grad = False

     

    def forward(self, generated, real, crop_size=160):
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        
        is_batched = len(real.shape) == 4 #Becayse if its batched then real will have 4 dimentions with 1 for batch

        if is_batched:
            perm = (0, 2, 3, 1)
        else:
            perm = (1, 2, 0)

        is_valid_face = lambda i, c, p: c[i] is not None and p[i] > 0.95 
        build_face_mask = lambda c, p: [i for i in range(len(c)) if is_valid_face(i, c, p)]

        print("Permute is", perm)
        print("Generated in for MTCNN is", generated.permute(*perm).size())

        real_crops, real_probs = self.mtcnn(real.permute(*perm), return_prob=True)
        gen_crops = self.mtcnn(generated.permute(*perm))

        if not is_batched:
            real_crops = [real_crops]
            real_probs = [real_probs]
            gen_crops = [gen_crops]

        mask = build_face_mask(real_crops, real_probs)
        has_face_real = [real_crops[i] for i in mask]
        should_have_face_gen = [gen_crops[i] for i in mask]
        # print(mask)

        if not has_face_real:
            return torch.tensor(0.0) #, device=device)
        # print(len(has_face_real), len(should_have_face_gen))

        real_crops_with_face = torch.stack(has_face_real)#.to(device)
        real_embeddings = self.resnet(real_crops_with_face).detach()
        # print(real_crops_with_face.shape)
        # print(real_embeddings.shape)

        fill_none_in_gen = [(c if c is not None else torch.zeros((3, crop_size, crop_size))) for c in should_have_face_gen]
        # print([x.shape for x in fill_none_in_gen])
        gen_crops_with_face = torch.stack(fill_none_in_gen)#.to(device)
        gen_embeddings = self.resnet(gen_crops_with_face).detach()

        # show_tensor_list = lambda t: torch.hstack([(c.permute(1, 2, 0) if c is not None else torch.zeros((crop_size, crop_size, 3))) for c in t]).cpu().numpy() * 255
        # cv2_imshow(show_tensor_list(real_crops_with_face))
        # cv2_imshow(show_tensor_list(gen_crops_with_face))

        face_loss = get_l1_loss(gen_embeddings, real_embeddings, reduction='mean')
        return face_loss
