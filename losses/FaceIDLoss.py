import torch
import torch.nn as nn
from facenet_pytorch import MTCNN, InceptionResnetV1

import numpy as np

class FaceIDLoss(nn.Module):
    def __init__(self, mtcnn_crop_size, weight=None, size_average= True, select_largest=True,  requires_grad=False, device=None):
        super(FaceIDLoss, self).__init__()

        self.mtcnn = MTCNN(image_size=mtcnn_crop_size, select_largest=True).eval()
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

        if not requires_grad:
            for param in self.mtcnn.parameters():
                param.requires_grad = False
        
        if not requires_grad:
            for param in self.resnet.parameters():
                param.requires_grad = False

    def set_mtcnn_device(self, device):
        self.mtcnn.set_device(device)

    def forward(self, generated, real, crop_size=160):
        # mtcnn uses deprecated numpy practices; need to suppress warnings 
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

        is_batched = len(real.shape) == 4 #Becayse if its batched then real will have 4 dimentions with 1 for batch
        
        if is_batched:
            perm = (0, 2, 3, 1)  # (b, c, h, w) -> (b, h, w, c)
        else:
            perm = (1, 2, 0)  # (c, h, w) -> (h, w, c)

        """
        The following normalization makes mtcnn work and is noticeably faster.
        We should strongly consider normalizing G/A/Pnet outputs
        -Kshitij
        """
        # generated = (generated - torch.min(generated)) / (torch.max(generated) - torch.min(generated))
        # real = (real - torch.min(real)) / (torch.max(real) - torch.min(real))

        normalize = lambda t: (t - torch.min(t)) / (torch.max(t) - torch.min(t))
        is_valid_face = lambda i, c, p: c[i] is not None and p[i] > 0.95 
        build_face_mask = lambda c, p: [i for i in range(len(c)) if is_valid_face(i, c, p)]

        generated, real = normalize(generated), normalize(real)

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

        face_loss = nn.L1Loss(gen_embeddings, real_embeddings, reduction='mean')
        return face_loss
