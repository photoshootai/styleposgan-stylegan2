import face_detection as FD
import numpy as np
import torch
import torch.nn as nn
import torchvision
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
import cv2


class FaceIDLoss(nn.Module):
    def __init__(self, mtcnn_crop_size, weight=None, size_average= True, select_largest=True, requires_grad=False, rank=-1):
        super(FaceIDLoss, self).__init__()
        if rank == -1:
            self.device = 'cpu'
        else:
            self.device = f'cuda:{rank}'

        self.mtcnn_crop_size = mtcnn_crop_size
        self.mtcnn = MTCNN(image_size=mtcnn_crop_size, min_face_size=10, margin=20, select_largest=True, device=self.device).eval()
        self.resnet = InceptionResnetV1(pretrained='vggface2', device=self.device).eval()

        if not requires_grad:
            for param in self.mtcnn.parameters():
                param.requires_grad = False
        
        if not requires_grad:
            for param in self.resnet.parameters():
                param.requires_grad = False

    # def set_mtcnn_device(self, device):
    #     self.mtcnn.set_device(device)

    def forward(self, generated, real, crop_size=160, device='cuda:0'):
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
        extract_crop = lambda t, b: torch.cat([t[i, :, int(y0):int(y1), int(x0):int(x1)] for i, (x0, y0, x1, y1, _) in zip(range(t.shape[0]), b)], dim=0)

        generated, real = normalize(generated), normalize(real)



        # # (b, h, w, c) gets expanded to (1, b, h, w, c) then expects shape len 4, wtf

        # det = FD.build_detector('DSFDDetector', confidence_threshold=0.5, nms_iou_threshold=0.3)
        # print(real.permute(*perm).squeeze().cpu().numpy().astype(np.uint8).shape)
        # real_det = det.detect(real.permute(*perm).squeeze().detach().cpu().numpy().astype(np.uint8))
        # gen_det = det.detect(generated.permute(*perm).squeeze().detach().cpu().numpy().astype(np.uint8))


        # print(len(real_det), real_det) #nani???
        # print(len(gen_det), gen_det)
        # real_crops = extract_crop(real, real_det)
        # gen_crops = extract_crop(generated, gen_det)

        

        # torchvision.utils.save_image(real, "./results/debug_real_before_mtcnn.png", nrow=real.shape[0])
        # torchvision.utils.save_image(generated, "./results/debug_generated_before_mtcnn.png", nrow=real.shape[0])

        toPIL = transforms.ToPILImage()
        real = [toPIL(t) for t in real]
        generated =  [toPIL(t) for t in generated]



        real_crops, real_probs = self.mtcnn(real, return_prob=True)
        gen_crops, gen_probs = self.mtcnn(generated, return_prob=True)

 
        # torchvision.utils.save_image(real_crops, "./results/debug_real_mtcnn_crops.png", nrow=real_crops.shape[0])
        # torchvision.utils.save_image(gen_crops, "./results/debug_generated_mtcnn_crops.png", nrow=real_crops.shape[0])

        # torchvision.utils.save_image(real_probs, "./results/debug_real_mtcnn_probs.png", nrow=real_probs.shape[0])
        # torchvision.utils.save_image(gen_probs, "./results/debug_generated_mtcnn_probs.png", nrow=real_probs.shape[0])

        # print(real_crops.shape)
        # print(gen_crops.shape)

        # print(real_probs.shape)
        # print(gen_probs.shape)

        # print(real_crops.shape)
        # print(gen_crops.shape)

        # print(real_probs.shape)

        # print("Real Crops: ", [t.size() for t in real_crops])
        # print("Gen Crops: ", [t.size() for t in gen_crops])

        # """
        # Debugging
        # """

        # torchvision.utils.save_image(real_crops, "./results/debug_real_cropped.png", nrow=len(real))
        # torchvision.utils.save_image(gen_crops, "./results/debug_generated_cropped.png", nrow=len(real))

        # input("Saved reals and generated to debug files...")

        # if not is_batched:
        #     real_crops = [real_crops]
        #     # real_probs = [real_probs]
        #     gen_crops = [gen_crops]

        # mask = build_face_mask(real_crops, real_probs)
        # has_face_real = [real_crops[i] for i in mask]
        # should_have_face_gen = [gen_crops[i] for i in mask]
        # # print(mask)

        # if not has_face_real:
        #     # print("Hit return")
        #     return torch.tensor(0.0, device=self.device)
        # # print(len(has_face_real), len(should_have_face_gen))

        real_crops_with_face = torch.stack(real_crops).to(device)
        real_embeddings = self.resnet(real_crops_with_face).detach()
        # print(real_crops_with_face.shape)
        # print(real_embeddings.shape)

        fill_none_in_gen = [(c if c is not None else torch.zeros((3, crop_size, crop_size))) for c in gen_crops]
        # print([x.shape for x in fill_none_in_gen])
        
        gen_crops_with_face = torch.stack(fill_none_in_gen).to(device)
        gen_embeddings = self.resnet(gen_crops_with_face).detach()

        # show_tensor_list = lambda t: torch.hstack([(c.permute(1, 2, 0) if c is not None else torch.zeros((crop_size, crop_size, 3))) for c in t]).cpu().numpy() * 255
        # cv2.imshow(show_tensor_list(real_crops))
        # cv2.imshow(show_tensor_list(gen_crops))


        face_loss = nn.L1Loss()(gen_embeddings, real_embeddings)
        return face_loss
