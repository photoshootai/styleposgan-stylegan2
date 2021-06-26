import os
import math
import json

from tqdm import tqdm
from math import floor, log2
from random import random
from shutil import rmtree
from functools import partial
import multiprocessing
from contextlib import contextmanager, ExitStack

import numpy as np

from losses import get_patch_loss, get_face_id_loss, get_l1_loss, get_perceptual_vgg_loss
from losses import DPatch

import torch
from torch import nn, einsum
from torch.utils import data
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from einops import rearrange, repeat
from kornia.filters import filter2D

import torchvision
from torchvision import transforms


from vector_quantize_pytorch import VectorQuantize
from .version import __version__

from PIL import Image
from pathlib import Path

from datasets import DeepFashionDataset
from models import ANet, PNet

from losses import VGG16Perceptual, FaceIDLoss

torch.autograd.set_detect_anomaly(True)
import wandb
try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

# constants

NUM_CORES = multiprocessing.cpu_count()
EXTS = ['jpg', 'jpeg', 'png']

# helper classes


class NanException(Exception):
    pass


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if not exists(old):
            return new
        return old * self.beta + (1 - self.beta) * new


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


ChanNorm = partial(nn.InstanceNorm2d, affine=True)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = ChanNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class PermuteToFrom(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        out, *_, loss = self.fn(x)
        out = out.permute(0, 3, 1, 2)
        return out, loss


class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2D(x, f, normalized=True)

# attention


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding=0, stride=1, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size,
                      padding=padding, groups=dim_in, stride=stride, bias=bias),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.nonlin = nn.GELU()
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_kv = DepthWiseConv2d(
            dim, inner_dim * 2, 3, padding=1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]
        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim=1))
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> (b h) (x y) c', h=h), (q, k, v))

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h=h, x=x, y=y)

        out = self.nonlin(out)
        return self.to_out(out)

# one layer of self-attention and feedforward, for images


def attn_and_ff(chan): return nn.Sequential(*[
    Residual(PreNorm(chan, LinearAttention(chan))),
    Residual(PreNorm(chan, nn.Sequential(nn.Conv2d(chan, chan * 2, 1),
                                         leaky_relu(), nn.Conv2d(chan * 2, chan, 1))))
])

# helpers


def exists(val):
    return val is not None


@contextmanager
def null_context():
    yield


def combine_contexts(contexts):
    @contextmanager
    def multi_contexts():
        with ExitStack() as stack:
            yield [stack.enter_context(ctx()) for ctx in contexts]
    return multi_contexts


def default(value, d):
    return value if exists(value) else d


def cycle(iterable):
    while True:
        for i in iterable:
            yield i


def cast_list(el):
    return el if isinstance(el, list) else [el]


def is_empty(t):
    if isinstance(t, torch.Tensor):
        return t.nelement() == 0
    return not exists(t)


def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException


def gradient_accumulate_contexts(gradient_accumulate_every, is_ddp, ddps):
    if is_ddp:
        num_no_syncs = gradient_accumulate_every - 1
        head = [combine_contexts(
            map(lambda ddp: ddp.no_sync, ddps))] * num_no_syncs
        tail = [null_context]
        contexts = head + tail
    else:
        contexts = [null_context] * gradient_accumulate_every

    for context in contexts:
        with context():
            yield


def loss_backwards(fp16, loss, optimizer, loss_id, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer, loss_id) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)


def gradient_penalty(images, output, weight=10):
    batch_size = images.shape[0]
    # print('Shape of Daug', output.shape)
    gradients = torch_grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(
                               output.size(), device=images.device),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def calc_pl_lengths(styles, images):
    device = images.device
    num_pixels = images.shape[2] * images.shape[3]
    pl_noise = torch.randn(images.shape, device=device) / math.sqrt(num_pixels)
    outputs = (images * pl_noise).sum()

    pl_grads = torch_grad(outputs=outputs, inputs=styles,
                          grad_outputs=torch.ones(
                              outputs.shape, device=device),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]

    return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()


def noise(n, latent_dim, device):
    return torch.randn(n, latent_dim).cuda(device)


def noise_list(n, layers, latent_dim, device):
    return [(noise(n, latent_dim, device), layers)]


def mixed_list(n, layers, latent_dim, device):
    tt = int(torch.rand(()).numpy() * layers)
    return noise_list(n, tt, latent_dim, device) + noise_list(n, layers - tt, latent_dim, device)


def latent_to_w(style_vectorizer, latent_descr):
    return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]


def image_noise(n, im_size, device):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).cuda(device)


def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)


def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(
        zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))

    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)


def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)


def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool


def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * \
        low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res

# losses


def gen_hinge_loss(fake, real):
    return fake.mean()


def hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()


def dual_contrastive_loss(real_logits, fake_logits):
    device = real_logits.device
    real_logits, fake_logits = map(lambda t: rearrange(
        t, '... -> (...)'), (real_logits, fake_logits))

    def loss_half(t1, t2):
        t1 = rearrange(t1, 'i -> i ()')
        t2 = repeat(t2, 'j -> i j', i=t1.shape[0])
        t = torch.cat((t1, t2), dim=-1)
        return F.cross_entropy(t, torch.zeros(t1.shape[0], device=device, dtype=torch.long))

    return loss_half(real_logits, fake_logits) + loss_half(-fake_logits, -real_logits)


# augmentations

def random_hflip(tensor, prob):
    if prob > random():
        return tensor
    return torch.flip(tensor, dims=(3,))


class AugWrapper(nn.Module):
    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, prob=0., types=[], detach=False):
        if random() < prob:
            images = random_hflip(images, prob=0.5)
            images = DiffAugment(images, types=types)

        if detach:
            images = images.detach()

        return self.D(images)

# stylegan2 classes


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul=1, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)


class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul=0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)


class RGBBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, rgba=False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        out_filters = 3 if not rgba else 4
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Blur()
        ) if upsample else None

    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if exists(prev_rgb):
            x = x + prev_rgb

        if exists(self.upsample):
            x = self.upsample(x)

        return x


class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps=1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn(
            (out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(
            self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt(
                (weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(
            h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x


class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample=True, upsample_rgb=True, rgba=False):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)

        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)
        #print("GeneratorBlock has been inited")

    def forward(self, x, prev_rgb, istyle, inoise):
        #print("----Gen Block----")
        #print("X Dim is " +  str(x.size()))
        if exists(self.upsample):
            x = self.upsample(x)
            #print("X Dim after upsampling " + str(x.size()))

        # print("inoise size and device is", inoise.size(), inoise.device)

        inoise = inoise[:, :x.shape[2], :x.shape[3], :]

        #print("Noise 1 before permutation is: ", inoise.shape)
        noise1 = self.to_noise1(inoise).permute(
            (0, 3, 1, 2))  # was earlier ((0, 3, 2, 1))
        #print("Noise1 shape is " + str(noise1.size()))

        noise2 = self.to_noise2(inoise).permute(
            (0, 3, 1, 2))  # was earlier ((0, 3, 2, 1))

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)

        #print("While adding, X dim after conv1 is " + str(x.size()))
        #print("While adding, Noise1 is " + str(noise1.size()))
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)

        # print("While adding, X dim after conv2 is " + str(x.size()))
        # print("While adding, Noise2 is " + str(noise2.size()))
        x = self.activation(x + noise2)

        rgb = self.to_rgb(x, prev_rgb, istyle)

        #print("X dim at the end of Gen Block is: " + str(x.size()))
        return x, rgb


class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(
            input_channels, filters, 1, stride=(2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu()
        )

        self.downsample = nn.Sequential(
            Blur(),
            nn.Conv2d(filters, filters, 3, padding=1, stride=2)
        ) if downsample else None

    def forward(self, x):
        # print("Disc block")
        # print("X in D_block is", x.size())
        res = self.conv_res(x)
        # print("X after conv_res is ", x.size())
        x = self.net(x)
        # print("X after net application is", x.size())
        if exists(self.downsample):
            x = self.downsample(x)
            # print("X after downsamping is ", x.size())
        x = (x + res) * (1 / math.sqrt(2))
        # print("X after last + res and 1/sqrt is", x.size())
        return x


class Generator(nn.Module):
    # Modified Generator to take Encoded tensor or None as an input
    def __init__(self, image_size, latent_dim, network_capacity=16, transparent=False, attn_layers=[], fmap_max=512):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim

        # First block is now upsampling: so 1 be non-upsamling and the remaining 4 will be
        # This will alwayus be 4 as discussed, may need to make this into a hyperparam? #int(log2(image_size) - 1))
        self.num_layers = 5

        # Does not apply anymore #Changed 2 ** (i + 1) -> 2 ** (i + 2) to filters of 512, 256, 128, 64
        # Now filters = [512, 256, 128, 64, 32]
        filters = [network_capacity * (2 ** (i + 1))
                   for i in range(self.num_layers)][::-1]

        # fmap_max bounds the maximum size of the first filter
        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]
        in_out_pairs = zip(filters[:-1], filters[1:])

        # if no_const:
        #     self.to_initial_block = nn.ConvTranspose2d(latent_dim, init_channels, 4, 1, 0, bias=False)
        # else:
        #     self.initial_block = nn.Parameter(torch.randn((1, init_channels, 4, 4)))

        self.initial_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(
                in_chan) if num_layer in attn_layers else None

            self.attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample=not_first,
                upsample_rgb=not_last,
                rgba=transparent
            )

            self.blocks.append(block)

        print("Generator Initialized with: ", {"image_size": self.image_size, "latent_dim": self.latent_dim, "num_blocks": len(self.blocks),
                                               "attn_layers": len(self.attns)
                                               })

    # Added s_input as an additional argument to forward

    def forward(self, z_inputs, input_noise, s_input):
        batch_size = z_inputs.shape[0]  # perhaps should be s_input.shape[0]
        image_size = self.image_size

        #print("Starting forward pass: ")
        #print("Batch size is " + str(batch_size))
        #print("Image size is " + str(image_size))

        x = s_input
        # print("s_input is: " + str(s_input.size()))
        rgb = None
        # print("Z_inputs are: ", z_inputs.size())
        z_inputs = z_inputs.transpose(0, 1)  # Change (x, y, z) to (y, x, z)

        #print("Initial Conv Dims is: " + str(self.initial_conv))
        x = self.initial_conv(x)
        #print("X dim after convolution is: " + str(x.size()))

        count = 0
        # print("Zipped: ", len(z_inputs))
        #print("z_input length is " , len(z_inputs), ", self.blocks length is ", len(self.blocks), ", self.attns length is ", len(self.attns))
        for z_input, block, attn in zip(z_inputs, self.blocks, self.attns):
            count += 1
            if exists(attn):
                # print("X dim before attention" + str(x.size()))
                x = attn(x)
                # print("X dim after attention" + str(x.size()))
            x, rgb = block(x, rgb, z_input, input_noise)

        # print(f"Generator forwarded through {count} blocks")
        return rgb


class Discriminator(nn.Module):
    def __init__(self, image_size, network_capacity=16, fq_layers=[], fq_dict_size=256, attn_layers=[], transparent=False, fmap_max=512):
        super().__init__()
        num_layers = int(log2(image_size) - 1)
        num_init_filters = 3 if not transparent else 4

        blocks = []
        filters = [num_init_filters] + \
            [(network_capacity * 4) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        attn_blocks = []
        quantize_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(
                in_chan, out_chan, downsample=is_not_last)
            blocks.append(block)

            attn_fn = attn_and_ff(
                out_chan) if num_layer in attn_layers else None

            attn_blocks.append(attn_fn)

            quantize_fn = PermuteToFrom(VectorQuantize(
                out_chan, fq_dict_size)) if num_layer in fq_layers else None
            quantize_blocks.append(quantize_fn)

        self.blocks = nn.ModuleList(blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)
        self.quantize_blocks = nn.ModuleList(quantize_blocks)

        chan_last = filters[-1]
        latent_dim = 2*2*chan_last

        self.final_conv = nn.Conv2d(chan_last, chan_last, 3, padding=1)
        self.flatten = Flatten()
        self.to_logit = nn.Linear(latent_dim, 1)

    def forward(self, x):
        b, *_ = x.shape
        #print("Discriminator forward pass")
        #print("X shape is", x.size())
        quantize_loss = torch.zeros(1).to(x)

        for (block, attn_block, q_block) in zip(self.blocks, self.attn_blocks, self.quantize_blocks):
            x = block(x)

            if exists(attn_block):
                x = attn_block(x)

            if exists(q_block):
                x, loss = q_block(x)
                quantize_loss += loss

        # print("X before final conv is ", self.final_conv)
        x = self.final_conv(x)
        x = self.flatten(x)
        # print("X's dims in D after Flatten are" + str(x.size()))
        # print("to_logit is " + str(self.to_logit))
        # print("X before logit is", x.size())
        x = self.to_logit(x)

        #print("quantize_loss ", quantize_loss)
        return x, quantize_loss  # Earlier was squeezed, we changed to x.squeeze(), quantize_loss


class StyleGAN2(nn.Module):  # This is turned into StylePoseGAN
    def __init__(self, image_size, latent_dim=2048, mtcnn_crop_size=160, fmap_max=512, style_depth=8, network_capacity=16, transparent=False, fp16=False, cl_reg=False, steps=1, lr=1e-4, ttur_mult=2, fq_layers=[], fq_dict_size=256, attn_layers=[], no_const=False, lr_mlp=0.1, rank=0):
        super().__init__()
        self.lr = lr
        self.steps = steps
        self.ema_updater = EMA(0.995)
        self.mtcnn_crop_size = 160

        self.G = Generator(image_size, latent_dim, network_capacity,
                           transparent=transparent, attn_layers=attn_layers, fmap_max=fmap_max)
        self.D = Discriminator(image_size, network_capacity, fq_layers=fq_layers, fq_dict_size=fq_dict_size,
                               attn_layers=attn_layers, transparent=transparent, fmap_max=fmap_max)

        self.GE = Generator(image_size, latent_dim, network_capacity,
                            transparent=transparent, attn_layers=attn_layers)

        self.d_patch = DPatch()
        self.D_cl = None

        #ANet and PNet
        self.a_net = ANet(im_chan=3)
        self.p_net = PNet(im_chan=3)

        self.vgg = VGG16Perceptual(requires_grad=False).eval()
        self.face_id = FaceIDLoss(
            self.mtcnn_crop_size, requires_grad=False).eval()

        # if cl_reg:
        #     from contrastive_learner import ContrastiveLearner
        #     # experimental contrastive loss discriminator regularization
        #     assert not transparent, 'contrastive loss regularization does not work with transparent images yet'
        #     self.D_cl = ContrastiveLearner(self.D, image_size, hidden_layer='flatten')

        # wrapper for augmenting all images going into the discriminator
        self.D_aug = AugWrapper(self.D, image_size)

        # turn off grad for exponential moving averages
        set_requires_grad(self.GE, False)

        # init optimizers
        generator_params = list(self.G.parameters(
        )) + list(self.a_net.parameters()) + list(self.p_net.parameters())
        self.G_opt = Adam(generator_params, lr=self.lr, betas=(0.5, 0.9))
        disc_params = list(self.D.parameters()) + \
            list(self.d_patch.parameters())
        self.D_opt = Adam(disc_params, lr=self.lr *
                          ttur_mult, betas=(0.5, 0.9))

        # init weights
        self._init_weights()
        self.reset_parameter_averaging()

        self.cuda(rank)

        print("StyleGAN2 initialized with args: ", {"image_size": image_size, "latent_dim" :latent_dim, "mtcnn_crop_size":mtcnn_crop_size, "fmap_max":fmap_max, "network_capacity":network_capacity, "attn_layers": attn_layers, "rank":rank})

        # startup apex mixed precision
        self.fp16 = fp16
        if fp16:
            (self.G, self.D, self.GE, self.a_net, self.p_net, self.d_patch, self.vgg, self.face_id), (self.G_opt, self.D_opt) = amp.initialize(
                [self.G, self.D, self.GE, self.a_net, self.p_net, self.d_patch, self.vgg, self.face_id], [self.G_opt, self.D_opt], opt_level='O1', num_losses=3)

    def _init_weights(self):
        for m in self.modules():
            # default pytorch is kaiming uniform instead of normal
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(
                    m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        for block in self.G.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(
                    old_weight, up_weight)

        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        return x


def get_d_total_loss(I_t, I_dash_s_to_t, pred_real_1, pred_fake_1, pred_real_2, pred_fake_2, d_patch):
    # where d_patch is an nn.module

    # GAN_d_loss1
    gan_d_loss_1 = hinge_loss(pred_real_1, pred_fake_1)
    # GAN_d_loss2
    gan_d_loss_2 = hinge_loss(pred_real_2, pred_fake_2)
    # patch loss
    patch_loss = get_patch_loss(I_dash_s_to_t, I_t, d_patch)

    d_total_loss = gan_d_loss_1 + gan_d_loss_2 + \
        torch.mul(patch_loss, -1)  # DPatch will maximize patch_loss
    return d_total_loss


def get_g_total_loss(I_s, I_t, I_dash_s, I_dash_s_to_t, fake_output_1, real_output_1, fake_output_2, real_output_2, vgg_model, face_id_model, d_patch_model, mtcnn_crop_size):

    weight_l1 = 1
    weight_vgg = 1
    weight_face = 1
    weight_gan = 1
    weight_patch = 1

    # GAN_d_loss1
    gan_g_loss_1 = gen_hinge_loss(fake_output_1, real_output_1)
    # GAN_d_loss2
    gan_g_loss_2 = gen_hinge_loss(fake_output_2, real_output_2)

    patch_loss = get_patch_loss(I_dash_s_to_t, I_t, d_patch_model)

    rec_loss_1 = weight_l1 * get_l1_loss(I_dash_s, I_s) + \
        weight_vgg * get_perceptual_vgg_loss(vgg_model, I_dash_s, I_s)  # + \
    #weight_face * get_face_id_loss(I_dash_s, I_s, face_id_model, crop_size=mtcnn_crop_size)

    rec_loss_2 = weight_l1 * get_l1_loss(I_dash_s_to_t, I_t) + \
        weight_vgg * \
        get_perceptual_vgg_loss(vgg_model, I_dash_s_to_t, I_t)  # + \
    # weight_face * get_face_id_loss(I_dash_s_to_t, I_t, face_id_model, crop_size=mtcnn_crop_size)



    g_loss_total = rec_loss_1 + rec_loss_2 + \
        gan_g_loss_1 + gan_g_loss_2 + patch_loss
    return g_loss_total


class Trainer():
    def __init__(
        self,
        name='default',
        results_dir='results',
        models_dir='models',
        base_dir='./',
        image_size=128,
        network_capacity=16,
        fmap_max=512,
        transparent=False,
        batch_size=4,
        mixed_prob=0.9,
        gradient_accumulate_every=1,
        lr=2e-4,
        lr_mlp=0.1,
        ttur_mult=2,
        rel_disc_loss=False,
        num_workers=None,
        save_every=500,
        evaluate_every=500,
        num_image_tiles=4,
        trunc_psi=0.6,
        fp16=False,
        cl_reg=False,
        no_pl_reg=False,
        fq_layers=[],
        fq_dict_size=256,
        attn_layers=[],
        no_const=False,
        aug_prob=0.,
        aug_types=['translation', 'cutout'],
        top_k_training=False,
        generator_top_k_gamma=0.99,
        generator_top_k_frac=0.5,
        dual_contrast_loss=False,
        dataset_aug_prob=0.,
        calculate_fid_every=None,
        calculate_fid_num_images=12800,
        clear_fid_cache=False,
        is_ddp=False,
        rank=0,
        world_size=1,
        log=False,
        *args,
        **kwargs
    ):
        self.GAN_params = [args, kwargs]
        self.GAN = None

        self.name = name

        base_dir = Path(base_dir)
        self.base_dir = base_dir
        self.results_dir = base_dir / results_dir
        self.models_dir = base_dir / models_dir
        self.fid_dir = base_dir / 'fid' / name
        self.config_path = self.models_dir / name / '.config.json'

        assert log2(image_size).is_integer(
        ), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        self.image_size = image_size
        self.network_capacity = network_capacity
        self.fmap_max = fmap_max
        self.transparent = transparent

        self.fq_layers = cast_list(fq_layers)
        self.fq_dict_size = fq_dict_size
        self.has_fq = len(self.fq_layers) > 0

        self.attn_layers = cast_list(attn_layers)
        self.no_const = no_const

        self.aug_prob = aug_prob
        self.aug_types = aug_types

        self.lr = lr
        self.lr_mlp = lr_mlp
        self.ttur_mult = ttur_mult
        self.rel_disc_loss = rel_disc_loss
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_prob = mixed_prob

        self.num_image_tiles = batch_size
        self.evaluate_every = evaluate_every
        self.save_every = save_every
        self.steps = 0

        self.av = None
        self.trunc_psi = trunc_psi

        self.no_pl_reg = no_pl_reg
        self.pl_mean = None

        self.gradient_accumulate_every = gradient_accumulate_every

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex is not available for you to use mixed precision training'
        self.fp16 = fp16

        self.cl_reg = cl_reg

        self.d_loss = 0
        self.g_loss = 0
        self.q_loss = None
        self.last_gp_loss = None
        self.last_cr_loss = None
        self.last_fid = None

        self.pl_length_ma = EMA(0.99)
        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

        self.calculate_fid_every = calculate_fid_every
        self.calculate_fid_num_images = calculate_fid_num_images
        self.clear_fid_cache = clear_fid_cache

        self.top_k_training = top_k_training
        self.generator_top_k_gamma = generator_top_k_gamma
        self.generator_top_k_frac = generator_top_k_frac

        self.dual_contrast_loss = dual_contrast_loss

        assert not (
            is_ddp and cl_reg), 'Contrastive loss regularization does not work well with multi GPUs yet'
        self.is_ddp = is_ddp
        self.is_main = rank == 0
        self.rank = rank
        self.world_size = world_size

        h_params = {'image_size': self.image_size, 
                    'network_capacity': self.network_capacity,
                    "fmap_max":self.fmap_max,
                    "batch_size": self.batch_size,
                    "gradient_accumulate_every":self.gradient_accumulate_every,
                    "lr":self.lr,
                    "fp16":self.fp16
                    }

        self.logger = wandb.init(project="stylegan2-edit", config=h_params) if log and self.is_main else None

    @property
    def image_extension(self):
        return 'jpg' if not self.transparent else 'png'

    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)

    @property
    def hparams(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity}

    def init_GAN(self):
        args, kwargs = self.GAN_params
        self.GAN = StyleGAN2(lr=self.lr, lr_mlp=self.lr_mlp, ttur_mult=self.ttur_mult, image_size=self.image_size, network_capacity=self.network_capacity, fmap_max=self.fmap_max,
                             transparent=self.transparent, fq_layers=self.fq_layers, fq_dict_size=self.fq_dict_size, attn_layers=self.attn_layers, fp16=self.fp16, cl_reg=self.cl_reg, rank=self.rank, *args, **kwargs)

        if self.is_ddp:
            ddp_kwargs = {'device_ids': [self.rank], 'broadcast_buffers': False}

            self.G_ddp = DDP(self.GAN.G, **ddp_kwargs)
            self.D_ddp = DDP(self.GAN.D, **ddp_kwargs)
            self.D_aug_ddp = DDP(self.GAN.D_aug, **ddp_kwargs)

            # ANet and PNet initialization for DDP
            self.a_net_ddp = DDP(self.GAN.a_net, **ddp_kwargs)
            self.p_net_ddp = DDP(self.GAN.p_net, **ddp_kwargs)
            self.d_patch_ddp = DDP(self.GAN.d_patch, **ddp_kwargs)

            self.vgg_ddp = self.GAN.vgg #DDP(self.GAN.vgg, **ddp_kwargs)
            self.face_id_ddp = self.GAN.face_id # DDP(self.GAN.face_id, **ddp_kwargs)

        if exists(self.logger):
            self.logger.watch(self.GAN)
 


    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists(
        ) else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.network_capacity = config['network_capacity']
        self.transparent = config['transparent']
        self.fq_layers = config['fq_layers']
        self.fq_dict_size = config['fq_dict_size']
        self.fmap_max = config.pop('fmap_max', 512)
        self.attn_layers = config.pop('attn_layers', [])
        self.no_const = config.pop('no_const', False)
        self.lr_mlp = config.pop('lr_mlp', 0.1)
        del self.GAN
        self.init_GAN()

    def config(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity, 'lr_mlp': self.lr_mlp, 'transparent': self.transparent, 'fq_layers': self.fq_layers, 'fq_dict_size': self.fq_dict_size, 'attn_layers': self.attn_layers, 'no_const': self.no_const}

    def set_data_src(self, folder):
        self.dataset = DeepFashionDataset(
            folder, self.image_size, transparent=self.transparent, aug_prob=self.dataset_aug_prob)
        num_workers = num_workers = default(
            self.num_workers, NUM_CORES if not self.is_ddp else 0)
        sampler = DistributedSampler(
            self.dataset, rank=self.rank, num_replicas=self.world_size, shuffle=True) if self.is_ddp else None
        dataloader = data.DataLoader(self.dataset, num_workers=num_workers, batch_size=math.ceil(
            self.batch_size / self.world_size), sampler=sampler, shuffle=not self.is_ddp, drop_last=True, pin_memory=True)
        self.loader = cycle(dataloader)

        # auto set augmentation prob for user if dataset is detected to be low
        num_samples = len(self.dataset)
        if not exists(self.aug_prob) and num_samples < 1e5:
            self.aug_prob = min(0.5, (1e5 - num_samples) * 3e-6)
            print(
                f'autosetting augmentation probability to {round(self.aug_prob * 100)}%')

    def train(self):
        assert exists(
            self.loader), 'You must first initialize the data source with `.set_data_src(<folder of images>)`'

        if not exists(self.GAN):  # PNet ANet initializations coupled with self.GAN
            self.init_GAN()

        self.GAN.train()
        total_disc_loss = torch.tensor(0.).cuda(self.rank)
        total_gen_loss = torch.tensor(0.).cuda(self.rank)

        batch_size = math.ceil(self.batch_size / self.world_size)

        image_size = self.GAN.G.image_size
        latent_dim = self.GAN.G.latent_dim
        num_layers = self.GAN.G.num_layers

        aug_prob = self.aug_prob
        aug_types = self.aug_types
        aug_kwargs = {'prob': aug_prob, 'types': aug_types}

        apply_gradient_penalty = self.steps % 4 == 0
        apply_path_penalty = not self.no_pl_reg and self.steps > 5000 and self.steps % 32 == 0
        apply_cl_reg_to_generated = self.steps > 20000

        G = self.GAN.G if not self.is_ddp else self.G_ddp
        D = self.GAN.D if not self.is_ddp else self.D_ddp
        D_aug = self.GAN.D_aug if not self.is_ddp else self.D_aug_ddp

        p_net = self.GAN.p_net if not self.is_ddp else self.p_net_ddp
        a_net = self.GAN.a_net if not self.is_ddp else self.a_net_ddp
        d_patch = self.GAN.d_patch if not self.is_ddp else self.d_patch_ddp

        vgg_model = self.GAN.vgg if not self.is_ddp else self.vgg_ddp
        face_id_model = self.GAN.face_id if not self.is_ddp else self.face_id_ddp
        mtcnn_crop_size = self.GAN.mtcnn_crop_size

        backwards = partial(loss_backwards, self.fp16)

        # setup losses

        if not self.dual_contrast_loss:
            D_loss_fn = hinge_loss
            G_loss_fn = gen_hinge_loss
            G_requires_reals = False
        else:
            D_loss_fn = dual_contrastive_loss
            G_loss_fn = dual_contrastive_loss
            G_requires_reals = True

        # train discriminator

        avg_pl_length = self.pl_mean
        self.GAN.D_opt.zero_grad()

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[D_aug, G, a_net, p_net, d_patch]):

            noise = image_noise(batch_size, image_size, device=self.rank)

            # Get batch inputs
            (I_s, S_pose_map, S_texture_map), (I_t, T_pose_map) = next(self.loader)
            I_s = I_s.cuda(self.rank)
            S_pose_map = S_pose_map.cuda(self.rank)
            S_texture_map = S_texture_map.cuda(self.rank)
            I_t = I_t.cuda(self.rank)
            T_pose_map = T_pose_map.cuda(self.rank)

            # Get encodings
            E_s = p_net(S_pose_map)
            E_t = p_net(T_pose_map)
            z_s = a_net(S_texture_map).expand(-1, num_layers, -1)

            # Generate I_dash_s
            I_dash_s = G(z_s, noise, E_s)  # I_dash_s
            fake_output_1, fake_q_loss_1 = D_aug(
                I_dash_s.clone().detach(), detach=True, **aug_kwargs)

            I_s.requires_grad_()  # keep
            real_output_1, real_q_loss_1 = D_aug(I_s, **aug_kwargs)

            real_output_loss_1 = real_output_1
            fake_output_loss_1 = fake_output_1

            # Generate I_dash_to_t
            I_dash_s_to_t = G(z_s, noise, E_t)
            fake_output_2, fake_q_loss_2 = D_aug(
                I_dash_s_to_t.clone().detach(), detach=True, **aug_kwargs)

            I_t.requires_grad_()
            # opt params are for self.D instead os self.D_aug
            real_output_2, real_q_loss_2 = D_aug(I_t, **aug_kwargs)

            real_output_loss_2 = real_output_2
            fake_output_loss_2 = fake_output_2

            if self.rel_disc_loss:
                real_output_loss_1 = real_output_loss_1 - \
                    fake_output_1.mean()  # does this need brackets?
                fake_output_loss_1 = fake_output_loss_1 - real_output_1.mean()

                real_output_loss_2 = real_output_loss_2 - fake_output_2.mean()
                fake_output_loss_2 = fake_output_loss_2 - real_output_2.mean()

            # divergence = D_loss_fn(real_output_loss, fake_output_loss)
            divergence = get_d_total_loss(I_t, I_dash_s_to_t, real_output_loss_1,
                                          fake_output_loss_1, real_output_loss_2, fake_output_loss_2, d_patch)
            disc_loss = divergence

            if self.has_fq:
                quantize_loss = (fake_q_loss_1 + real_q_loss_1).mean()
                self.q_loss = float(quantize_loss.detach().item())

                disc_loss = disc_loss + quantize_loss

            if apply_gradient_penalty:
                gp = gradient_penalty(I_s, real_output_1)
                self.last_gp_loss = gp.clone().detach().item()  # remove item
                self.track(self.last_gp_loss, 'GP')
                disc_loss = disc_loss + gp

            if self.has_fq:
                quantize_loss = (fake_q_loss_2 + real_q_loss_2).mean()
                self.q_loss = float(quantize_loss.detach().item())

                disc_loss = disc_loss + quantize_loss

            if apply_gradient_penalty:
                gp = gradient_penalty(I_t, real_output_2)
                self.last_gp_loss = gp.clone().detach().item()
                self.track(self.last_gp_loss, 'GP')
                disc_loss = disc_loss + gp

            disc_loss = disc_loss / self.gradient_accumulate_every
            disc_loss.register_hook(raise_if_nan)
            backwards(disc_loss, self.GAN.D_opt, loss_id=1)

            total_disc_loss += divergence.detach().item() / self.gradient_accumulate_every

        self.d_loss = float(total_disc_loss)
        self.track(self.d_loss, 'D')

        self.GAN.D_opt.step()

        # train generator

        self.GAN.G_opt.zero_grad()

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[G, D_aug, a_net, p_net, d_patch]):

            (I_s, S_pose_map, S_texture_map), (I_t, T_pose_map) = next(self.loader)
            I_s = I_s.cuda(self.rank)
            S_pose_map = S_pose_map.cuda(self.rank)
            S_texture_map = S_texture_map.cuda(self.rank)
            I_t = I_t.cuda(self.rank)
            T_pose_map = T_pose_map.cuda(self.rank)

            noise = image_noise(batch_size, image_size, device=self.rank)

            # Get encodings
            E_s = p_net(S_pose_map)
            E_t = p_net(T_pose_map)
            z_s = a_net(S_texture_map).expand(-1, num_layers, -1)

            I_dash_s = G(z_s, noise, E_s)  # I_dash_s
            fake_output_1, _ = D_aug(I_dash_s, **aug_kwargs)
            fake_output_loss_1 = fake_output_1

            real_output_1 = None

            I_dash_s_to_t = G(z_s, noise, E_t)  # I_dash_s
            fake_output_2, _ = D_aug(I_dash_s_to_t, **aug_kwargs)
            fake_output_loss_2 = fake_output_2

            real_output_2 = None

            if G_requires_reals:
                image_batch_I_s = I_s
                real_output_1, _ = D_aug(
                    image_batch_I_s, detach=True, **aug_kwargs)
                real_output_1 = real_output_1.detach()

                image_batch_I_t = I_t
                real_output_2, _ = D_aug(
                    image_batch_I_t, detach=True, **aug_kwargs)
                real_output_2 = real_output_2.detach()

            if self.top_k_training:
                epochs = (self.steps * batch_size *
                          self.gradient_accumulate_every) / len(self.dataset)
                k_frac = max(self.generator_top_k_gamma **
                             epochs, self.generator_top_k_frac)
                k = math.ceil(batch_size * k_frac)

                if k != batch_size:
                    fake_output_loss_1, _ = fake_output_loss_1.topk(
                        k=k, largest=False)
                    fake_output_loss_2, _ = fake_output_loss_2.topk(
                        k=k, largest=False)

            # loss = G_loss_fn(fake_output_loss, real_output)
            loss = get_g_total_loss(I_s, I_t, I_dash_s, I_dash_s_to_t, fake_output_1, real_output_1,
                                    fake_output_2, real_output_2, vgg_model, face_id_model, d_patch, mtcnn_crop_size)
            gen_loss = loss

            if apply_path_penalty:
                pl_lengths = calc_pl_lengths(z_s, I_dash_s)
                avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

                if not is_empty(self.pl_mean):
                    pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
                    if not torch.isnan(pl_loss):
                        gen_loss = gen_loss + pl_loss

            gen_loss = gen_loss / self.gradient_accumulate_every
            gen_loss.register_hook(raise_if_nan)
            backwards(gen_loss, self.GAN.G_opt, loss_id=2)

            total_gen_loss += loss.detach().item() / self.gradient_accumulate_every

        self.g_loss = float(total_gen_loss)
        self.track(self.g_loss, 'G')

        self.GAN.G_opt.step()

        # calculate moving averages

        if apply_path_penalty and not np.isnan(avg_pl_length):
            self.pl_mean = self.pl_length_ma.update_average(
                self.pl_mean, avg_pl_length)
            self.track(self.pl_mean, 'PL')

        if self.is_main and self.steps % 10 == 0 and self.steps > 20000:
            self.GAN.EMA()

        if self.is_main and self.steps <= 25000 and self.steps % 1000 == 2:
            self.GAN.reset_parameter_averaging()

        # save from NaN errors

        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(
                f'NaN detected for generator or discriminator. Loading from checkpoint #{self.checkpoint_num}')
            self.load(self.checkpoint_num)
            raise NanException

        # periodically save results

        if self.is_main:
            if self.steps % self.save_every == 0:
                self.save(self.checkpoint_num)

            if self.steps % self.evaluate_every == 0 or (self.steps % 125 == 0 and self.steps < 2500):
                print("Evaluating and saving images")
                self.evaluate(floor(self.steps / self.evaluate_every))

            if exists(self.calculate_fid_every) and self.steps % self.calculate_fid_every == 0 and self.steps != 0:
                print("Calculating FIDs")
                num_batches = math.ceil(
                    self.calculate_fid_num_images / self.batch_size)
                fid = self.calculate_fid(num_batches)
                self.last_fid = fid

                with open(str(self.results_dir / self.name / f'fid_scores.txt'), 'a') as f:
                    f.write(f'{self.steps},{fid}\n')

        self.steps += 1
        self.av = None

    @torch.no_grad()
    def evaluate(self, num=0, trunc=1.0):

        self.GAN.eval()
        ext = self.image_extension
        num_rows = self.num_image_tiles

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers


        # Get batch inputs
        (I_s, S_pose_map, S_texture_map), (I_t, T_pose_map) = next(self.loader)
        I_s = I_s.cuda(self.rank)
        S_pose_map = S_pose_map.cuda(self.rank)
        S_texture_map = S_texture_map.cuda(self.rank)
        I_t = I_t.cuda(self.rank)
        T_pose_map = T_pose_map.cuda(self.rank)

        batch_size = I_t.shape[0]

        # Get encodings
        E_t = self.GAN.p_net(T_pose_map)
        z_s = self.GAN.a_net(S_texture_map).expand(-1, num_layers, -1)

        noise = image_noise(batch_size, image_size, device=self.rank)

        # regular

        generated_images = self.generate_truncated(self.GAN.G, z_s, noise, E_t)
        torchvision.utils.save_image(generated_images, str(
            self.results_dir / self.name / f'{str(num)}.{ext}'), nrow=batch_size)

        images = wandb.Image(generated_images, caption="Generations Regular")
        self.track(images, "generations_regular")

        # moving averages

        generated_images = self.generate_truncated(self.GAN.GE, z_s, noise, E_t)
        torchvision.utils.save_image(generated_images, str(
            self.results_dir / self.name / f'{str(num)}-ema.{ext}'), nrow=batch_size)

        images = wandb.Image(generated_images, caption="Generations EMA")
        self.track(images, "generations_ema")

        """
        Don't need mixed regularities
        """
        # # mixing regularities

        # def tile(a, dim, n_tile):
        #     init_dim = a.size(dim)
        #     repeat_idx = [1] * a.dim()
        #     repeat_idx[dim] = n_tile
        #     a = a.repeat(*(repeat_idx))
        #     order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda(self.rank)
        #     return torch.index_select(a, dim, order_index)

        # nn = noise(num_rows, latent_dim, device=self.rank)
        # tmp1 = tile(nn, 0, num_rows)
        # tmp2 = nn.repeat(num_rows, 1)

        # tt = int(num_layers / 2)
        # mixed_latents = [(tmp1, tt), (tmp2, num_layers - tt)]

        # generated_images = self.generate_truncated(self.GAN.GE, mixed_latents, n, )
        # torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-mr.{ext}'), nrow=num_rows)

    @torch.no_grad()
    def calculate_fid(self, num_batches):
        from pytorch_fid import fid_score
        torch.cuda.empty_cache()

        real_path = self.fid_dir / 'real'
        fake_path = self.fid_dir / 'fake'

        # Get batch inputs
        (I_s, S_pose_map, S_texture_map), (I_t, T_pose_map) = next(self.loader)
        I_s = I_s.cuda(self.rank)
        S_pose_map = S_pose_map.cuda(self.rank)
        S_texture_map = S_texture_map.cuda(self.rank)
        I_t = I_t.cuda(self.rank)
        T_pose_map = T_pose_map.cuda(self.rank)

        # remove any existing files used for fid calculation and recreate directories

        if not real_path.exists() or self.clear_fid_cache:
            rmtree(real_path, ignore_errors=True)
            os.makedirs(real_path)

            for batch_num in tqdm(range(num_batches), desc='calculating FID - saving reals'):
                for k, image in enumerate(I_t.unbind(0)):
                    filename = str(k + batch_num * self.batch_size)
                    torchvision.utils.save_image(
                        image, str(real_path / f'{filename}.png'))

        # generate a bunch of fake images in results / name / fid_fake

        rmtree(fake_path, ignore_errors=True)
        os.makedirs(fake_path)

        self.GAN.eval()
        ext = self.image_extension

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # Get encodings
        E_t = self.GAN.p_net(T_pose_map)
        z_s = self.GAN.a_net(S_texture_map).expand(-1, num_layers, -1)

        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
            # latents and noise
            noise = image_noise(self.batch_size, image_size, device=self.rank)

            # moving averages
            generated_images = self.generate_truncated(
                self.GAN.GE, z_s, noise, E_t)

            for j, image in enumerate(generated_images.unbind(0)):
                torchvision.utils.save_image(image, str(
                    fake_path / f'{str(j + batch_num * self.batch_size)}-ema.{ext}'))

        return fid_score.calculate_fid_given_paths([str(real_path), str(fake_path)], 256, noise.device, 2048)

    @torch.no_grad()
    def truncate_style(self, tensor, trunc_psi=0.75):
        S = self.GAN.S
        batch_size = self.batch_size
        latent_dim = self.GAN.G.latent_dim

        if not exists(self.av):
            z = noise(2000, latent_dim, device=self.rank)
            samples = evaluate_in_chunks(batch_size, S, z).cpu().numpy()
            self.av = np.mean(samples, axis=0)
            self.av = np.expand_dims(self.av, axis=0)

        av_torch = torch.from_numpy(self.av).cuda(self.rank)
        tensor = trunc_psi * (tensor - av_torch) + av_torch
        return tensor

    @torch.no_grad()
    def truncate_style_defs(self, w, trunc_psi=0.75):
        w_space = []
        for tensor, num_layers in w:
            tensor = self.truncate_style(tensor, trunc_psi=trunc_psi)
            w_space.append((tensor, num_layers))
        return w_space

    @torch.no_grad()
    def generate_truncated(self, G, z_s, noi, E, trunc_psi=0.75, num_image_tiles=8):
        generated_images = evaluate_in_chunks(self.batch_size, G, z_s, noi, E)
        return generated_images.clamp_(0., 1.)

    @torch.no_grad()
    def generate_interpolation(self, num=0, num_image_tiles=8, trunc=1.0, num_steps=100, save_frames=False):
        self.GAN.eval()
        ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        latents_low = noise(num_rows ** 2, latent_dim, device=self.rank)
        latents_high = noise(num_rows ** 2, latent_dim, device=self.rank)
        n = image_noise(num_rows ** 2, image_size, device=self.rank)

        ratios = torch.linspace(0., 8., num_steps)

        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            latents = [(interp_latents, num_layers)]
            generated_images = self.generate_truncated(
                self.GAN.SE, self.GAN.GE, latents, n, trunc_psi=self.trunc_psi)
            images_grid = torchvision.utils.make_grid(
                generated_images, nrow=num_rows)
            pil_image = transforms.ToPILImage()(images_grid.cpu())

            if self.transparent:
                background = Image.new("RGBA", pil_image.size, (255, 255, 255))
                pil_image = Image.alpha_composite(background, pil_image)

            frames.append(pil_image)

        frames[0].save(str(self.results_dir / self.name /
                           f'{str(num)}.gif'), save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)

        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind)}.{ext}'))

    def print_log(self):
        data = [
            ('G', self.g_loss),
            ('D', self.d_loss),
            ('GP', self.last_gp_loss),
            ('PL', self.pl_mean),
            ('CR', self.last_cr_loss),
            ('Q', self.q_loss),
            ('FID', self.last_fid)
        ]

        data = [d for d in data if exists(d[1])]
        log = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.2f}', data))
        print(log)

    def track(self, value, name):
        if not exists(self.logger):
            return
        self.logger.log({name: value})

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(str(self.models_dir / self.name), True)
        rmtree(str(self.results_dir / self.name), True)
        rmtree(str(self.fid_dir), True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    def save(self, num):
        save_data = {
            'GAN': self.GAN.state_dict(),
            'version': __version__
        }

        if self.GAN.fp16:
            save_data['amp'] = amp.state_dict()

        torch.save(save_data, self.model_name(num))
        self.write_config()

    def load(self, num=-1):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [p for p in Path(
                self.models_dir / self.name).glob('model_*.pt')]
            saved_nums = sorted(
                map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every

        load_data = torch.load(self.model_name(name))

        if 'version' in load_data:
            print(f"loading from version {load_data['version']}")

        try:
            self.GAN.load_state_dict(load_data['GAN'])
        except Exception as e:
            print('unable to load save model. please try downgrading the package to the version specified by the saved model')
            raise e
        if self.GAN.fp16 and 'amp' in load_data:
            amp.load_state_dict(load_data['amp'])


class ModelLoader:
    def __init__(self, *, base_dir, name='default', load_from=-1):
        self.model = Trainer(name=name, base_dir=base_dir)
        self.model.load(load_from)

    def noise_to_styles(self, noise, trunc_psi=None):
        noise = noise.cuda()
        w = self.model.GAN.SE(noise)
        if exists(trunc_psi):
            w = self.model.truncate_style(w)
        return w

    def styles_to_images(self, w):
        batch_size, *_ = w.shape
        num_layers = self.model.GAN.GE.num_layers
        image_size = self.model.image_size
        w_def = [(w, num_layers)]

        w_tensors = styles_def_to_tensor(w_def)
        noise = image_noise(batch_size, image_size, device=0)

        images = self.model.GAN.GE(w_tensors, noise)
        images.clamp_(0., 1.)
        return images
