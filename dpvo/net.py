import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import cv2
import os

import torch_scatter
from torch_scatter import scatter_sum

from . import fastba
from . import altcorr
from . import lietorch
from .lietorch import SE3

from .extractor import BasicEncoder, BasicEncoder4
from .blocks import GradientClip, GatedResidual, SoftAgg

from .utils import *
from .ba import BA
from . import projective_ops as pops

autocast = torch.cuda.amp.autocast
import matplotlib.pyplot as plt
from . import torch_embeded

DIM = 384

class Update(nn.Module):
    def __init__(self, p):
        super(Update, self).__init__()

        self.c1 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))

        self.c2 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))
        
        self.norm = nn.LayerNorm(DIM, eps=1e-3)

        self.agg_kk = SoftAgg(DIM)
        self.agg_ij = SoftAgg(DIM)

        self.gru = nn.Sequential(
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
        )

        self.corr = nn.Sequential(
            nn.Linear(2*49*p*p, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
        )

        self.d = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip())

        self.w = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip(),
            nn.Sigmoid())


    def forward(self, net, inp, corr, flow, ii, jj, kk):
        """ update operator """

        net = net + inp + self.corr(corr)
        net = self.norm(net)

        ix, jx = fastba.neighbors(kk, jj)
        mask_ix = (ix >= 0).float().reshape(1, -1, 1)
        mask_jx = (jx >= 0).float().reshape(1, -1, 1)

        net = net + self.c1(mask_ix * net[:,ix])
        net = net + self.c2(mask_jx * net[:,jx])

        net = net + self.agg_kk(net, kk)
        net = net + self.agg_ij(net, ii*12345 + jj)

        net = self.gru(net)

        return net, (self.d(net), self.w(net), None)


class Patchifier(nn.Module):
    def __init__(self, patch_size=3):
        super(Patchifier, self).__init__()
        self.patch_size = patch_size
        self.fnet = BasicEncoder4(output_dim=128, norm_fn='instance')
        self.inet = BasicEncoder4(output_dim=DIM, norm_fn='none')

    def __image_gradient(self, images):
        gray = ((images + 0.5) * (255.0 / 2)).sum(dim=2)
        dx = gray[...,:-1,1:] - gray[...,:-1,:-1]
        dy = gray[...,1:,:-1] - gray[...,:-1,:-1]
        g = torch.sqrt(dx**2 + dy**2)
        g = F.avg_pool2d(g, 4, 4)
        return g

    def forward(self, images, patches_per_image=80, disps=None, centroid_sel_strat='RANDOM', return_color=False, total_steps=1, drawing=False):
        """ extract patches from input images """
        fmap = self.fnet(images) / 4.0
        imap = self.inet(images) / 4.0

        #print("imap.shape:  ",imap.shape)  (b,n,384,120,160)
        #print("fmap.shape:  ",fmap.shape)  (b,n,128,120,160)
        
        b, n, c, h, w = fmap.shape
        P = self.patch_size

        
        
        extractor_kpts_f=torch_embeded.LF_Net(fmap).to(device="cuda")
        extractor_kpts_i=torch_embeded.LF_Net(imap).to(device="cuda")
        coords_i=extractor_kpts_i(imap)
        coords_f=extractor_kpts_f(fmap)
        coords=torch.cat([coords_f,coords_i],dim=1).float()
        #print("coords.shape:  ",coords.shape)  #(n,number of patches,2)
        
        #print(total_steps)

        """# bias patch selection towards regions with high gradient
        if centroid_sel_strat == 'GRADIENT_BIAS':
            g = self.__image_gradient(images)
            x = torch.randint(1, w-1, size=[n, 3*patches_per_image], device="cuda")
            y = torch.randint(1, h-1, size=[n, 3*patches_per_image], device="cuda")

            coords = torch.stack([x, y], dim=-1).float()
            g = altcorr.patchify(g[0,:,None], coords, 0).view(n, 3 * patches_per_image)
            
            ix = torch.argsort(g, dim=1)
            x = torch.gather(x, 1, ix[:, -patches_per_image:])
            y = torch.gather(y, 1, ix[:, -patches_per_image:])

        elif centroid_sel_strat == 'RANDOM':
            x = torch.randint(1, w-1, size=[n, patches_per_image], device="cuda")
            y = torch.randint(1, h-1, size=[n, patches_per_image], device="cuda")

        else:
            raise NotImplementedError(f"Patch centroid selection not implemented: {centroid_sel_strat}")

        coords = torch.stack([x, y], dim=-1).float()"""


        imap = altcorr.patchify(imap[0], coords, 0).view(b, -1, DIM, 1, 1)
        #print("imap.shape after_patchify:  ", imap.shape)
        gmap = altcorr.patchify(fmap[0], coords, P//2).view(b, -1, 128, P, P)
        #print("gmap.shape after_patchify:  ", gmap.shape)
        

        if return_color:
            clr = altcorr.patchify(images[0], 4*(coords + 0.5), 0).view(b, -1, 3)

        if disps is None:
            disps = torch.ones(b, n, h, w, device="cuda")

        grid, _ = coords_grid_with_index(disps, device=fmap.device)
        
        patches = altcorr.patchify(grid[0], coords, P//2).view(b, -1, 3, P, P)
        #这个grid零只是个自己掏出来的张量，没有图像的信息的，提取补丁的位置和fmap同步应该就可以了

        index = torch.arange(n, device="cuda").view(n, 1)
        index = index.repeat(1, patches_per_image).reshape(-1)
        
        #print("patches",patches.shape)
        
        
        if return_color:
            return fmap, gmap, imap, patches, index, clr
        
        if drawing==True:
            return fmap, gmap, imap, patches, index, coords

        

        return fmap, gmap, imap, patches, index


class CorrBlock:
    def __init__(self, fmap, gmap, radius=3, dropout=0.2, levels=[1,4]):
        self.dropout = dropout
        self.radius = radius
        self.levels = levels

        self.gmap = gmap
        self.pyramid = pyramidify(fmap, lvls=levels)

    def __call__(self, ii, jj, coords):
        corrs = []
        for i in range(len(self.levels)):
            corrs += [ altcorr.corr(self.gmap, self.pyramid[i], coords / self.levels[i], ii, jj, self.radius, self.dropout) ]
        return torch.stack(corrs, -1).view(1, len(ii), -1)


class VONet(nn.Module):
    def __init__(self, use_viewer=False):
        super(VONet, self).__init__()
        self.P = 3
        self.patchify = Patchifier(self.P)
        self.update = Update(self.P)

        self.DIM = DIM
        self.RES = 4


    @autocast(enabled=False)
    def forward(self, images, poses, disps, intrinsics, M=1024, STEPS=12, P=1, structure_only=False, rescale=False, total_steps=1):
        """ Estimates SE3 or Sim3 between pair of frames """
        images_cp=images.cpu()

        images = 2 * (images / 255.0) - 0.5
        intrinsics = intrinsics / 4.0
        disps = disps[:, :, 1::4, 1::4].float()

        fmap, gmap, imap, patches, ix, coords_draw = self.patchify(images, disps=disps, patches_per_image=96, drawing=True)
        
        #把每一帧的关键点选取都画出来
        if(total_steps % 10000==0):
            folder_name = f'step_{total_steps}'
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            
            # 对于每一帧，在图像上绘制numbers个点
            for i in range(fmap.shape[1]):
                x_coords, y_coords = coords_draw[i][..., 0], coords_draw[i][..., 1]  # 解包x和y坐标
                frame = images_cp[0, i]  # 获取当前帧
                frame_cpu=frame.permute(1, 2, 0)
                #frame_cpu=frame_cpu.cpu()
                frame_cpu=frame_cpu.numpy()
                #print(frame_cpu.shape)
                mat = cv2.UMat(frame_cpu)
                for x, y in zip(x_coords, y_coords):
                    # 使用OpenCV绘制红色圆点
                    cv2.circle(mat, (int(x)*4, int(y)*4), radius=1, color=(0, 0, 255), thickness=-1)

                
                # 保存图像
                file_name = os.path.join(folder_name, f'frame_{i}.png')
                cv2.imwrite(file_name, mat)

            print(f'Images saved in folder: {folder_name}')

        corr_fn = CorrBlock(fmap, gmap)
        
        #print("fmap.shape:  ",fmap.shape) (b,n,c,h,w)
        #print("gmap.shape:  ",gmap.shape) (b,n*patches_number,c,3,3) 应该是已经补丁化了

        b, N, c, h, w = fmap.shape
        p = self.P

        patches_gt = patches.clone()
        Ps = poses

        d = patches[..., 2, p//2, p//2]
        patches = set_depth(patches, torch.rand_like(d))
        #print("patches.shape after set_depth ",patches.shape)

        kk, jj = flatmeshgrid(torch.where(ix < 8)[0], torch.arange(0,8, device="cuda"), indexing='ij')
        #print("kk_outside.shape:  ",kk.shape)
        ii = ix[kk]
        
        #print("imap.shape:  ",imap.shape)

        imap = imap.view(b, -1, DIM)
        #print("imap.shape_after_view:  ",imap.shape)
        net = torch.zeros(b, len(kk), DIM, device="cuda", dtype=torch.float)
        
        Gs = SE3.IdentityLike(poses)

        if structure_only:
            Gs.data[:] = poses.data[:]

        traj = []
        bounds = [-64, -64, w + 64, h + 64]
        
        while len(traj) < STEPS:
            Gs = Gs.detach()
            patches = patches.detach()

            #print("outside_ii.shape:  ",ii.shape)

            n = ii.max() + 1
            #print("n:  ",n)
            if len(traj) >= 8 and n < images.shape[1]:
                if not structure_only: Gs.data[:,n] = Gs.data[:,n-1]
                kk1, jj1 = flatmeshgrid(torch.where(ix  < n)[0], torch.arange(n, n+1, device="cuda"), indexing='ij')
                kk2, jj2 = flatmeshgrid(torch.where(ix == n)[0], torch.arange(0, n+1, device="cuda"), indexing='ij')

                ii = torch.cat([ix[kk1], ix[kk2], ii])  #看一下这些变量的维度
                jj = torch.cat([jj1, jj2, jj])
                kk = torch.cat([kk1, kk2, kk])

                net1 = torch.zeros(b, len(kk1) + len(kk2), DIM, device="cuda")
                net = torch.cat([net1, net], dim=1)

                """
                print("ii.shape:  ",ii.shape)
                print("jj.shape:  ",jj.shape)
                print("kk.shape:  ",kk.shape)
                print("ix[kk1].shape:  ",ix[kk1].shape)
                print("ix[kk2].shape:  ",ix[kk2].shape)
                print("jj1.shape:  ",jj1.shape)
                print("jj2.shape:  ",jj2.shape)
                print("kk1.shape:  ",kk1.shape)
                print("kk2.shape:  ",kk2.shape)
                """

                if np.random.rand() < 0.1:
                    k = (ii != (n - 4)) & (jj != (n - 4))
                    ii = ii[k]
                    jj = jj[k]
                    kk = kk[k]
                    net = net[:,k]

                patches[:,ix==n,2] = torch.median(patches[:,(ix == n-1) | (ix == n-2),2])
                n = ii.max() + 1

            coords = pops.transform(Gs, patches, intrinsics, ii, jj, kk)
            #print("coords.shape ",coords.shape)
            coords1 = coords.permute(0, 1, 4, 2, 3).contiguous()

            corr = corr_fn(kk, jj, coords1)

            #print("corr.shape:  ",corr.shape)
            #print("imap.shape:  ",imap.shape)
            #print("kk.shape:  ",kk.shape)
            #print("imap[:,kk].shape:  ",imap[:,kk].shape)

            net, (delta, weight, _) = self.update(net, imap[:,kk], corr, None, ii, jj, kk)

            lmbda = 1e-4
            target = coords[...,p//2,p//2,:] + delta

            ep = 10
            for itr in range(2):
                Gs, patches = BA(Gs, patches, intrinsics, target, weight, lmbda, ii, jj, kk, 
                    bounds, ep=ep, fixedp=1, structure_only=structure_only)

            kl = torch.as_tensor(0)
            dij = (ii - jj).abs()
            k = (dij > 0) & (dij <= 2)

            coords = pops.transform(Gs, patches, intrinsics, ii[k], jj[k], kk[k])
            coords_gt, valid, _ = pops.transform(Ps, patches_gt, intrinsics, ii[k], jj[k], kk[k], jacobian=True)
            #print("coords_gt.shape ",coords_gt.shape)

            traj.append((valid, coords, coords_gt, Gs[:,:n], Ps[:,:n], kl))

        return traj

