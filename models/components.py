import torch
import torch.nn as nn
import clip
import math
import torch.nn.functional as F
from .pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal embedding for time step.
    """
    def __init__(self, dim, scale=1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, time):
        time = time * self.scale
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1 + 1e-5)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time.unsqueeze(-1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

    def __len__(self):
        return self.dim


class TextEncoder(nn.Module):
    """_summary_
    Text Encoder class.
    """
    def __init__(self, device):
        super(TextEncoder, self).__init__()
        self.device = device
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
    def forward(self, affordance_text):
        """_summary_
        affordance_text can be a single string or a list of strings.
        """
        tokens = clip.tokenize(affordance_text).to(self.device)
        text_features = self.clip_model.encode_text(tokens).to(self.device).to(torch.float32)
        return text_features


class PointNetPlusPlus(nn.Module):
    """_summary_
    PointNet++ class.
    """
    def __init__(self):
        super(PointNetPlusPlus, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [
                                             32, 64, 128], 3, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(
            128, [0.4, 0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)

        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=134, mlp=[128, 128])

        self.conv1 = nn.Conv1d(128, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)

    def forward(self, xyz):
        """_summary_
        Return point-wise features and point cloud representation.
        """
        # Set Abstraction layers
        xyz = xyz.contiguous().transpose(1, 2)
        l0_xyz = xyz
        l0_points = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        c = l3_points.squeeze()

        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat(
            [l0_xyz, l0_points], 1), l1_points)
        l0_points = self.bn1(self.conv1(l0_points))
        return l0_points, c

class ResBlock(nn.Module):
    def __init__(self, Fin, Fout, n_neurons=256):
        group_norm = True
        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        if group_norm:
            self.bn1 = nn.GroupNorm(8,n_neurons)
        else:
            self.bn1 = nn.BatchNorm1d(n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)
        if group_norm:
            self.bn2 = nn.GroupNorm(8, Fout)
        else:
            self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)
        if group_norm:
            self.ll = nn.GELU()
        else:
            self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_ll=True):
        if self.Fin == self.Fout:
            Xin = x
        else:
            Xin = self.fc3(x)
            Xin = self.ll(Xin)

        Xout = self.fc1(x)
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout

        if final_ll:
            return self.ll(Xout)
        return Xout


class BPSMLP(nn.Module):
    def __init__(self,
                 n_neurons=512,
                 in_bps=4096,
                 dtype=torch.float64,
                 **kwargs):
        super().__init__()

        self.bn1 = nn.BatchNorm1d(in_bps)
        self.rb1 = ResBlock(in_bps, n_neurons)
        self.rb2 = ResBlock(in_bps + n_neurons, n_neurons)
        self.rb3 = ResBlock(in_bps + n_neurons, 1024)

        self.dout = nn.Dropout(0.3)
        # self.sigmoid = nn.Sigmoid()

        self.dtype = dtype


    def forward(self, data, return_mean_var=False):
        """Run one forward iteration to evaluate the success probability of given grasps

        Args:
            data (dict): keys should be rot_matrix, transl, joint_conf, bps_object,

        Returns:
            p_success (tensor, batch_size*1): Probability that a grasp will be successful.
        """
        X = data

        X0 = self.bn1(X)
        X = self.rb1(X0)
        X = self.dout(X)
        X = self.rb2(torch.cat([X, X0], dim=1))
        X = self.dout(X)
        X = self.rb3(torch.cat([X, X0], dim=1))

        return X


class PoseNet(nn.Module):
    """_summary_
    ContextPoseNet class. This class is for a denoising step in the diffusion.
    """
    def __init__(self,cfg):
        super(PoseNet, self).__init__()
        grasp_dim = cfg.grasp_dim
        # scale_down = [6,4,2]
        scale_down = cfg.scale_down

        self.cloud_net0 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GroupNorm(8, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 32)
        )
        self.cloud_net3 = nn.Sequential(
            nn.Linear(32, 16),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Linear(16, scale_down[0])
        )
        self.cloud_net2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Linear(16, scale_down[1])
        )
        self.cloud_net1 = nn.Sequential(
            nn.Linear(32, 16),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Linear(16, scale_down[2])
        )
        self.cloud_influence_net3 = nn.Sequential(
            nn.Linear(scale_down[0] + scale_down[0] + grasp_dim, scale_down[0]),
            nn.GELU(),
            nn.Linear(scale_down[0], scale_down[0])
        )
        self.cloud_influence_net2 = nn.Sequential(
            nn.Linear(scale_down[1] + scale_down[1] + grasp_dim, scale_down[1]),
            nn.GELU(),
            nn.Linear(scale_down[1], scale_down[1])
        )
        self.cloud_influence_net1 = nn.Sequential(
            nn.Linear(scale_down[2] + scale_down[2] + grasp_dim, scale_down[2]),
            nn.GELU(),
            nn.Linear(scale_down[2], scale_down[2])
        )

        # self.text_net0 = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.GroupNorm(8, 256),
        #     nn.GELU(),
        #     nn.Linear(256, 128),
        #     nn.GELU(),
        #     nn.Linear(128, 32)
        # )
        # self.text_net3 = nn.Sequential(
        #     nn.Linear(32, 16),
        #     nn.GroupNorm(4, 16),
        #     nn.GELU(),
        #     nn.Linear(16, 6)
        # )
        # self.text_net2 = nn.Sequential(
        #     nn.Linear(32, 16),
        #     nn.GroupNorm(4, 16),
        #     nn.GELU(),
        #     nn.Linear(16, 4)
        # )
        # self.text_net1 = nn.Sequential(
        #     nn.Linear(32, 16),
        #     nn.GroupNorm(4, 16),
        #     nn.GELU(),
        #     nn.Linear(16, 2)
        # )
        # self.text_influence_net3 = nn.Sequential(
        #     nn.Linear(6 + 6 + grasp_dim, 6),
        #     nn.GELU(),
        #     nn.Linear(6, 6)
        # )
        # self.text_influence_net2 = nn.Sequential(
        #     nn.Linear(4 + 4 + grasp_dim, 4),
        #     nn.GELU(),
        #     nn.Linear(4, 4)
        # )
        # self.text_influence_net1 = nn.Sequential(
        #     nn.Linear(2 + 2 + grasp_dim, 2),
        #     nn.GELU(),
        #     nn.Linear(2, 2)
        # )

        self.time_net3 = SinusoidalPositionEmbeddings(dim=scale_down[0])
        self.time_net2 = SinusoidalPositionEmbeddings(dim=scale_down[1])
        self.time_net1 = SinusoidalPositionEmbeddings(dim=scale_down[2])

        self.down1 = nn.Sequential(
            nn.Linear(grasp_dim, scale_down[0]),
            nn.GELU(),
            nn.Linear(scale_down[0], scale_down[0])
        )
        self.down2 = nn.Sequential(
            nn.Linear(scale_down[0], scale_down[1]),
            nn.GELU(),
            nn.Linear(scale_down[1], scale_down[1])
        )
        self.down3 = nn.Sequential(
            nn.Linear(scale_down[1], scale_down[2]),
            nn.GELU(),
            nn.Linear(scale_down[2], scale_down[2])
        )

        self.up1 = nn.Sequential(
            nn.Linear(scale_down[2] + scale_down[1], scale_down[1]),
            nn.GELU(),
            nn.Linear(scale_down[1], scale_down[1])
        )
        self.up2 = nn.Sequential(
            nn.Linear(scale_down[1] + scale_down[0], scale_down[0]),
            nn.GELU(),
            nn.Linear(scale_down[0], scale_down[0])
        )
        self.up3 = nn.Sequential(
            nn.Linear(scale_down[0] + grasp_dim, grasp_dim),
            nn.GELU(),
            nn.Linear(grasp_dim, grasp_dim)
        )

    def forward_affordance(self, g, c, t, context_mask, _t):
        """_summary_

        Args:
            g: pose representations, size [B, 7]
            c: point cloud representations, size [B, 1024]
            t: affordance texts, size [B, 512]
            context_mask: masks {0, 1} for the contexts, size [B, 1]
            _t is for the timesteps, size [B,]
        """
        c = c * context_mask
        c0 = self.cloud_net0(c)
        c1 = self.cloud_net1(c0)
        c2 = self.cloud_net2(c0)
        c3 = self.cloud_net3(c0)

        t = t * context_mask
        t0 = self.text_net0(t)
        t1 = self.text_net1(t0)
        t2 = self.text_net2(t0)
        t3 = self.text_net3(t0)

        _t0 = _t.unsqueeze(1)
        _t1 = self.time_net1(_t0)
        _t2 = self.time_net2(_t0)
        _t3 = self.time_net3(_t0)

        g = g.float()
        g_down1 = self.down1(g) # 6
        g_down2 = self.down2(g_down1) # 4
        g_down3 = self.down3(g_down2) # 2

        c1_influence = self.cloud_influence_net1(torch.cat((c1, g, _t1), dim=1))
        t1_influence = self.text_influence_net1(torch.cat((t1, g, _t1), dim=1))
        influences1 = F.softmax(torch.cat((c1_influence.unsqueeze(1), t1_influence.unsqueeze(1)), dim=1), dim=1)
        ct1 = (c1 * influences1[:, 0, :] + t1 * influences1[:, 1, :])
        up1 = self.up1(torch.cat((g_down3 * ct1 + _t1, g_down2), dim=1))

        c2_influence = self.cloud_influence_net2(torch.cat((c2, g, _t2), dim=1))
        t2_influence = self.text_influence_net2(torch.cat((t2, g, _t2), dim=1))
        influences2 = F.softmax(torch.cat((c2_influence.unsqueeze(1), t2_influence.unsqueeze(1)), dim=1), dim=1)
        ct2 = (c2 * influences2[:, 0, :] + t2 * influences2[:, 1, :])
        up2 = self.up2(torch.cat((up1 * ct2 + _t2, g_down1), dim=1))

        c3_influence = self.cloud_influence_net3(torch.cat((c3, g, _t3), dim=1))
        t3_influence = self.text_influence_net3(torch.cat((t3, g, _t3), dim=1))
        influences3 = F.softmax(torch.cat((c3_influence.unsqueeze(1), t3_influence.unsqueeze(1)), dim=1), dim=1)
        ct3 = (c3 * influences3[:, 0, :] + t3 * influences3[:, 1, :])
        up3 = self.up3(torch.cat((up2 * ct3 + _t3, g), dim=1))  # size [B, 7]

        return up3

    def forward(self, g, c, _t, context_mask=None):
        """_summary_

        Args:
            g: pose representations, size [B, 7]
            c: point cloud representations, size [B, 1024]
            # t: affordance texts, size [B, 512]
            # context_mask: masks {0, 1} for the contexts, size [B, 1]
            _t is for the timesteps, size [B,]
        """
        if context_mask is not None:
            c = c * context_mask
            
        c0 = self.cloud_net0(c)
        c1 = self.cloud_net1(c0)
        c2 = self.cloud_net2(c0)
        c3 = self.cloud_net3(c0)

        _t0 = _t.unsqueeze(1) # [B,1]
        _t1 = self.time_net1(_t0).squeeze() # [B,1,2]
        _t2 = self.time_net2(_t0).squeeze() # [B,1,4]
        _t3 = self.time_net3(_t0).squeeze() # [B,1,6]

        g = g.float()
        g_down1 = self.down1(g) # 6
        g_down2 = self.down2(g_down1) # 4
        g_down3 = self.down3(g_down2) # 2

        simple_modify = False

        c1_influence = self.cloud_influence_net1(torch.cat((c1, g, _t1), dim=1)) # [B,2]
        # t1_influence = self.text_influence_net1(torch.cat((t1, g, _t1), dim=1))
        # influences1 = F.softmax(torch.cat((c1_influence.unsqueeze(1), t1_influence.unsqueeze(1)), dim=1), dim=1)
        # ct1 = (c1 * influences1[:, 0, :] + t1 * influences1[:, 1, :])
        # up1 = self.up1(torch.cat((g_down3 * ct1 + _t1, g_down2), dim=1))
        up1 = self.up1(torch.cat((g_down3 * c1_influence + _t1, g_down2), dim=1))

        c2_influence = self.cloud_influence_net2(torch.cat((c2, g, _t2), dim=1))
        # t2_influence = self.text_influence_net2(torch.cat((t2, g, _t2), dim=1))
        # influences2 = F.softmax(torch.cat((c2_influence.unsqueeze(1), t2_influence.unsqueeze(1)), dim=1), dim=1)
        # ct2 = (c2 * influences2[:, 0, :] + t2 * influences2[:, 1, :])
        # up2 = self.up2(torch.cat((up1 * ct2 + _t2, g_down1), dim=1))
        up2 = self.up2(torch.cat((up1 * c2_influence + _t2, g_down1), dim=1))

        c3_influence = self.cloud_influence_net3(torch.cat((c3, g, _t3), dim=1))
        # t3_influence = self.text_influence_net3(torch.cat((t3, g, _t3), dim=1))
        # influences3 = F.softmax(torch.cat((c3_influence.unsqueeze(1), t3_influence.unsqueeze(1)), dim=1), dim=1)
        # ct3 = (c3 * influences3[:, 0, :] + t3 * influences3[:, 1, :])
        # up3 = self.up3(torch.cat((up2 * ct3 + _t3, g), dim=1))  # size [B, 7]
        up3 = self.up3(torch.cat((up2 * c3_influence + _t3, g), dim=1))  # size [B, 6]

        return up3
