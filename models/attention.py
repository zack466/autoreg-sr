import torch
from torch import nn
import torch.nn.functional as F

# taken from https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py#L8
class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation=None):
        super(SelfAttention, self).__init__()
        self.channel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
        inputs :
            x : input feature maps( B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = (
            self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        )  # B X CX(N)
        proj_key = self.key_conv(x).view(
            m_batchsize, -1, width * height
        )  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # B X (N) X (N)
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width * height
        )  # B X C X N

        out = torch.bmm(
            proj_value, attention.permute(0, 2, 1)
        )  # (B C N) x (B N N) => (B C N)
        out = out.view(m_batchsize, C, width, height)  # (B C H W)

        out = self.gamma * out + x
        return out, attention


class MaskedSelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, mask_kernel=3, activation=None, device="cuda"):
        super(MaskedSelfAttention, self).__init__()
        self.channel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        ).to(device)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        ).to(device)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        ).to(device)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

        self.in_dim = in_dim
        self.mask_kernel = mask_kernel
        self.weight_maskUpdater = torch.ones(
            self.in_dim,
            self.in_dim,
            self.mask_kernel,
            self.mask_kernel,
        ).to(device)

    def forward(self, x, mask):
        """
        inputs :
            x : input feature maps (B x C x W x H)
            mask : binary mask with same shape as x
        returns :
            out : self attention value + input feature
            mask : updated mask based on convolution conditioning
        """
        x = x.masked_fill(mask.bool(), -9e9)
        m_batchsize, C, width, height = x.size()
        proj_query = (
            self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        )  # B x N x C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B x C x N
        energy = torch.bmm(proj_query, proj_key)  # B x N x N
        attention = self.softmax(energy)  # dim -1
        proj_value = self.value_conv(x).view(
            m_batchsize, -1, width * height
        )  # B X C X N

        out = torch.bmm(
            proj_value, attention.permute(0, 2, 1)
        )  # (B C N) x (B N N) => (B C N)
        out = out.view(m_batchsize, C, width, height)  # (B C H W)

        out = self.gamma * out + x

        with torch.no_grad():
            self.update_mask = F.conv2d(
                mask,
                self.weight_maskUpdater,
                bias=None,
                stride=1,
                padding=self.mask_kernel // 2,
                dilation=1,
                groups=1,
            )
            self.update_mask = torch.clamp(self.update_mask, 0, 1)

        return out, self.update_mask


class RelativePosEmbedding(nn.Module):
    def __init__(self):
        super(RelativePosEmbedding, self).__init__()


class LocalAttention2d(nn.Module):
    def __init__(self):
        super(LocalAttention2d, self).__init__()


# taken from https://github.com/leaderj1001/Attention-Augmented-Conv2d/blob/master/attention_augmented_conv.py
class AugmentedConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dk,
        dv,
        Nh,
        shape=0,
        relative=False,
        stride=1,
    ):
        super(AugmentedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.shape = shape
        self.relative = relative
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert (
            self.dk % self.Nh == 0
        ), "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert (
            self.dv % self.Nh == 0
        ), "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.conv_out = nn.Conv2d(
            self.in_channels,
            self.out_channels - self.dv,
            self.kernel_size,
            stride=stride,
            padding=self.padding,
        )

        self.qkv_conv = nn.Conv2d(
            self.in_channels,
            2 * self.dk + self.dv,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=self.padding,
        )

        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

        if self.relative:
            self.key_rel_w = nn.Parameter(
                torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True)
            )
            self.key_rel_h = nn.Parameter(
                torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True)
            )

    def forward(self, x):
        # Input x
        # (batch_size, channels, height, width)
        # batch, _, height, width = x.size()

        # conv_out
        # (batch_size, out_channels, height, width)
        conv_out = self.conv_out(x)
        batch, _, height, width = conv_out.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, height * width, dvh or dkh)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v
        # (batch_size, Nh, height, width, dv or dk)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(
            x, self.dk, self.dv, self.Nh
        )
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
        weights = F.softmax(logits, dim=-1)

        # attn_out
        # (batch, Nh, height * width, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(
            attn_out, (batch, self.Nh, self.dv // self.Nh, height, width)
        )
        # combine_heads_2d
        # (batch, out_channels, height, width)
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)

        return torch.cat((conv_out, attn_out), dim=1)

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        N, _, H, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q *= dkh ** -0.5
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, "w")
        rel_logits_h = self.relative_logits_1d(
            torch.transpose(q, 2, 3), self.key_rel_h, W, H, Nh, "h"
        )

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum("bhxyd,md->bhxym", q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = (
                torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
            )
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1 :]
        return final_x


class Attention2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dk,
        dv,
        Nh,
        shape=0,
        relative=False,
        stride=1,
    ):
        super(Attention2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.shape = shape
        self.relative = relative
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert (
            self.dk % self.Nh == 0
        ), "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert (
            self.dv % self.Nh == 0
        ), "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.conv_out = nn.Conv2d(
            self.in_channels,
            self.out_channels - self.dv,
            self.kernel_size,
            stride=stride,
            padding=self.padding,
        )

        self.qkv_conv = nn.Conv2d(
            self.in_channels,
            2 * self.dk + self.dv,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=self.padding,
        )

        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

        if self.relative:
            self.key_rel_w = nn.Parameter(
                torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True)
            )
            self.key_rel_h = nn.Parameter(
                torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True)
            )

    def forward(self, x):
        # Input x
        # (batch_size, channels, height, width)
        # batch, _, height, width = x.size()

        # conv_out
        # (batch_size, out_channels, height, width)
        conv_out = self.conv_out(x)
        batch, _, height, width = conv_out.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, height * width, dvh or dkh)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v
        # (batch_size, Nh, height, width, dv or dk)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(
            x, self.dk, self.dv, self.Nh
        )
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
        weights = F.softmax(logits, dim=-1)

        # attn_out
        # (batch, Nh, height * width, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(
            attn_out, (batch, self.Nh, self.dv // self.Nh, height, width)
        )
        # combine_heads_2d
        # (batch, out_channels, height, width)
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)

        return attn_out
        return torch.cat((conv_out, attn_out), dim=1)

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        N, _, H, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q *= dkh ** -0.5
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, "w")
        rel_logits_h = self.relative_logits_1d(
            torch.transpose(q, 2, 3), self.key_rel_h, W, H, Nh, "h"
        )

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum("bhxyd,md->bhxym", q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = (
                torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
            )
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1 :]
        return final_x