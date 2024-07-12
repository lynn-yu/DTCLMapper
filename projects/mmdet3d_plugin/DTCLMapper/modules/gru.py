import torch
import torch.nn as nn
from IPython import embed
from mmdet.models import NECKS
from mmcv.cnn.utils import kaiming_init, constant_init


@NECKS.register_module()
class ConvGRU(nn.Module):
    def __init__(self, out_channels):
        super(ConvGRU, self).__init__()
        kernel_size = 1
        padding = kernel_size // 2
        self.convz = nn.Conv2d(2*out_channels, 
            out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.convr = nn.Conv2d(2*out_channels, 
            out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.convq = nn.Conv2d(2*out_channels, 
            out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.ln = nn.LayerNorm(out_channels)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, h, x):
        if len(h.shape) == 3:
            h = h.unsqueeze(0)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        hx = torch.cat([h, x], dim=1) # [1, 2c, h, w]
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        new_x = torch.cat([r * h, x], dim=1) # [1, 2c, h, w]
        q = self.convq(new_x)

        out = ((1 - z) * h + z * q)#.squeeze(0) # (1, C, H, W)
        out = self.ln(out.permute(0,2, 3, 1)).permute(0,3, 1, 2).contiguous()
        return out


class NewGRU(nn.Module):
    def __init__(self, out_channels):
        super(NewGRU, self).__init__()
        kernel_size = 3
        padding = kernel_size // 2
        # self.conv0 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, bias=False)
        # self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv0 = nn.Sequential(nn.Conv2d(out_channels, 64, kernel_size=1, padding=0, bias=False),nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),nn.ReLU(),
                                  nn.Conv2d(64, out_channels, kernel_size=1, padding=0, bias=False)
                                  )
        self.relu = nn.ReLU()
        #self.conv1 = nn.Conv2d(out_channels*2, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.avg_pooling_1 = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pooling_1 = torch.nn.MaxPool2d((200,100))
        self.fc_transform_1 = nn.Sequential(nn.Linear(256, 256),nn.ReLU())
        #self.fc_transform_2 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())

        #self.softmax = nn.Softmax(dim=-1)
    def forward(self, pre, x):

        pre = self.conv0(pre)
        h = x-pre
        # h = torch.sigmoid(self.conv1(h))
        # out = x+h*pre

        b,c,hh,ww = x.shape
        # pre = self.relu(pre+self.conv0(pre))
        # h = torch.cat((x,pre),dim=1)
        # h = self.conv1(h)
        # out = x+0.1*h

        # h = self.avg_pooling_1(h)#4 256
        # h1 = self.fc_transform_1(h.squeeze(-1).squeeze(-1))
        # h2 = self.fc_transform_2(h.squeeze(-1).squeeze(-1))
        # h = torch.cat((h1.unsqueeze(-1),h2.unsqueeze(-1)),dim=2)#4 256 2
        # h = self.softmax(h)#4 256 2
        # h = h.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,hh,ww)
        # out = h[:,:,0]*x + h[:,:,1]*pre

        h1 = self.avg_pooling_1(h)
        h2= self.max_pooling_1(h)
        h1= self.fc_transform_1(h1.squeeze(-1).squeeze(-1))
        h2 = self.fc_transform_1(h2.squeeze(-1).squeeze(-1))#4 256
        h = self.relu(h1+h2)
        h = h.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, hh, ww)
        a = torch.ones_like(h)
        out = (a+h)*x#+(a+h)*pre
        return out


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        # conv则是实际进行的卷积操作，注意这里步长设置为卷积核大小，因为与该卷积核进行卷积操作的特征图是由输出特征图中每个点扩展为其对应卷积核那么多个点后生成的。
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        # p_conv是生成offsets所使用的卷积，输出通道数为卷积核尺寸的平方的2倍，代表对应卷积核每个位置横纵坐标都有偏移量。
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation  # modulation是可选参数,若设置为True,那么在进行卷积操作时,对应卷积核的每个位置都会分配一个权重。
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        # 由于卷积核中心点位置是其尺寸的一半，于是中心点向左（上）方向移动尺寸的一半就得到起始点，向右（下）方向移动另一半就得到终止点
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        # p0_y、p0_x就是输出特征图每点映射到输入特征图上的纵、横坐标值。
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    # 输出特征图上每点（对应卷积核中心）加上其对应卷积核每个位置的相对（横、纵）坐标后再加上自学习的（横、纵坐标）偏移量。
    # p0就是将输出特征图每点对应到卷积核中心，然后映射到输入特征图中的位置；
    # pn则是p0对应卷积核每个位置的相对坐标；
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        # 计算双线性插值点的4邻域点对应的权重
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


