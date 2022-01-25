import skimage
import torch


class WaveSource(torch.nn.Module):
    def __init__(self, x, y, dim=0):
        super().__init__()

        self.register_buffer('x', torch.tensor(x, dtype=torch.int64))
        self.register_buffer('y', torch.tensor(y, dtype=torch.int64))
        self.register_buffer('dim', torch.tensor(dim, dtype=torch.int32))

    def forward(self, B, Bt):
        B = B.clone()
        B[self.dim, self.x, self.y] = B[self.dim, self.x, self.y] + Bt
        return B

    def coordinates(self):
        return self.x.cpu().numpy(), self.y.cpu().numpy()


class WaveLineSource(WaveSource):
    def __init__(self, r0, c0, r1, c1, dim=0):
        x, y = skimage.draw.line(r0, c0, r1, c1)

        self.r0 = r0
        self.c0 = c0
        self.r1 = r1
        self.c1 = c1
        super().__init__(x, y, dim)


