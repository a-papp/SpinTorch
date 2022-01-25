import torch
import skimage


class WaveProbe(torch.nn.Module):
	def __init__(self, x, y):
		super().__init__()

		self.register_buffer('x', torch.tensor(x, dtype=torch.int64))
		self.register_buffer('y', torch.tensor(y, dtype=torch.int64))

	def forward(self, m):
		return m[:,0, self.x, self.y]

	def coordinates(self):
		return self.x.cpu().numpy(), self.y.cpu().numpy()

class WaveIntensityProbe(WaveProbe):
	def __init__(self, x, y):
		super().__init__(x, y)

	def forward(self, m):
		return super().forward(m).pow(2)

class WaveIntensityProbeDisk(WaveProbe):
	def __init__(self, x, y, r):
		x, y = skimage.draw.disk((x, y), r)
		super().__init__(x, y)

	def forward(self, m):
		return super().forward(m).sum().pow(2).unsqueeze(0)
