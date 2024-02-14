import torch
import random
import os
import numpy as np

def seed_torch(seed=214):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
