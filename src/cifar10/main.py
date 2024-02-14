from model import Baseline_Net, BnModel, KaimingInitModel, DoubleChannelModel, BetterBaselineModel, ResNet
from trainer import Trainer
from config import Config
from utils import seed_torch
	
def main():
    seed_torch(Config.RANDOM_SEED)
    model = ResNet()
    trainer = Trainer(model)
    trainer.train()

if __name__ == '__main__':  
    main()