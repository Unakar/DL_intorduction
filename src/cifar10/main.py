from model import BaselineModel, BnModel, KaimingInitModel, DoubleChannelModel, BetterBaselineModel, ResNet
from trainer import Trainer
from config import Config
from utils import seed_torch
	
def main(model):
    seed_torch(Config.RANDOM_SEED)
    model = BaselineModel()
    trainer = Trainer(model)
    trainer.train()

if __name__ == '__main__':  
    main()