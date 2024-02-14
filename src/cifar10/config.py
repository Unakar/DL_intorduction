class Config:
    # 模型路径及是否预训练
    SAVE_MODEL_PATH = 'cifar_10_model.pth'
    SAVE_FINAL_MODEL_PATH = 'cifar_10_final_model.pth'
    PRETRAINED_MODEL_PATH = 'cifar_10_model.pth'
    RESNET_PRETRAINED_PATH = 'resnet18-f37072fd.pth'
    PRETRAINED = False

    # 数据集路径
    DATASET_PATH = 'data'

    # 训练参数
    EPOCHS = 12
    BATCH_SIZE = 64
    LEARNING_RATE = 0.002
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001
    
    STEP_LR_STEP_SIZE = 5
    STEP_LR_GAMMA = 0.5

    RANDOM_SEED = 214