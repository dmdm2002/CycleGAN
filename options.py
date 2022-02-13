import datetime


class Options(object):
    def __init__(self):
        self.root = 'D:/sideproject/Cyclegan/dataset'
        self.classes = ''
        self.GAN_type = 'CycleGAN'
        self.lr = 2e-4
        self.beta_1 = 0.5

        self.OUTPUT_DIR = 'D:/sideproject/CycleGAN/{date:%Y-%m-%d_%H%M%S}/'.format(date=datetime.datetime.now())
        self.OUTPUT_DIR_CKP = f'{self.OUTPUT_DIR}/checkpoints'
        self.OUTPUT_DIR_SAMPLE = f'{self.OUTPUT_DIR}/sampling'
        self.OUTPUT_DIR_LOSS = f'{self.OUTPUT_DIR}/loss'
        self.OUTPUT_DIR_TEST = 'D:/sideproject/'

        self.epochs = 300
        self.cnt = 300
        self.batchsz = 3


class CycleGAN_Options(Options):
    def __init__(self):
        super(CycleGAN_Options, self).__init__()
        self.input_size = (224, 224, 3)