DATA_FILE = "output2.csv"
DATA_LENGTH = 10000
INPUT_DIM = 7 if DATA_FILE == "output2.csv" else 8
OUTPUT_DIM = 2
TRAINING_RATIO = 0.7
SEED = 42
IS_WITH_TENSORBOARD = True

BATCH_SIZE = 128
NUM_EPOCHS = 200
LEARNING_RATE = 0.001


### transformer parameters ###
# 以下参数都**可调整**

'''
一定要比数据的维度高。
建议增大。
'''
D_MODEL = 64
'''
一定要能整除`D_MODEL`。
此参数建议保持适中。
`D_MODEL` / `NHEAD`(商) 建议比数据的维度高，比此参数高，建议不要太小。
'''
NHEAD = 2
'''
增大此参数会很快地使模型变复杂。
建议在1到6之间。
可以与`NUM_DECODER_LAYERS`不同。
'''
NUM_ENCODER_LAYERS = 1
'''
增大此参数会很快地使模型变复杂。
建议在1到6之间。
可以与`NUM_ENCODER_LAYERS`不同。
'''
NUM_DECODER_LAYERS = 1
'''
一定要比`D_MODEL`高。
建议增大。
'''
DIM_FEEDFORWARD = 256
'''
建议在0.1到0.5之间。模型更复杂，建议此参数增大.
'''
DROPOUT = 0.1