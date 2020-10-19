import tensorflow as tf
from tensorflow.python.tools import freeze_graph

def freeze():
    
    freeze_graph.freeze_graph('./graph/graph.pbtxt',
                              "",
                              False,
                              './model_logs/places2/snap-0',
                              'inpaint_net/Tanh_1',
                              "save/restore_all",
                              "save/Const",
                              './tf-lite/frozen_graph.pb',
                              True,
                              '')

    print('freeze graph done.')

    

if __name__ == '__main__':
    freeze()

