import os
import tensorflow as tf

# checkpoint_path=os.path.join(os.getcwd(),'ckpt1/ner.ckpt')
reader=tf.train.NewCheckpointReader("/usr/local/lib/python2.7/dist-packages/tensor2tensor/t2t_train/translate_retro_syn/transformer-transformer_base_single_gpu/-0")
var=reader.get_variable_to_shape_map()
for key in var:
    print("tensor_name",key)
    # print(reader.get_tensor(key))
