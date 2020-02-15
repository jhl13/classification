# encoding = utf-8
import tensorflow as tf

def get_available_gpus(num_gpus = None):
    """
    Modified on http://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    However, the original code will occupy all available gpu memory.
    The modified code need a parameter: num_gpus. It does nothing but return the device handler name
    It will work well on single-maching-training, but I don't know whether it will work well on a cluster.
    """
    if num_gpus == None:
        from tensorflow.python.client import device_lib as _device_lib
        local_device_protos = _device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']
    else:
        return ['/gpu:%d'%(idx) for idx in range(num_gpus)]
