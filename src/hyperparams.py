#!/env/python

"""
This is the hyper-parameters class

Args:
    # DATASET HYPERPARAMS
    batch_size (int)  : Size of batches of data
    num_workers (int) : Number of workers

    # NEURAL NET HYPERPARAMS
    filter_size (int) : Size of the filter for CNN
    pool_size (int)   : Size of pooling layers
    More to be added
"""

class HyperParameters():
    def __init__(self,batch_size=4,num_workers=1,filter_size=3,pool_size=2):
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.filter_size=filter_size
        self.pool_size=pool_size

