import os
import shutil
from tensorboardX import SummaryWriter


class Logger_1:
    """Logging with TensorboardX. """
    def __init__(self, logdir, verbose=True):
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        try:
            shutil.rmtree(logdir)
        except FileNotFoundError:
            pass

        self.verbose = verbose
        ### text file
        fname = str(logdir) + ".txt"
        self.writer = open(fname, "w")#SummaryWriter(logdir)

    def log_histograms(self, dic, step):
        """Log dictionary of tensors as histograms. """
        for k, v in dic.items():
            self.writer.add_histogram(k, v, step)

    def log_scalars(self, dic, step):
        """Log dictionary of scalar values. """
        for k, v in dic.items():
            #self.writer.add_scalar(k, v, step)
            self.writer.write(' Step {}   {}  v {} \n'.format(step, k, v))

        if self.verbose:
            print(f"Step {step}, {dic}")

    def log_dowrite(self, obj):
        self.writer.write('{} \n'.format(obj))

    def close(self):
        self.writer.close()