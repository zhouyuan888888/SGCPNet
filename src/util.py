import json
import os
import torch

def compute_params(model):
    n_total_params = 0
    for name, m in model.named_parameters():
        n_elem = m.numel()
        n_total_params += n_elem
    return n_total_params

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Saver():
    def __init__(self, args, ckpt_dir, best_val=0, condition=lambda x,y: x > y):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        with open('{}/args.json'.format(ckpt_dir), 'w') as f:
            json.dump({k:v for k,v in args.items() if isinstance(v, (int, float, str))}, f,
                      sort_keys = True, indent = 4, ensure_ascii = False)
        self.ckpt_dir = ckpt_dir
        self.best_val = best_val
        self.condition = condition
        self._counter = 0

    def _do_save(self, new_val):
        return self.condition(new_val, self.best_val)

    def save(self, new_val, dict_to_save, logger):
        self._counter += 1
        if self._do_save(new_val):
            logger.info(" New best value {:.4f}, was {:.4f}".format(new_val, self.best_val))
            self.best_val = new_val
            dict_to_save['best_val'] = new_val
            torch.save(dict_to_save, '{}/checkpoint.pth.tar'.format(self.ckpt_dir))
            return True
        return False
