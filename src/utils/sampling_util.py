import numpy as np

class rw_sampling(object):
    def __init__(self, diffusion_state):
        self.diffusion = diffusion_state

    def sampling(self, idx, num_pos, num_neg):
        print("Start sampling...")
        left_idx = []
        right_idx = []
        target = []
        for i in idx:
            p = self.diffusion[i].copy()
            if p[i] != 1:
                p[i] = 0
            p = p / np.sum(p)
            diffusion = self.diffusion[i].copy()
            diffusion = np.clip(diffusion, 1e-3, None)
            negp = (1. / diffusion)
            negp[i] = 0
            negp = negp / np.sum(negp)
            posid = np.random.choice(len(idx), num_pos, p=p)
            negid = np.random.choice(len(idx), num_neg, p=negp)
            left_idx.extend([i] * (len(posid) + len(negid)))
            right_idx.extend(posid)
            target.extend([1.] * len(posid))
            right_idx.extend(negid)
            target.extend([0.] * len(negid))
        return left_idx, right_idx, target
