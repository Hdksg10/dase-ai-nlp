import torch.nn.functional as F

def onehot(description, diagnosis, vocab_size=1300):
    # one-hot encodding
    desc_encoded = F.one_hot(description, vocab_size)
    diagn_encoded = F.one_hot(diagnosis, vocab_size)
    return desc_encoded, diagn_encoded