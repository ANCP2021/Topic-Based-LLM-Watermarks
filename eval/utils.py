
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def is_important_word(pos_tag):
    return pos_tag.startswith(('N'))
