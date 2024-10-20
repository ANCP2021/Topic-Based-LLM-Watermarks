from topic_watermark_processor import TopicWatermarkLogitsProcessor
from watermark_processor import WatermarkLogitsProcessor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList
)
import torch
from functools import partial
from nltk import pos_tag

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def is_important_word(pos_tag):
    return pos_tag.startswith(('N'))
