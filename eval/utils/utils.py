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

# Instatiate the WatermarkLogitsProcessor according to the watermark parameters
# and generate watermarked text by passing it to the generate method of the model
# as a logits processor.
def generate(prompt, detected_topic, args, model=None, tokenizer=None):
    
    # if set to is_topic, use topic based watermarks
    if args['is_topic']:
        watermark_processor = TopicWatermarkLogitsProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=args['gamma'],
            delta=args['delta'],
            seeding_scheme=args['seeding_scheme'],
            select_green_tokens=args['select_green_tokens'],
            topic_token_mapping=args['topic_token_mapping'],
            detected_topic=detected_topic,
        )
    else: # else use regular watermarking
        watermark_processor = WatermarkLogitsProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=args['gamma'],
            delta=args['delta'],
            seeding_scheme=args['seeding_scheme'],
            select_green_tokens=args['select_green_tokens']
        )

    gen_kwargs = dict(max_new_tokens=args['max_new_tokens'])

    if args['use_sampling']:
        gen_kwargs.update(dict(
            do_sample=True, 
            top_k=0,
            temperature=args['sampling_temp']
        ))
    else:
        gen_kwargs.update(dict(
            num_beams=args['n_beams']
        ))

    # generate with watermark using LogitsProcessorList
    generate_with_watermark = partial(
        model.generate,
        logits_processor=LogitsProcessorList([watermark_processor]), 
        **gen_kwargs
    )

    if args['prompt_max_length']:
        pass
    elif hasattr(model.config,"max_position_embedding"):
        args['prompt_max_length'] = model.config.max_position_embeddings - args['max_new_tokens']
    else:
        args['prompt_max_length'] = 2048 - args['max_new_tokens']

    tokenized_input = tokenizer(
        prompt, 
        return_tensors="pt", 
        add_special_tokens=True, 
        truncation=True, 
        max_length=args['prompt_max_length']
    ).to(device)

    torch.manual_seed(args['generation_seed'])

    # optional to seed before second generation, but will not be the same again generally, unless delta==0.0, no-op watermark
    if args['seed_separately']: 
        torch.manual_seed(args['generation_seed'])
    output_with_watermark = generate_with_watermark(**tokenized_input)

    if args['decoder']:
        # need to isolate the newly generated tokens
        output_with_watermark = output_with_watermark[:,tokenized_input["input_ids"].shape[-1]:]

    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]

    return decoded_output_with_watermark

def llm_topic_extraction(input_text):

    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt = f"Extract main topics from the following text. Only format topics that are one word. \nHere is the text:\n\n{input_text}\n\nTopics:"

    inputs = tokenizer(prompt, return_tensors="pt")

    output_sequences = model.generate(
        inputs['input_ids'],
        max_length=inputs['input_ids'].shape[1] + 50,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        num_beams=5
    )

    # Decode the output
    output_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    # Extract the topics from the output text
    topics_start = output_text.find("Topics:") + len("Topics:")
    topics_text = output_text[topics_start:].strip().split(', ')
    topics_text = [x.lower() for x in topics_text]

    return topics_text