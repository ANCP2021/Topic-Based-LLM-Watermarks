# from lmw.watermark_processor import WatermarkLogitsProcessor
from topic_watermark_processor import WatermarkLogitsProcessor
from lmw.normalizers import normalization_strategy_lookup
from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        AutoModelForCausalLM,
        LogitsProcessorList
    )
import torch
from functools import partial


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = {
    # 'run_gradio': False, 
    # 'run_gradio': True,
    'demo_public': False, 
    'model_name_or_path': 'facebook/opt-125m', 
    # 'model_name_or_path': 'facebook/opt-1.3b', 
    # 'model_name_or_path': 'facebook/opt-2.7b', 
    # 'model_name_or_path': 'facebook/opt-6.7b',
    # 'model_name_or_path': 'facebook/opt-13b',
    # 'load_fp16' : True,
    'load_fp16' : False,
    'prompt_max_length': None, 
    'max_new_tokens': 200, 
    'generation_seed': 123, 
    'use_sampling': True, 
    'n_beams': 1, 
    'sampling_temp': 0.7, 
    'use_gpu': True, 
    'seeding_scheme': 'simple_1', 
    'gamma': 0.25, 
    'delta': 2.0, 
    'normalizers': '', 
    'ignore_repeated_bigrams': False, 
    'detection_z_threshold': 4.0, 
    'select_green_tokens': True,
    'skip_model_load': False,
    'seed_separately': True,
}

# Loads and returns the model
def load_model(args):

    is_seq2seq_model = any([(model_type in args['model_name_or_path']) for model_type in ["t5","T0"]])
    args['seq2seq'] = is_seq2seq_model

    is_decoder_only_model = any([(model_type in args['model_name_or_path']) for model_type in ["gpt","opt","bloom"]])
    args['decoder'] = is_decoder_only_model
    
    if is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args['model_name_or_path'])
    elif is_decoder_only_model:
        if args['load_fp16']:
            model = AutoModelForCausalLM.from_pretrained(args['model_name_or_path'],torch_dtype=torch.float16, device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(args['model_name_or_path'])
    else:
        raise ValueError(f"Unknown model type: {args['model_name_or_path']}")

    tokenizer = AutoTokenizer.from_pretrained(args['model_name_or_path'])

    # model.eval()

    return model, tokenizer


# Instatiate the WatermarkLogitsProcessor according to the watermark parameters
# and generate watermarked text by passing it to the generate method of the model
# as a logits processor.
def generate(prompt, args, model=None, tokenizer=None):
    
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

    ################################

    generate_without_watermark = partial(
        model.generate,
        **gen_kwargs
    )
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

    truncation_warning = True if tokenized_input["input_ids"].shape[-1] == args['prompt_max_length'] else False
    redecoded_input = tokenizer.batch_decode(tokenized_input["input_ids"], skip_special_tokens=True)[0]

    torch.manual_seed(args['generation_seed'])

    output_without_watermark = generate_without_watermark(**tokenized_input)

    # optional to seed before second generation, but will not be the same again generally, unless delta==0.0, no-op watermark
    if args['seed_separately']: 
        torch.manual_seed(args['generation_seed'])
    output_with_watermark = generate_with_watermark(**tokenized_input)

    if args['decoder']:
        # need to isolate the newly generated tokens
        output_without_watermark = output_without_watermark[:,tokenized_input["input_ids"].shape[-1]:]
        output_with_watermark = output_with_watermark[:,tokenized_input["input_ids"].shape[-1]:]

    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]

    return (redecoded_input,
            int(truncation_warning),
            decoded_output_without_watermark, 
            decoded_output_with_watermark,
) 


if __name__ == '__main__':
    args['normalizers'] = (args['normalizers'].split(",") if args['normalizers'] else [])

    # model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")
    # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")

    model, tokenizer = load_model(args)

    # watermark_processor = WatermarkLogitsProcessor(
    #     vocab=list(tokenizer.get_vocab().values()), 
    #     gamma=0.25, 
    #     delta=2.0, 
    #     seeding_scheme="selfhash"
    # )

    input_text = (
        "The diamondback terrapin or simply terrapin (Malaclemys terrapin) is a "
        "species of turtle native to the brackish coastal tidal marshes of the "
        "Northeastern and southern United States, and in Bermuda.[6] It belongs "
        "to the monotypic genus Malaclemys. It has one of the largest ranges of "
        "all turtles in North America, stretching as far south as the Florida Keys "
        "and as far north as Cape Cod.[7] The name 'terrapin' is derived from the "
        "Algonquian word torope.[8] It applies to Malaclemys terrapin in both "
        "British English and American English. The name originally was used by "
        "early European settlers in North America to describe these brackish-water "
        "turtles that inhabited neither freshwater habitats nor the sea. It retains "
        "this primary meaning in American English.[8] In British English, however, "
        "other semi-aquatic turtle species, such as the red-eared slider, might "
        "also be called terrapins. The common name refers to the diamond pattern "
        "on top of its shell (carapace), but the overall pattern and coloration "
        "vary greatly. The shell is usually wider at the back than in the front, "
        "and from above it appears wedge-shaped. The shell coloring can vary "
        "from brown to grey, and its body color can be grey, brown, yellow, "
        "or white. All have a unique pattern of wiggly, black markings or spots "
        "on their body and head. The diamondback terrapin has large webbed "
        "feet.[9] The species is"
    )

    # input_text = (
    #     "Basketball, a sport that has become a global phenomenon, was invented by Dr. James Naismith "
    #     "in December 1891. Naismith, a physical education instructor, created the game as a way to keep "
    #     "his students active indoors during the harsh winters in Springfield, Massachusetts. Using a soccer "
    #     "ball and two peach baskets, he developed a game with 13 basic rules. The objective was simple: score "
    #     "by shooting the ball into the opposing team's basket. From these humble beginnings, basketball has "
    #     "evolved into a sophisticated sport with a rich history and a profound impact on culture and society. "
    #     "\n"
    #     "In the early 20th century, basketball rapidly gained popularity in the United States. The formation "
    #     "of the National Basketball Association (NBA) in 1946 marked a significant milestone, providing a "
    #     "professional platform for the sport. The NBA facilitated the rise of basketball as a major spectator sport, "
    #     "attracting millions of fans with its high-flying dunks, precise shooting, and strategic gameplay. Players like "
    #     "Wilt Chamberlain, Bill Russell, and later, Michael Jordan, became household names, elevating the sport's status "
    #     "and inspiring countless young athletes around the world."
    #     "\n"
    #     "Basketball's influence extends beyond the court. It has become a cultural force, impacting fashion, music, and "
    #     "lifestyle. The \"streetball\" culture, "
    #     )

    watermark_processor = generate(input_text, args, model, tokenizer)

    print(f"Prompt:\n {input_text}")

    redecoded_input, truncation_warning, decoded_output_without_watermark, decoded_output_with_watermark = generate(
        input_text, 
        args, 
        model=model, 
        tokenizer=tokenizer
    )

    print(f"Output without watermark:\n {decoded_output_without_watermark}\n")

    print(f"Output with watermark:\n {decoded_output_with_watermark}\n")

    # tokenized_input = tokenizer(input_text, return_tensors='pt').to(model.device)

    # output_tokens = model.generate(**tokenized_input, logits_processor=LogitsProcessorList([watermark_processor]))

    # output_tokens = output_tokens[:,tokenized_input["input_ids"].shape[-1]:]

    # output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

    # print(f"Generated Text:\n {output_text}")