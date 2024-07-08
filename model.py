from topic_watermark_processor import TopicWatermarkLogitsProcessor, TopicWatermarkDetector
from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector
from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        AutoModelForCausalLM,
        LogitsProcessorList
    )
import torch
from functools import partial
from topic_extractions import llm_topic_extraction
from pprint import pprint

DEBUG = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = {
    'demo_public': False, 
    # 'model_name_or_path': 'facebook/opt-125m', 
    'model_name_or_path': 'facebook/opt-1.3b', 
    # 'model_name_or_path': 'facebook/opt-2.7b', 
    # 'model_name_or_path': 'facebook/opt-6.7b',
    # 'model_name_or_path': 'facebook/opt-13b',
    # 'load_fp16' : True,
    'load_fp16' : False,
    'prompt_max_length': None, 
    'max_new_tokens': 170, 
    'generation_seed': 123, 
    'use_sampling': True, 
    'n_beams': 1, 
    'sampling_temp': 0.7, 
    'use_gpu': True, 
    'seeding_scheme': 'simple_1', 
    'gamma': 0.50, 
    'delta': 3.5, 
    'normalizers': '', 
    'ignore_repeated_bigrams': False, 
    'detection_z_threshold': 2.0, 
    'select_green_tokens': True,
    'skip_model_load': False,
    'seed_separately': True,
    'is_topic': False,
    'topic_token_mapping': {
        "sports": list(range(22000)),
        "animals": list(range(22000, 44000)),
        "turtles": list(range(44000, 66000)),
        # Add more topics and corresponding tokens as needed
    },
    'detected_topic': "",
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

    return model, tokenizer

# Instatiate the WatermarkLogitsProcessor according to the watermark parameters
# and generate watermarked text by passing it to the generate method of the model
# as a logits processor.
def generate(prompt, detected_topics, args, model=None, tokenizer=None):
    
    # if set to is_topic, use topic based watermarks
    if args['is_topic']:
        detected_topic = ''
        for topic in detected_topics:
            if topic.lower() in args['topic_token_mapping']:
                detected_topic = topic.lower()
                break
        if detected_topic == '':
            detected_topic = detected_topics[0]

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

    # generate without the watermark 
    generate_without_watermark = partial(
        model.generate,
        **gen_kwargs
    ) 

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

def format_names(s):
    """Format names for the gradio demo interface"""
    s=s.replace("num_tokens_scored","Tokens Counted (T)")
    s=s.replace("num_green_tokens","# Tokens in Greenlist")
    s=s.replace("green_fraction","Fraction of T in Greenlist")
    s=s.replace("z_score","z-score")
    s=s.replace("p_value","p value")
    s=s.replace("prediction","Prediction")
    s=s.replace("confidence","Confidence")
    return s

def list_format_scores(score_dicts, detection_threshold):
    """Format the detection metrics into a gradio dataframe input format"""
    lst_2d = []
    # lst_2d.append(["z-score threshold", f"{detection_threshold}"])
    if isinstance(score_dicts, dict):  # For backward compatibility if a single dict is passed
        score_dicts = [score_dicts]

    for score_dict in score_dicts:
        topic_scores = []    
        for k,v in score_dict.items():
            if k=='green_fraction': 
                topic_scores.append([format_names(k), f"{v:.1%}"])
            elif k=='confidence': 
                topic_scores.append([format_names(k), f"{v:.3%}"])
            elif isinstance(v, float): 
                topic_scores.append([format_names(k), f"{v:.3g}"])
            elif isinstance(v, bool):
                topic_scores.append([format_names(k), ("Watermarked" if v else "Human/Unwatermarked")])
            else: 
                topic_scores.append([format_names(k), f"{v}"])

        if "confidence" in score_dict:
            topic_scores.insert(-2,["z-score Threshold", f"{detection_threshold}"])
        else:
            topic_scores.insert(-1,["z-score Threshold", f"{detection_threshold}"])

    lst_2d.extend(topic_scores)
    lst_2d.append([])

    return lst_2d

def detect(original_prompt, input_text, args, device=None, tokenizer=None):
    """Instantiate the WatermarkDetection object and call detect on
        the input text returning the scores and outcome of the test"""
    
    if args['is_topic']:
        detected_topics = llm_topic_extraction(original_prompt)
        print(f"FInished detected topics for generated text: {detected_topics}")
        watermark_detector = TopicWatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                            gamma=args['gamma'],
                                            seeding_scheme=args['seeding_scheme'],
                                            device=device,
                                            tokenizer=tokenizer,
                                            z_threshold=args['detection_z_threshold'],
                                            normalizers=args['normalizers'],
                                            ignore_repeated_bigrams=args['ignore_repeated_bigrams'],
                                            select_green_tokens=args['select_green_tokens'],
                                            topic_token_mapping=args['topic_token_mapping'])
    else:
        watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                            gamma=args['gamma'],
                                            seeding_scheme=args['seeding_scheme'],
                                            device=device,
                                            tokenizer=tokenizer,
                                            z_threshold=args['detection_z_threshold'],
                                            normalizers=args['normalizers'],
                                            ignore_repeated_bigrams=args['ignore_repeated_bigrams'],
                                            select_green_tokens=args['select_green_tokens'])

    if len(input_text)-1 > watermark_detector.min_prefix_len:
        if args['is_topic']:
            score_dict = watermark_detector.detect(input_text, detected_topics=detected_topics)
        else:
            score_dict = watermark_detector.detect(input_text)

        output = list_format_scores(score_dict, watermark_detector.z_threshold)
    else:
        output = [["Error","string too short to compute metrics"]]
        output += [["",""] for _ in range(6)]
    return output

if __name__ == '__main__':

    args['normalizers'] = (args['normalizers'].split(",") if args['normalizers'] else [])

    model, tokenizer = load_model(args)

    # input_text = (
    #     "The diamondback terrapin or simply terrapin (Malaclemys terrapin) is a "
    #     "species of turtle native to the brackish coastal tidal marshes of the "
    #     "Northeastern and southern United States, and in Bermuda.[6] It belongs "
    #     "to the monotypic genus Malaclemys. It has one of the largest ranges of "
    #     "all turtles in North America, stretching as far south as the Florida Keys "
    #     "and as far north as Cape Cod.[7] The name 'terrapin' is derived from the "
    #     "Algonquian word torope.[8] It applies to Malaclemys terrapin in both "
    #     "British English and American English. The name originally was used by "
    #     "early European settlers in North America to describe these brackish-water "
    #     "turtles that inhabited neither freshwater habitats nor the sea. It retains "
    #     "this primary meaning in American English.[8] In British English, however, "
    #     "other semi-aquatic turtle species, such as the red-eared slider, might "
    #     "also be called terrapins. The common name refers to the diamond pattern "
    #     "on top of its shell (carapace), but the overall pattern and coloration "
    #     "vary greatly. The shell is usually wider at the back than in the front, "
    #     "and from above it appears wedge-shaped. The shell coloring can vary "
    #     "from brown to grey, and its body color can be grey, brown, yellow, "
    #     "or white. All have a unique pattern of wiggly, black markings or spots "
    #     "on their body and head. The diamondback terrapin has large webbed "
    #     "feet.[9] The species is"
    # )

    input_text = (
        "Sports have been an integral part of human culture for centuries, serving as a means of entertainment, "
        "physical fitness, and social interaction. They are not merely games but vital activities that contribute "
        "to the holistic development of individuals and communities. The significance of sports transcends the boundaries "
        "of competition, impacting physical health, mental well-being, social cohesion, and even economic growth.\n"
        "Engaging in sports is one of the most effective ways to maintain physical health. Regular participation in physical "
        "activities helps in the prevention of chronic diseases such as obesity, cardiovascular diseases, diabetes, and hypertension. "
        "Sports improve cardiovascular fitness, strengthen muscles, enhance flexibility, and boost overall stamina. For children "
        "and adolescents, sports are crucial for developing healthy growth patterns and preventing lifestyle-related diseases "
        "later in life.\n"
        "The mental health benefits of sports are equally profound. Physical activity triggers the release of endorphins, "
        "which are natural mood lifters. This can help reduce stress, anxiety, and depression. Sports also improve cognitive "
        "function, enhancing concentration, memory, and learning abilities. The discipline and focus required in sports "
        "can translate into improved academic and professional performance, fostering a sense of accomplishment and boosting self-esteem.\n"
        "Sports serve as a powerful tool for social integration. They bring people together, fostering a sense of community and belonging. "
        "Team sports, in particular, teach essential life skills such as teamwork, leadership, communication, and cooperation. These skills are"
        )

    if args['is_topic']:
        detected_topics = llm_topic_extraction(input_text)
    else:
        detected_topics = []

    if DEBUG: print(f"Topic extraction is finished for watermarking: {detected_topics}")

    print(f"Prompt:\n {input_text}")

    redecoded_input, truncation_warning, decoded_output_without_watermark, decoded_output_with_watermark = generate(
        input_text, 
        detected_topics,
        args, 
        model=model, 
        tokenizer=tokenizer
    )

    if DEBUG: print("Decoding with and without watermarkings are finished")

    input_prompt = input_text + decoded_output_without_watermark

    without_watermark_detection_result = detect(input_prompt, decoded_output_without_watermark, 
                                                args, 
                                                device=device, 
                                                tokenizer=tokenizer)
    if DEBUG: print("Finished without watermark detection")

    input_prompt = input_text + decoded_output_with_watermark

    with_watermark_detection_result = detect(input_prompt, decoded_output_with_watermark, 
                                                args, 
                                                device=device, 
                                                tokenizer=tokenizer)
    if DEBUG: print("Finished with watermark detection")


    print("#########################################")
    print("Output without watermark:")
    print(decoded_output_without_watermark)
    print(("#########################################"))
    print(f"Detection result @ {args['detection_z_threshold']}:")
    pprint(without_watermark_detection_result)
    print(("#########################################"))

    print(("#########################################"))
    print("Output with watermark:")
    print(decoded_output_with_watermark)
    print(("#########################################"))
    print(f"Detection result @ {args['detection_z_threshold']}:")
    pprint(with_watermark_detection_result)
    print(("#########################################"))


    # print(f"Output without watermark:\n {decoded_output_without_watermark}\n")

    # print(f"Output with watermark:\n {decoded_output_with_watermark}\n")
