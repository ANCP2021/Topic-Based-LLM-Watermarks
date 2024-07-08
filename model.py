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
import random
import matplotlib.pyplot as plt
import textstat


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
    'gamma': 0.70, 
    'delta': 3.5, 
    'normalizers': '', 
    'ignore_repeated_bigrams': False, 
    'detection_z_threshold': 2.0, 
    'select_green_tokens': True,
    'skip_model_load': False,
    'seed_separately': True,
    'is_topic': True,
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
                                            topic_token_mapping=args['topic_token_mapping'],
                                            )
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


def modify_text(text, n_edits, edit_type='insert'):
    words = text.split()
    for _ in range(n_edits):
        if edit_type == 'insert':
            pos = random.randint(0, len(words))
            words.insert(pos, random.choice(words))
        elif edit_type == 'delete' and len(words) > 1:
            pos = random.randint(0, len(words) - 1)
            words.pop(pos)
        elif edit_type == 'modify':
            pos = random.randint(0, len(words) - 1)
            words[pos] = random.choice(words)
    return ' '.join(words)

def analyze_robustness(original_prompt, output, args, device, tokenizer, num_edits_list, edit_type='insert'):
    green_fractions = []
    z_scores = []
    readability_scores = []


    for n_edits in num_edits_list:
        modified_output = modify_text(output, n_edits, edit_type)
        detection_result = detect(original_prompt, modified_output, args, device, tokenizer)
        for item in detection_result:
            if len(item) > 1:
                if item[0] == 'scores':
                    # Extracting 'green_fraction' and 'z_score' from the string
                    score_dict_str = item[1]
                    score_dict = eval(score_dict_str)  # Evaluate the string as a dictionary

                    # Assign values to variables
                    green_fractions.append(score_dict.get('green_fraction'))
                    z_scores.append(score_dict.get('z_score'))
        # scores = detection_result[0][0][1]
        # scores_dict = eval(scores)
        # green_fractions.append(scores_dict['green_fraction'])
        # z_scores.append(scores_dict['z_score'])
        readability_score = textstat.flesch_reading_ease(modified_output)
        readability_scores.append(readability_score)

    return green_fractions, z_scores, readability_scores

def plot_robustness(num_edits_list, green_fractions, z_scores, readability_scores):
    fig, ax1 = plt.subplots()

    # Plotting Green Fraction
    color = 'tab:blue'
    ax1.set_xlabel('Number of Edits')
    ax1.set_ylabel('Green Fraction', color=color)
    ax1.plot(num_edits_list, green_fractions, color=color, label='Green Fraction')
    ax1.tick_params(axis='y', labelcolor=color)

    # Adding another y-axis for Z-Score
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Z-Score', color=color)
    ax2.plot(num_edits_list, z_scores, color=color, label='Z-Score')
    ax2.tick_params(axis='y', labelcolor=color)

    # Adding another y-axis for readability scores
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.15))  # Offset the right spine of ax3
    color = 'tab:green'
    ax3.set_ylabel('Text Quality', color=color)
    ax3.plot(num_edits_list, readability_scores, color=color, label='Text Quality')
    ax3.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('')
    plt.legend(loc='upper left')
    plt.show()



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

    # decoded_output_without_watermark =  (
    #     " especially important for children, especially those from disadvantaged backgrounds, who are less likely to have the opportunity to participate in sports."
    #     "While sports are an essential part of our lives, their impact is not enough to maintain a healthy lifestyle. The Global Sport and Well-Being Report, conducted by the World Health Organization (WHO), indicates that one in six people live with the burden of chronic diseases such as diabetes, cardiovascular disease, obesity, and hypertension."
    #     "The impact of these diseases is felt across the world, and they are largely preventable. The WHO estimates that 1.7 billion people are overweight or obese and, by 2030, the number of overweight and obese people in the world will increase by 230 million people. The WHO estimates that, in the next decade, the number of overweight and obese people will increase by 526 million people."
    #     "This"
    # )   

    # decoded_output_with_watermark = (
    #     " particularly important when it comes to maintaining healthy and strong friendships. These activities also foster a sense of community, serving to strengthen ties between different age groups. They foster positive relationships between individuals, helping to prevent aggression and bullying among young children, adolescents, and young adults."
    #     "Gender and ethnicity, however, also affect athletes. Women, particularly, experience a range of negative experiences during sports training and competition. These experiences include harassment, bullying, and violence. These experiences affect athletes of different gender, ethnicity, and age. Women, particularly, experience a range of negative experiences during sports training and competition. These experiences include harassment, bullying, and violence. These experiences affect athletes of different gender, ethnicity, and age."
    #     "This situation, however, is changing. Women now make up a growing number of sports athletes and their involvement and involvement increase with the rise "
    # )

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

    
    num_edits_list = [0, 5, 10, 15, 20, 25, 30]
    input_prompt = input_text + decoded_output_with_watermark

    green_fractions, z_scores, readability_scores = analyze_robustness(
        input_prompt,
        decoded_output_with_watermark, 
        args, 
        device, 
        tokenizer, 
        num_edits_list, 
        edit_type='insert'
    )
    plot_robustness(num_edits_list, green_fractions, z_scores, readability_scores)

    green_fractions, z_scores, readability_scores = analyze_robustness(
        input_prompt,
        decoded_output_with_watermark, 
        args, 
        device, 
        tokenizer, 
        num_edits_list, 
        edit_type='modify'
    )
    plot_robustness(num_edits_list, green_fractions, z_scores, readability_scores)

    green_fractions, z_scores, readability_scores = analyze_robustness(
        input_prompt,
        decoded_output_with_watermark, 
        args, 
        device, 
        tokenizer, 
        num_edits_list, 
        edit_type='delete'
    )
    plot_robustness(num_edits_list, green_fractions, z_scores, readability_scores)
