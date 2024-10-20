import torch
from model import load_model, generate, detect
from topic_extractions import llm_topic_extraction
from inputs import sports_input, technology_input, animals_input, medicine_input, music_input
from pprint import pprint

DEBUG = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_token_mappings():
    total_tokens = 100000
    topics = ["sports", "animals", "technology", "music", "medicine"]

    # Initialize the mapping dictionary
    topic_token_mapping = {topic: [] for topic in topics}

    # Distribute the tokens in a staggered manner
    for i in range(total_tokens):
        topic_index = i % len(topics)
        topic = topics[topic_index]
        topic_token_mapping[topic].append(i)
    return topic_token_mapping
token_mappings = get_token_mappings()

args = {
    'demo_public': False, 
    'model_name_or_path': 'facebook/opt-1.3b', 
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
    'is_topic': False,
    'topic_token_mapping': token_mappings,
    'detected_topic': "",
}

if __name__ == '__main__':
    input_text = sports_input()

    args['normalizers'] = (args['normalizers'].split(",") if args['normalizers'] else [])

    model, tokenizer = load_model(args)

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