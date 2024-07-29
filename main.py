import torch
from model import load_model, generate, detect
from topic_extractions import llm_topic_extraction
from inputs import sports_input, technology_input, animals_input
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
    'max_new_tokens': 150, 
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
        "sports": list(range(20000)),
        "animals": list(range(20000, 40000)),
        "technology": list(range(40000, 60000)),
        # Add more topics and corresponding tokens as needed
    },
    'detected_topic': "",
}

if __name__ == '__main__':
    input_text = sports_input()
    # input_text = technology_input()
    # input_text = animals_input()

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