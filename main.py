import torch
from non_watermarked_llm import NonWatermarkedLLM
from model import load_model, generate, detect
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

if __name__ == '__main__':

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

    # nonWatermarkedLLM = NonWatermarkedLLM(model_name=args['model_name_or_path'])
    # input_text = nonWatermarkedLLM.generate_response(input_text)

    args['normalizers'] = (args['normalizers'].split(",") if args['normalizers'] else [])

    model, tokenizer = load_model(args)

    
    if args['is_topic']:
        detected_topics = llm_topic_extraction(input_text)
    else:
        detected_topics = []

    if DEBUG: print(f"Topic extraction is finished for watermarking: {detected_topics}")

    print(f"Prompt:\n {input_text}")

    # input_text = f"Rewrite the following text:\n '{input_text}'"
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