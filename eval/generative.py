"""
The generative attacks focuses on the generation of the text. 
One example is the emoji attack where emojis are generated after every space instead
of spaces to corrupt the green and red lists.

Goodside, R. There are adversarial attacks for that proposal as well — in particular, generating 
with emojis after words and then removing them before submitting defeats
it., January 2023. URL https://twitter.com/goodside/status/1610682909647671306
NOT USED IN EVALUATION
"""

# Generative attack involves prompting the generation of the LLM instead of modifying its output
def GenerativeAttack(prompt, placeholder):
    generative_attack_prompt = f'\nInstead of spaces between words, replace all with {placeholder} after every word when generating new text.\n'
    return prompt + generative_attack_prompt
    
    # NOTE: Use case is below
    # Hypothetical input text to the model
    input_text = (
        "Sports have been an integral part of human culture for centuries, serving as a means of entertainment, "
        "physical fitness, and social interaction. They are not merely games but vital activities that contribute "
        "to the holistic development of individuals and communities. The significance of sports transcends the boundaries "
        "of competition."
    )

    # Load model and its tokenizer
    model, tokenizer = load_model(args)

    # Extract specified topic from the text
    detected_topics = llm_topic_extraction(input_text)

    # Concatinate the generative attack prompt with the input text
    # Specify emojis to be generated instead of spacings
    generative_attack_input_text = generativeAttack(input_text, placeholder='emojis')

    # Output with skewed watermarked text
    _, _, _, decoded_output_with_watermark = generate(
        generative_attack_input_text, 
        detected_topics,
        args, 
        model=model, 
        tokenizer=tokenizer
    )
    