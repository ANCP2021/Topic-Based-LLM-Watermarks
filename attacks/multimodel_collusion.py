"""
A multi-model collusion attack begins with an original text sequence that has been produced by an LLM,
which incorporates a watermarking scheme based on topic relevance. Ideally, we assume that all existing LLMs have
a topic-based watermarking scheme implemented for output text generation. The attacker solicits outputs to
the same prompt from several LLMs, aiming to blend these multiple responses int a single output through insertion,
manipulation, or deletion. This process might modify the topics extracted from the text, depending on the extraction
methods and the defined list pairs of general and specific topics.
"""
import torch
from transformers import ( 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM
)
from utils.utils import generate, llm_topic_extraction

class MultiModelCollusion:
    def __init__(self, models):
        self.models = models
        self.load_fp16 = False
    
    # Split the watermark input text into equal parts depending on the number of LLMs used in the attack
    def split_input(self, input_text, num_parts):
        input_text = input_text.strip()
        words = input_text.split()
        # Number of words per part
        num_words = len(words)
        words_per_part = num_words // num_parts
        model_inputs = []
        current_part = []
        current_word_count = 0
        
        # Split the words into parts
        for word in words:
            current_part.append(word)
            current_word_count += 1
            
            if current_word_count >= words_per_part and len(model_inputs) < num_parts - 1:
                model_inputs.append(' '.join(current_part))
                current_part = []
                current_word_count = 0
        
        # Add the last part
        model_inputs.append(' '.join(current_part))
        return model_inputs
    
    # Modified model loader from main watermarking scheme
    def load_model(self, model_name, load_fp16):
        is_seq2seq_model = any([(model_type in model_name) for model_type in ["t5", "T0"]])
        is_decoder_only_model = any([(model_type in model_name) for model_type in ["gpt", "opt", "bloom"]])

        if is_seq2seq_model:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        elif is_decoder_only_model:
            if load_fp16:
                model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            raise ValueError(f"Unknown model type: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return model, tokenizer

    # Generate the output from multiple LLMs and string them together for collusion attack
    def modeler(self, input_text, args):
        num_models = len(self.models)
        detected_topic = llm_topic_extraction(input_text)
        model_inputs = self.split_input(input_text, num_models)
        
        for idx, model_input in enumerate(model_inputs):
            model, tokenizer = self.load_model(self.models[idx], self.load_fp16)
            if idx == 0: # No continuation from previous LLM output
                prompt = f'Rephrase the following text while keeping the same length.\nHere is the text:\n{model_input}'
            else: # Continuation of previous LLM output
                prompt = f'Rephrase the following text while keeping the same length and a continuation of {decoded_output_with_watermark}.\nHere is the text:\n{model_input}\n\n'

            decoded_output_with_watermark = generate(
                prompt, 
                detected_topic,
                args, 
                model=model, 
                tokenizer=tokenizer
            )
        return decoded_output_with_watermark

# Use Case Example
if __name__ == "__main__":
    model_args = {
        'prompt_max_length': None, 
        'max_new_tokens': 170, 
        'generation_seed': 123, 
        'use_sampling': True, 
        'n_beams': 1, 
        'sampling_temp': 0.7, 
        'seeding_scheme': 'simple_1', 
        'gamma': 0.70, 
        'delta': 3.5, 
        'select_green_tokens': True,
        'seed_separately': True,
        'is_topic': True,
        'topic_token_mapping': {
            "sports": list(range(22000)),
            "animals": list(range(22000, 44000)),
            "technology": list(range(44000, 66000)),
            # Add more topics and corresponding tokens as needed
        },
        'detected_topic': "",
    }

    # Hypothetical watermarked text
    watermarked_text = (
        "Sports have been an integral part of human culture for centuries, serving as a means of entertainment, "
        "physical fitness, and social interaction. They play a crucial role in promoting teamwork, discipline, and "
        "perseverance among individuals. From ancient times to the modern era, sports have evolved and diversified "
        "significantly, with various forms emerging across different cultures and regions.\n"
        "The Olympic Games, for instance, have a long-standing history dating back to ancient Greece, where they "
        "were held to honor the gods. Today, they are a global event that brings together athletes from around the "
        "world to compete in a wide range of sports. The influence of sports extends beyond physical activities, "
        "affecting economics, politics, and even education.\n"
        "In recent years, the rise of technology has further transformed the landscape of sports. Innovations such "
        "as data analytics, wearable technology, and virtual reality are now integral parts of athletic training "
        "and performance assessment. These advancements not only enhance the capabilities of athletes but also "
        "provide fans with new ways to engage with their favorite sports."
    )

    # Example of 3 LLMs used in collusion attack scheme
    models = ["facebook/opt-1.3b", "openai-community/gpt2", "google/flan-t5-small"]
    collusion = MultiModelCollusion(models)
    generated_text = collusion.modeler(watermarked_text, model_args)
    print(generated_text)
   