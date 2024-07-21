from transformers import pipeline

class NonWatermarkedLLM:
    def __init__(self, model_name='facebook/opt-1.3b', max_new_tokens=170, temperature=0.7):
        self.generator = pipeline('text-generation', model=model_name)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate_response(self, prompt):
        response = self.generator(
            prompt, 
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature, 
            do_sample=True,  
            num_return_sequences=1
        )
        
        return response[0]['generated_text'].strip()
