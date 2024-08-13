"""
A paraphrasing attack is a category of a baseline substitution attack. 
Execution of this attack may be manual by an individual or by rephrasing the output via an LLM. 
"""
import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from langchain_huggingface import ChatHuggingFace # Using langchain for faster inference
access_token = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

class ParaphrasingAttack:
    def __init__(self, model_name="HuggingFaceH4/zephyr-7b-beta", access_token=None):
        self.access_token = access_token
        self.model_loader = self.load_model(model_name)
        self.llm = ChatHuggingFace(llm=self.model_loader)

    # Load specified model for text generation
    def load_model(self, model_name):
        llm = HuggingFaceEndpoint(
            repo_id=model_name,
            huggingfacehub_api_token=self.access_token,
            task="text-generation",
            max_new_tokens=1024,
        )
        return llm

    # LLM message formatting
    def create_message(self, instruct, msg):
        return [
            SystemMessage(content=instruct),
            HumanMessage(content=msg),
        ]

    # Rephrasel of watermarked text
    def rephrase(self, text, topic=None, inference=False):
        # Assumption that there is a watermark, topic words are the target
        if inference:
            init_prompt = f'Can you rephrase the following text while keeping the same length and maintaining the main topic of {topic}:\nHere is the text:\n'
        else:
            init_prompt = 'Rephrase the following text while keeping the same length.\nHere is the text:\n'
        
        messages = self.create_message(init_prompt, text)
        response = self.llm.invoke(messages)
        return response.content

# Use Case Example
if __name__ == '__main__':
    # Hypothetical watermarked text
    watermarked_text = (
        "Sports have been an integral part of human culture for centuries, serving as a means of entertainment, "
        "physical fitness, and social interaction. They are not merely games but vital activities that contribute "
        "to the holistic development of individuals and communities. The significance of sports transcends the boundaries "
        "of competition."
    )
    # Example of paraphrasal use case with access token
    paraphraser = ParaphrasingAttack(access_token=access_token)
    response = paraphraser.rephrase(watermarked_text)
    print(response)
