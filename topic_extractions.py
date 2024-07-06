import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from gensim import corpora
from gensim.models import LdaModel
import ssl
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# POS tags for lemmatization
def wordnet_pos_tags(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# preprocessing function
def txt_preprocess_pipeline(input_text):
    # text to lowercase
    standard_txt = input_text.lower()

    # remove multiple white spaces and line breaks
    clean_text = re.sub(r'\n', ' ', standard_txt)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    clean_text = clean_text.strip()
    
    # tokenize text
    tokens = word_tokenize(clean_text)
    # remove non-alphabetic tokens
    filtered_tokens_alpha = [word for word in tokens if word.isalpha() and not re.match(r'^[ivxlcdm]+$', word)]
    
    # NLTK stopword list and add original stopwords to be removed
    stop_words = stopwords.words('english')
    # remove stopwords
    filtered_tokens_final = [w for w in filtered_tokens_alpha if not w in stop_words]

    # POS tagging
    pos_tags = nltk.pos_tag(filtered_tokens_final)

    # lemmatize word-tokens via assigned POS tags
    lemmatizer = WordNetLemmatizer()
    lemma_tokens = [lemmatizer.lemmatize(token, wordnet_pos_tags(pos_tag)) for token, pos_tag in pos_tags]

    return lemma_tokens

# Latent Dirichlet Allocation 
def lda(input, num_topics):
    tokens = txt_preprocess_pipeline(input)
    texts = [tokens]
    dictionary = corpora.Dictionary(texts)

    # generate corpus as BoW
    corpus = [dictionary.doc2bow(text) for text in texts]

    # train LDA model
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, random_state=100, chunksize=20, num_topics=7, passes=50, iterations=100)

    return lda_model

def generate_response(prompt, model, tokenizer, max_length):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(input_ids=inputs['input_ids'], max_length=max_length)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

def llm_topic_extraction(input_text):

    model_name = "facebook/opt-1.3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt = f"Extract main topics from the following text:\n\n{input_text}\n\nTopics:"

    inputs = tokenizer(prompt, return_tensors="pt")

    output_sequences = model.generate(
        inputs['input_ids'],
        max_length=inputs['input_ids'].shape[1] + 50,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        num_beams=5
    )

    # Decode the output
    output_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    # Extract the topics from the output text
    topics_start = output_text.find("Topics:") + len("Topics:")
    topics_text = output_text[topics_start:].strip().split(', ')

    return topics_text


if __name__ == "__main__":
    with open('input_prompt.txt', 'r') as file:
        input_prompt = file.read()

    num_topics = 3

    lda_model = lda(input_prompt, num_topics)
    
    topics = llm_topic_extraction(input_prompt)

