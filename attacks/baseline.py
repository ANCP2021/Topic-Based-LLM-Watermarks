"""
Baseline attack consists of the insertion, substitution, and deletion of text for a given output
sequence. The attacker selects a single or a combination of techniques with the objective to diminish 
detection accuracy.
"""
import random
from nltk.corpus import wordnet
from nltk import pos_tag
from utils.utils import is_important_word

class BaselineAttack:
    def __init__(self):
        super().__init__()
        
    # Synonym helper function for substitution attack
    def get_synonym(self, word):
        synonyms = set()
        
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())

        if word in synonyms:
            synonyms.remove(word)  
            
        return list(synonyms)

    # Modify text randomly through insertion, deletion, and substitution
    def modify_text(self, text, n_edits, edit_type='insert'):
        words = text.split()

        for _ in range(n_edits):
            if edit_type == 'insert':
                pos = random.randint(0, len(words))
                words.insert(pos, random.choice(words))
            elif edit_type == 'delete' and len(words) > 1:
                pos = random.randint(0, len(words) - 1)
                words.pop(pos)
            elif edit_type == 'substitute':
                pos = random.randint(0, len(words) - 1)
                words[pos] = random.choice(words)

        return ' '.join(words)

    # Modify text under the assumption that there is a watermark, important words are the target
    # choosing more important words (excluding 'the', 'and', etc.) randomly 
    def inference_modify_text(self, text, n_edits, edit_type='insert'):
        words = text.split()
        tagged_words = pos_tag(words) 
        important_words = [word for word, tag in tagged_words if is_important_word(tag)]
        
        for _ in range(n_edits):
            if edit_type == 'insert':
                pos = random.randint(0, len(words))
                word_to_insert = random.choice(important_words)
                words.insert(pos, word_to_insert)
            elif edit_type == 'delete' and len(important_words) > 0:
                word_to_delete = random.choice(important_words)
                if word_to_delete in words:
                    pos = words.index(word_to_delete)
                    words.pop(pos)
            elif edit_type == 'substitute':
                pos = random.randint(0, len(words) - 1)
                word = words[pos]
                synonyms = self.get_synonym(word)
                if synonyms:
                    new_word = random.choice(synonyms)
                    words[pos] = new_word

        return ' '.join(words)

    # Combination function for insertion, deletion, and substitution
    def combination_modify_text(self, text, insertion_n_edits=0, insertion_is_inferenced=False, 
                                deletion_n_edits=0, deletion_is_inferenced=False, 
                                substitution_n_edits=0, 
                                substitution_is_inferenced=False):
        if insertion_n_edits > 0:
            if insertion_is_inferenced:
                text = self.inference_modify_text(text, insertion_n_edits, edit_type='insert')
            else:
                text = self.modify_text(text, insertion_n_edits, edit_type='insert')

        if deletion_n_edits > 0:
            if deletion_is_inferenced:
                text = self.inference_modify_text(text, deletion_n_edits, edit_type='delete')
            else:
                text = self.modify_text(text, deletion_n_edits, edit_type='delete')

        if substitution_n_edits > 0:
            if substitution_is_inferenced:
                text = self.inference_modify_text(text, substitution_n_edits, edit_type='substitute')
            else:
                text = self.modify_text(text, substitution_n_edits, edit_type='substitute')

        return text

# Use Case Example
if __name__ == '__main__':
    # Hypothetical watermarked text
    watermarked_text = (
        "Sports have been an integral part of human culture for centuries, serving as a means of entertainment, "
        "physical fitness, and social interaction. They are not merely games but vital activities that contribute "
        "to the holistic development of individuals and communities. The significance of sports transcends the boundaries "
        "of competition."
    )
    # Example of 3 random insertions where there is no assumption of a watermark
    baseline = BaselineAttack()
    text = baseline.inference_modify_text(watermarked_text, 3, 'insert')
    print(text)
    