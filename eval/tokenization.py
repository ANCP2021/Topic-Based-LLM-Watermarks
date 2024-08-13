"""
A tokenization attack is classified as a form of insertion, where a token is modified into
multiple subsequent tokens. For example, given the token “sports” within a G list. An attacker can add characters
such as '_' or '*', creating additional tokens categorized as R listed tokens by the detection scheme. The original G
listed token, “sports”, is manipulated into “s_p_o_r_t_s”, creating multiple new R listed tokens in the text sequence.
"""
import random
from nltk import pos_tag
from utils.utils import is_important_word

class TokenizationAttack:
    def __init__(self):
        super().__init__()

    # Function to modify text to change subword tokenization
    def tokenization_attack(self, text, n_edits, insert_char=' ', max_insertions_per_word=1, inference=False):
        words = text.split()
        tagged_words = pos_tag(words) 
        important_words = [word for word, tag in tagged_words if is_important_word(tag)]
        
        for _ in range(n_edits):
            # Assumption that there is a watermark, important words are the target
            if inference and important_words: 
                word = random.choice(important_words)
                pos = words.index(word)
            else:
                pos = random.randint(0, len(words) - 2)
                word = words[pos]
            
            if len(word) > 1:
                num_insertions = random.randint(1, min(max_insertions_per_word, len(word) - 1))
                insert_positions = random.sample(range(1, len(word)), num_insertions)
                
                # Insert character to randomly chosen word
                modified_word = list(word)
                for insert_pos in sorted(insert_positions, reverse=True):
                    modified_word.insert(insert_pos, insert_char)

                words[pos] = ''.join(modified_word)

                # Do not duplicate important words
                if inference:
                    important_words = [w for w in important_words if w != word]

        return ' '.join(words)

# Use Case Example
if __name__ == '__main__':
    # Hypothetical watermarked text
    watermarked_text = (
        "Sports have been an integral part of human culture for centuries, serving as a means of entertainment, "
        "physical fitness, and social interaction. They are not merely games but vital activities that contribute "
        "to the holistic development of individuals and communities. The significance of sports transcends the boundaries "
        "of competition."
    )
    # Example of 2 inserted '_' characters in 3 words which are not in important list, so there is no watermark assumed by the attacker
    tokenization = TokenizationAttack()
    text = tokenization.tokenization_attack(watermarked_text, n_edits=3, insert_char='_', max_insertions_per_word=2)
    print(text)
