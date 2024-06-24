# from lmw.watermark_processor import WatermarkLogitsProcessor
from lmw.extended_watermark_processor import WatermarkLogitsProcessor
from lmw.normalizers import normalization_strategy_lookup
from transformers import (
        pipeline, 
        AutoModelForTokenClassification,
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        AutoModelForCausalLM,
        LogitsProcessorList
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

        watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                gamma=0.25,
                                                delta=2.0,
                                                seeding_scheme="selfhash")

        input_text = (
                "Sally sells sea shells by the sea "
                # "The diamondback terrapin or simply terrapin (Malaclemys terrapin) is a "
                # "species of turtle native to the brackish coastal tidal marshes of the "
                # "Northeastern and southern United States, and in Bermuda.[6] It belongs "
                # "to the monotypic genus Malaclemys. It has one of the largest ranges of "
                # "all turtles in North America, stretching as far south as the Florida Keys "
                # "and as far north as Cape Cod.[7] The name 'terrapin' is derived from the "
                # "Algonquian word torope.[8] It applies to Malaclemys terrapin in both "
                # "British English and American English. The name originally was used by "
                # "early European settlers in North America to describe these brackish-water "
                # "turtles that inhabited neither freshwater habitats nor the sea. It retains "
                # "this primary meaning in American English.[8] In British English, however, "
                # "other semi-aquatic turtle species, such as the red-eared slider, might "
                # "also be called terrapins. The common name refers to the diamond pattern "
                # "on top of its shell (carapace), but the overall pattern and coloration "
                # "vary greatly. The shell is usually wider at the back than in the front, "
                # "and from above it appears wedge-shaped. The shell coloring can vary "
                # "from brown to grey, and its body color can be grey, brown, yellow, "
                # "or white. All have a unique pattern of wiggly, black markings or spots "
                # "on their body and head. The diamondback terrapin has large webbed "
                # "feet.[9] The species is"
                )

        tokenized_input = tokenizer(input_text, return_tensors='pt').to(model.device)

        output_tokens = model.generate(**tokenized_input,
                                logits_processor=LogitsProcessorList([watermark_processor]))

        output_tokens = output_tokens[:,tokenized_input["input_ids"].shape[-1]:]

        output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

        print(output_text)