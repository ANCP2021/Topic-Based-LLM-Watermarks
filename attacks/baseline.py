# TODO Baseline attacks, insertion, substitution, deletion



def modify_text(text, n_edits, edit_type='insert'):
    words = text.split()
    for _ in range(n_edits):
        if edit_type == 'insert':
            pos = random.randint(0, len(words))
            words.insert(pos, random.choice(words))
        elif edit_type == 'delete' and len(words) > 1:
            pos = random.randint(0, len(words) - 1)
            words.pop(pos)
        elif edit_type == 'modify':
            pos = random.randint(0, len(words) - 1)
            words[pos] = random.choice(words)
    return ' '.join(words)

def analyze_robustness(original_prompt, output, args, device, tokenizer, num_edits_list, edit_type='insert'):
    green_fractions = []
    z_scores = []
    readability_scores = []


    for n_edits in num_edits_list:
        modified_output = modify_text(output, n_edits, edit_type)
        detection_result = detect(original_prompt, modified_output, args, device, tokenizer)
        for item in detection_result:
            if len(item) > 1:
                if item[0] == 'scores':
                    # Extracting 'green_fraction' and 'z_score' from the string
                    score_dict_str = item[1]
                    score_dict = eval(score_dict_str)  # Evaluate the string as a dictionary

                    # Assign values to variables
                    green_fractions.append(score_dict.get('green_fraction'))
                    z_scores.append(score_dict.get('z_score'))
        # scores = detection_result[0][0][1]
        # scores_dict = eval(scores)
        # green_fractions.append(scores_dict['green_fraction'])
        # z_scores.append(scores_dict['z_score'])
        readability_score = textstat.flesch_reading_ease(modified_output)
        readability_scores.append(readability_score)

    return green_fractions, z_scores, readability_scores

def plot_robustness(num_edits_list, green_fractions, z_scores, readability_scores):
    fig, ax1 = plt.subplots()

    # Plotting Green Fraction
    color = 'tab:blue'
    ax1.set_xlabel('Number of Edits')
    ax1.set_ylabel('Green Fraction', color=color)
    ax1.plot(num_edits_list, green_fractions, color=color, label='Green Fraction')
    ax1.tick_params(axis='y', labelcolor=color)

    # Adding another y-axis for Z-Score
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Z-Score', color=color)
    ax2.plot(num_edits_list, z_scores, color=color, label='Z-Score')
    ax2.tick_params(axis='y', labelcolor=color)

    # Adding another y-axis for readability scores
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.15))  # Offset the right spine of ax3
    color = 'tab:green'
    ax3.set_ylabel('Text Quality', color=color)
    ax3.plot(num_edits_list, readability_scores, color=color, label='Text Quality')
    ax3.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('')
    plt.legend(loc='upper left')
    plt.show()



if __name__ == '__main__':

    

    num_edits_list = [0, 5, 10, 15, 20, 25, 30]
    input_prompt = input_text + decoded_output_with_watermark

    green_fractions, z_scores, readability_scores = analyze_robustness(
        input_prompt,
        decoded_output_with_watermark, 
        args, 
        device, 
        tokenizer, 
        num_edits_list, 
        edit_type='insert'
    )
    plot_robustness(num_edits_list, green_fractions, z_scores, readability_scores)

    green_fractions, z_scores, readability_scores = analyze_robustness(
        input_prompt,
        decoded_output_with_watermark, 
        args, 
        device, 
        tokenizer, 
        num_edits_list, 
        edit_type='modify'
    )
    plot_robustness(num_edits_list, green_fractions, z_scores, readability_scores)

    green_fractions, z_scores, readability_scores = analyze_robustness(
        input_prompt,
        decoded_output_with_watermark, 
        args, 
        device, 
        tokenizer, 
        num_edits_list, 
        edit_type='delete'
    )
    plot_robustness(num_edits_list, green_fractions, z_scores, readability_scores)