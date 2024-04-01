def add_img_postfix(prompt, num_img_tokens=4):
    if isinstance(prompt, str):
        prompt_postfixed = prompt + ' '  + ' '.join([f'[IMG{i}]' for i in range(1, num_img_tokens + 1)])
    else:
        prompt_postfixed = tuple(p + ' '  + ' '.join([f'[IMG{i}]' for i in range(1, num_img_tokens + 1)]) for p in prompt)
    return prompt_postfixed


def get_expanded_ids(prompt_postfixed, tokenizer):
    out = tokenizer(
        prompt_postfixed,
        add_special_tokens=False,
        return_tensors="pt",
        padding="max_length",
        truncation=True
      )
    input_ids = out.input_ids
    return input_ids
