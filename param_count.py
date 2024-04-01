import os

def save_param_count_txt(model, log_dir = './'):
    param_counts_text = get_params_count_str(model)
    with open(os.path.join(log_dir, 'param_count.txt'), 'w') as f:
        f.write(param_counts_text)
        
def get_params_count(model, max_name_len: int = 60):
    params = [(name[:max_name_len], p.numel(), str(tuple(p.shape)), p.requires_grad) for name, p in model.named_parameters()]
    total_trainable_params = sum([x[1] for x in params if x[-1]])
    total_nontrainable_params = sum([x[1] for x in params if not x[-1]])
    return params, total_trainable_params, total_nontrainable_params

def get_params_count_str(model, max_name_len: int = 60):
    padding = 70  # Hardcoded depending on desired amount of padding and separators.
    params, total_trainable_params, total_nontrainable_params = get_params_count(model, max_name_len)
    param_counts_text = ''
    param_counts_text += '=' * (max_name_len + padding) + '\n'
    param_counts_text += f'| {"Module":<{max_name_len}} | {"Trainable":<10} | {"Shape":>15} | {"Param Count":>12} |\n'
    param_counts_text += '-' * (max_name_len + padding) + '\n'
    for name, param_count, shape, trainable in params:
        param_counts_text += f'| {name:<{max_name_len}} | {"True" if trainable else "False":<10} | {shape:>15} | {param_count:>12,} |\n'
    param_counts_text += '-' * (max_name_len + padding) + '\n'
    param_counts_text += f'| {"Total trainable params":<{max_name_len}} | {"":<10} | {"":<15} | {total_trainable_params:>12,} |\n'
    param_counts_text += f'| {"Total non-trainable params":<{max_name_len}} | {"":<10} | {"":<15} | {total_nontrainable_params:>12,} |\n'
    param_counts_text += '=' * (max_name_len + padding) + '\n'
    return param_counts_text


