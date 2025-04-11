ROBERTA_DIM = 768

SIMILAR_TOPK_1 = 5 
SAME_KEYWORD_THRESHOLD = 2

class TRAIN_Args():
    text_max_len = 256
    D = 768
    D_2 = 384
    R = 3 
    device = "cuda:7"
    warmup_step = 128
    logging_steps = 3
    save_steps = 12
    eval_steps = 12
    train_data_start_id = 0
    train_data_end_id = 200
    model_name_or_path = 'path2model'
    save_dir = "path2save_ckpt"
    train_data_file = "path2train_data"
    cache_data_path = "path2cache_data"
    data_split = [0.8, 0.1, 0.1]
    mlp_dropout_rate = 0.2
    linear_dropout_rate = 0.2
    batch_size = 12
    gradient_accumulation_steps = 3
    train_epochs = 2
    LM_lr = 1e-5
    LM_lr_str = "1e-5"
    MLP_lr = 1e-5
    MLP_lr_str = "1e-5"
    weight_decay = 1e-5
    max_grad_norm = 1.0
    adam_epsilon = 1e-8

ARGS = TRAIN_Args()

board_file = "path2board"
negative_ratio = 7
max_extract_keyword_try = 3
mock_embedding = [0.0] * ARGS.D