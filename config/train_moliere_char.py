out_dir = "out-moliere-char"
eval_interval = 200
eval_iters = 50
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = "nanogpt-bitnet-moliere"
wandb_run_name = "moliere-char-training"

dataset = "moliere"
gradient_accumulation_steps = 1
batch_size = 16
block_size = 128


# architecture of the model :

n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.2
bias = True

# Optimizer

learning_rate = 1e-3
max_iters = 3000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

decay_lr = True
warmup_iters = 100
lr_decay_iters = 3000
min_lr = 1e-4




device = "cuda"
dtype = "float32"
compile = False

seed = 1337