import os

root_path = "/root/FastMRI_challenge"
python_path = "/root/anaconda3/envs/mri/bin/python"
train_path = os.path.join(root_path, "train.py")
cmd_file_path = os.path.join(root_path, "train.sh")

# frequently change
special_name = "base"
epoch = 5 # 실제로 돌아가는 에폭 수
retrain = False
retrain_epoch = 5
acc_only_list = [
    4, 
    8
    ]  # 0 for all acc
anatomy_only_list = [
    'brain', 
    'knee',
    ]
slice_moe = 1
lr = 3e-4
scheduler = "cosine"
criterion = "AnatomicalSSIM"
report_interval = 100
use_random_mask = False
random_mask_prop = 0.5
betas = "0.9 0.999"


# Training hyperparameters
batch = 1
accumulation_step = 2
start_epoch = retrain_epoch if retrain else 0 # 처음이면 0, retrain이면 retrain_epoch과 똑같이 설정
# criterion = "AnatomicalSSIM_L1"


# Model hyperparameters
model = 'fivarnet'
feature_cascades = 8
image_cascades = 2
attention_stride = 0
chans = 32
sen_chans = 8

# MoE hyperparameters
use_moe = True
class_split_path = os.path.join(root_path, "class_indices")

# K-Fold hyperparameters
k_fold = True
num_folds = 5

# Data hyperparameters
data_root = "/root/Data"
data_path_train = os.path.join(data_root, "train")
data_path_val = os.path.join(data_root, "val")
data_augmentation = False

# Scheduler hyperparameters
# scheduler = "constant"
# lr = 3e-5
lr_min1 = 0.00005       
lr_max2 = 0.00015      
lr_min2 = 0.0         
warmup1 = 10   
anneal1 = 40           
warmup2 = 10           
anneal2 = 40           

# Saving hyperparameters
seed = 2025
result_path = os.path.join(root_path, "results")
architecture_part = f"{model}_f{feature_cascades}_i{image_cascades}_attn{attention_stride}_c{chans}_s{sen_chans}"
architecture_part += f"{'_' if len(special_name) != 0 else ''}{special_name}"
scenario_part = f"epoch{epoch}_fold{num_folds}_slice{slice_moe}{'_retrain' if retrain else ''}"

model_name = f"{architecture_part}__{scenario_part}"

instruction_template = [[(
    f"{python_path} -u {train_path} "
    f"-g {gpu_num} "
    f"--acc-only {acc_only} "
    f"--anatomy-only {anatomy_only} "
    f"-b {batch} "
    f"--start-epoch {start_epoch} "
    f"-e {epoch} "
    f"--accumulation-step {accumulation_step} "
    f"--criterion {criterion} "
    f"--retrain {retrain} "
    f"--retrain-epoch {retrain_epoch} "
    f"--use-random-mask {use_random_mask} "
    f"--random-mask-prop {random_mask_prop} "
    f"--betas {betas} "
    f"--model {model} "
    f"-f {feature_cascades} "
    f"-i {image_cascades} "
    f"-a {attention_stride} "
    f"--chans {chans} "
    f"--sens-chans {sen_chans} "
    f"--use-moe {use_moe} "
    f"--slice-moe {slice_moe} "
    f"--class-split-path {class_split_path} "
    f"--k-fold {k_fold} "
    f"--num-folds {num_folds} "
    f"--data-augmentation {data_augmentation} "
    f"--data-path-train {data_path_train} "
    f"--data-path-val {data_path_val} "
    f"--scheduler {scheduler} "
    f"-m {model} "
    f"--lr {lr} "
    f"--lr-min1 {lr_min1} "
    f"--lr-max2 {lr_max2} "
    f"--lr-min2 {lr_min2} "
    f"--warmup1 {warmup1} "
    f"--anneal1 {anneal1} "
    f"--warmup2 {warmup2} "
    f"--anneal2 {anneal2} "
    f"--seed {seed} "
    f"--result-path {result_path} "
    f"--report-interval {report_interval} "
    f"-n {model_name}")
    for anatomy_only in anatomy_only_list]
    for gpu_num, acc_only in enumerate(acc_only_list)
    ]


architecture_part, scenario_part = model_name.split("__")

log_path = [[os.path.join(result_path, architecture_part, scenario_part, f"acc{acc}-{anatomy}.log")
            for anatomy in anatomy_only_list]
            for acc in acc_only_list]


full_instruction = list()
result_dir = os.path.join(result_path, architecture_part, scenario_part)
for acc_inst_list, acc_log_list in zip(instruction_template, log_path):
    acc_inststruction = "nohup bash -c \" \n"
    acc_inststruction += f"mkdir -p {result_dir}; \\ \n"
    for idx, (anatomy_inst, anatomy_log) in enumerate(zip(acc_inst_list, acc_log_list)):
        anatomy_inst = anatomy_inst.replace('\\', '/')
        anatomy_log = anatomy_log.replace('\\', '/')
        acc_inststruction += f"{anatomy_inst} > {anatomy_log} 2>&1"
        if idx + 1 != len(acc_inst_list):
            acc_inststruction += '; \\\n'
        else:
            acc_inststruction += '\n'
    acc_inststruction += "\" > /dev/null 2>&1 &\n"
    full_instruction.append(acc_inststruction)

with open(cmd_file_path, 'w') as f:
    for inst in full_instruction:
        f.write(inst)

