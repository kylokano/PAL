export CUDA_VISIBLE_DEVICES=1,2
export HYDRA_FULL_ERROR=1

# Step 1: Multi task training
seed=42
python -u train.py model=gpt2-small  datasets=[persona_chat_revised] loss=sft exp_name=gpt2-mt-revised-seed${seed} gradient_accumulation_steps=4 batch_size=128 eval_batch_size=128 trainer=FSDPTrainer sample_during_eval=false n_epochs=10 lr=2e-5 data_mode=mixture do_first_eval=false seed=${seed} +is_chinese=false +is_llama3=false
# python -u train.py model=gpt2-small  datasets=[persona_chat_origin] loss=sft exp_name=gpt2-mt-origin-seed${seed} gradient_accumulation_steps=4 batch_size=128 eval_batch_size=128 trainer=FSDPTrainer sample_during_eval=false n_epochs=10 lr=2e-5 data_mode=mixture do_first_eval=false seed=${seed} +is_chinese=false +is_llama3=false
# python -u train.py model=gpt2-chinese-small datasets=[persona_chat] loss=sft exp_name=gpt2-mt-ch-seed${seed} gradient_accumulation_steps=4 batch_size=128 eval_batch_size=128 trainer=FSDPTrainer sample_during_eval=false n_epochs=10 lr=2e-5 data_mode=mixture do_first_eval=false seed=${seed} +is_chinese=true +is_llama3=false

# Generate samples for dpo
export CONFIG_PATH=<path to your checkpoint>
python -u decode.py strategy=beam_search num_beam=3 max_new_tokens=40 exp_name= evaluate_model_path=<path to your checkpoint> eval_batch_size=16 n_epochs=1 wandb.project=persona_dialogue_decode data_mode=predict_select_then_generate

# Step 2: Persona Alignment
python -u train_lora.py lr=1e-6 loss.beta=0.1 model=gpt2-small eval_every=5000 datasets=[persona_chat_origin] loss=dpo exp_name=gpt2-dpo-origin gradient_accumulation_steps=4 batch_size=128 eval_batch_size=128 trainer=FSDPTrainer sample_during_eval=false n_epochs=3 model.archive=<path to your checkpoint> data_mode=golden_select_then_generate dpo_data_mode=generate_for_dpo dpo_path=<path to your generated samples> do_first_eval=false +is_chinese=false +is_lora=true
# python -u train_lora.py lr=1e-6 loss.beta=0.1 model=gpt2-small eval_every=5000 datasets=[persona_chat_revised] loss=dpo exp_name=gpt2-dpo-revised gradient_accumulation_steps=4 batch_size=128 eval_batch_size=128 trainer=FSDPTrainer sample_during_eval=false n_epochs=3 model.archive=<path to your checkpoint> data_mode=golden_select_then_generate dpo_data_mode=generate_for_dpo dpo_path=<path to your generated samples> do_first_eval=false +is_chinese=false +is_lora=true
# python -u train_lora.py lr=1e-6 loss.beta=0.1 model=gpt2-chinese-small eval_every=5000 datasets=[persona_chat_ch] loss=dpo exp_name=gpt2-dpo-ch gradient_accumulation_steps=4 batch_size=32 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false n_epochs=3 model.archive=<path to your checkpoint> data_mode=golden_select_then_generate dpo_data_mode=generate_for_dpo dpo_path=<path to your generated samples> do_first_eval=false +is_chinese=true +is_lora=true

# Generate prediction
export CONFIG_PATH=<path to your checkpoint>
python -u decode.py strategy=beam_search num_beam=3 max_new_tokens=40 exp_name=gpt2-dpo-ch evaluate_model_path=${CONFIG_PATH}/LATEST/ +select_data_path=<path to your generated samples> eval_batch_size=16 n_epochs=1 data_mode=predict_select_then_generate