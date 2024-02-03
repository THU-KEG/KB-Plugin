from llama import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel, PeftConfig, LoraConfig
import torch

def transpose(weight, fan_in_fan_out):
    return weight.T if fan_in_fan_out else weight

def merge_lora(model, adapter_config, adapters_weights):
    for key in adapter_weights:
        assert key.endswith(".weight"), key
    scaling = adapter_config.lora_alpha / adapter_config.r
    fan_in_fan_out = adapter_config.fan_in_fan_out
    for name, p in model.named_parameters():
        lora_A_key = ".".join(name.split(".")[:-1]) + ".lora_A." + name.split(".")[-1]
        if not isinstance(model, PeftModel):
            lora_A_key = "base_model.model." + lora_A_key
        if lora_A_key not in adapters_weights:
            continue
        lora_B_key = lora_A_key.replace("lora_A", "lora_B")
        lora_A, lora_B = adapters_weights[lora_A_key].to(p.device).type_as(p.data), adapters_weights[lora_B_key].type_as(p.data).to(p.device)
        p.data += transpose(lora_B @ lora_A, fan_in_fan_out) * scaling
        
def unmerge_lora(model, adapter_config, adapters_weights):
    for key in adapter_weights:
        assert key.endswith(".weight"), key
    scaling = adapter_config.lora_alpha / adapter_config.r
    fan_in_fan_out = adapter_config.fan_in_fan_out
    for name, p in model.named_parameters():
        lora_A_key = ".".join(name.split(".")[:-1]) + ".lora_A." + name.split(".")[-1]
        if not isinstance(model, PeftModel):
            lora_A_key = "base_model.model." + lora_A_key
        if lora_A_key not in adapters_weights:
            continue
        lora_B_key = lora_A_key.replace("lora_A", "lora_B")
        lora_A, lora_B = adapters_weights[lora_A_key].to(p.device).type_as(p.data), adapters_weights[lora_B_key].type_as(p.data).to(p.device)
        p.data -= transpose(lora_B @ lora_A, fan_in_fan_out) * scaling

peft_model_id = "/share/zjj/kb-plugin/kqapro/sft_llama/checkpoints/train_el_lora_1_3epoch/checkpoint-6077"
config = LoraConfig.from_pretrained(peft_model_id)
print(config)
filename = f"{peft_model_id}/adapter_model.bin"
adapter_weights = torch.load(
    filename, map_location="cpu"
)


# model = LlamaForCausalLM.from_pretrained(config.base_model_name_or_path)
model1 = LlamaForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.bfloat16, device_map=0)
model1 = PeftModel.from_pretrained(model1, peft_model_id)

# model = LlamaForCausalLM.from_pretrained(config.base_model_name_or_path)
model2 = LlamaForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.bfloat16, device_map=0)
model2 = PeftModel.from_pretrained(model2, peft_model_id).bfloat16()

np1, np2 = model1.named_parameters(), model2.named_parameters()
for (n1, p1), (n2, p2) in zip(np1, np2):
    if "lora" in n1 or "proj" not in n1:
        continue
    print(n1, p1.dtype, n2, p2.dtype)
    # print(p1)
    # print(p2)
    print((p1.data - p2.data).abs().max().item())


merge_lora(model1, config, adapter_weights)
model2.merge_adapter()

np1, np2 = model1.named_parameters(), model2.named_parameters()
for (n1, p1), (n2, p2) in zip(np1, np2):
    if "lora" in n1 or "proj" not in n1:
        continue
    print(n1, p1.dtype, n2, p2.dtype)
    # print(p1)
    # print(p2)
    print((p1.data - p2.data).abs().max().item())

