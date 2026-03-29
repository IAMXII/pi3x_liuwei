import torch
from safetensors.torch import load_file, save_file

def force_replace_modules(base_model_path, inject_model_path, output_path):
    print("正在加载模型...")
    state_dict_base = load_file(base_model_path)
    state_dict_inject = load_file(inject_model_path)
    
    # 使用元组，方便 startswith 直接批量匹配
    target_modules = (
        "point_decoder.",
        "point_head.",
        "conf_decoder.",
        "conf_head."
    )
    
    # ==========================================
    # 步骤 1: 删掉基础模型中原有的这四个头的全部参数 (多了就删掉)
    # ==========================================
    keys_to_delete = [k for k in state_dict_base.keys() if k.startswith(target_modules)]
    for k in keys_to_delete:
        del state_dict_base[k]
    
    print(f"已从基础模型中清理掉 {len(keys_to_delete)} 个旧参数。")

    # ==========================================
    # 步骤 2: 将注入模型中的这四个头的参数全部移植过去 (替换/补上)
    # ==========================================
    keys_to_inject = [k for k in state_dict_inject.keys() if k.startswith(target_modules)]
    for k in keys_to_inject:
        # 直接赋值，自带新模型的 shape 和 dtype
        state_dict_base[k] = state_dict_inject[k]
        
    print(f"已将新模型中的 {len(keys_to_inject)} 个参数注入到基础模型中。")
    
    # ==========================================
    # 步骤 3: 保存
    # ==========================================
    print("-" * 30)
    if len(keys_to_inject) > 0:
        print(f"正在保存融合后的模型至 {output_path} ...")
        # safetensors 要求张量必须是 contiguous 的，保险起见调用一下 contiguous()
        state_dict_base = {k: v.contiguous() for k, v in state_dict_base.items()}
        save_file(state_dict_base, output_path)
        print("🎉 保存完毕！")
    else:
        print("⚠️ 在注入模型中没有找到目标模块的参数，未生成新模型。")

# ================= 运行配置 =================
if __name__ == "__main__":
    BASE_MODEL = "outputs/pi3_highres_0324/ckpts/best_model/model.safetensors"   # 基础模型
    INJECT_MODEL = "ckpts/pi3/model_pi3x.safetensors" # 提供新参数的模型
    OUTPUT_MODEL = "model_merged.safetensors" # 生成的新模型
    
    force_replace_modules(BASE_MODEL, INJECT_MODEL, OUTPUT_MODEL)