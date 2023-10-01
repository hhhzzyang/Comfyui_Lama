# Comfyui-Lama

a costumer node is realized to remove anything/inpainting anything from a picture by mask inpainting. Many thanks to brilliant work 🔥🔥🔥 of project [lama](https://github.com/advimman/lama) and [inpatinting anything](https://github.com/geekyutao/Inpaint-Anything)!

# Nodes
- LamaaModelLoad
- LamaApply
- YamlConfigLoader:  Because lama load models with yaml config file. So I add a extra node which could transfer yaml file to object which can easily handle in following process.
   
# Models requirement
- [big-lama](https://huggingface.co/hhhzzz/big-lama/resolve/main/big-lama.ckpt) (put into models folder of customer node)

# Example
- ## Oringinal picture
![1 little girl](https://github.com/hhhzzyang/Comfyui_Lama/assets/124335463/53054425-7afc-4149-8c05-8004d83ae5fc) 
- ## After Lama
![ComfyUI_temp_uitxc_00007_](https://github.com/hhhzzyang/Comfyui_Lama/assets/124335463/5c709e57-5409-450b-a3a9-b3f5e9f7fbd6) 
- ## After Lama+sd
![ComfyUI_temp_sflvn_00011_](https://github.com/hhhzzyang/Comfyui_Lama/assets/124335463/7aaaf333-df69-484e-a79d-1bb1a2863dfd) 
</p>

# To-do list


# Acknowledgement
- [lama](https://github.com/advimman/lama) 
- [inpatinting anything](https://github.com/geekyutao/Inpaint-Anything)


