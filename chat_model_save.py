
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM
master_model="OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov"


def llm():
    # model_name="OpenVINO/gemma-2b-it-int8-ov"
    model_name = master_model

    ov_model= OVModelForCausalLM.from_pretrained(model_name, device="GPU")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = pipeline(task="text-generation",model=ov_model, tokenizer=tokenizer, temperature=0.01, max_new_tokens=1500, do_sample=False, return_full_text=False)
    # model= HuggingFacePipeline(pipeline=model)
    return ov_model, tokenizer

if __name__ == "__main__":
    model, tokenizer = llm()
    model.save_pretrained("./models/qwen",from_pt=True)
    tokenizer.save_pretrained("./models/qwen_tokenizer",from_pt=True)