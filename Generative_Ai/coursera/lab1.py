from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer,GenerationConfig

# huggingface_dataset="knkarthick/dialogsum"

# dataset=load_dataset(huggingface_dataset)
# print(dataset)


model_name="google/flan-t5-base"
model=AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer=AutoTokenizer.from_pretrained(model_name,use_fast=True)



# sentence_encoder=tokenizer(sentence,return_tensors="pt")
# sentence_decoder=tokenizer.decode(sentence_encoder["input_ids"][0],
#                     skip_special_tokens=True,)
# print("encode",sentence_encoder["input_ids"][0])
# print("decode",sentence_decoder)
while True:
    sentence=input("enter prompt\n")
    inputs=tokenizer(sentence,return_tensors="pt")
    output=tokenizer.decode(model.generate(
        inputs["input_ids"],max_new_tokens=50)[0],skip_special_tokens=True
    )


    print(output)
    print("*************end****************\n")

