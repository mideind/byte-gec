from transformers import pipeline

model = ""

gec_pipeline = pipeline("text2text-generation", model=model, tokenizer="google/byt5-base")

while True:
    sentence = input("Write a sentence: " + "\n")
    print(gec_pipeline(sentence.strip(), max_length=512)[0]["generated_text"] + "\n")
