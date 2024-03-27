from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers 
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)