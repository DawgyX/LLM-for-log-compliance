import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


input_text = "Sample log entry or policy statement."

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# Make predictions
with torch.no_grad():
    outputs = model(**inputs)

predicted_class = torch.argmax(outputs.logits).item()

print("Predicted Class:", predicted_class)
