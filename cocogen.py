import torch
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorForLanguageModeling


# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def fine_tune_model(train_dataset):
    train_texts = [doc for doc, _ in train_dataset]

    # Tokenize the training texts
    train_encodings = tokenizer(train_texts, return_tensors="pt", padding=True, truncation=True)

    train_dataset = torch.utils.data.TensorDataset(train_encodings["input_ids"], train_encodings["attention_mask"])

    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_dir="./logs",
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    return model


def load_compliance_documents(file_path):
    with open(file_path, "r") as f:
        compliance_documents = f.read().splitlines()
    return compliance_documents

def detect_log_format(log_line):
    # Use the pre-trained language model to detect log format based on headings
    inputs = tokenizer(log_line, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits).item()
    
    if predicted_class == 0:  # Example of a heading detection prompt
        return "csv"  # Replace with your format identifier
    # Add more format detection logic using different prompts if needed
    return None

def parse_log_file(file_path):
    logs = []
    with open(file_path, "r") as f:
        log_lines = f.readlines()
        if not log_lines:
            return logs
        
        log_format = detect_log_format(log_lines[0])  # Detect format using the first line
        if log_format == "csv":
            headings = log_lines[0].strip().split(",")
            for line in log_lines[1:]:
                parts = line.strip().split(",")
                log_entry = {heading: part for heading, part in zip(headings, parts)}
                logs.append(log_entry)
        # Add more parsing logic for different formats here
    return logs

def analyze_logs(logs, fine_tuned_model, compliance_documents):
    for log in logs:
        input_text = log["activity"] + " " + log["details"]
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        
        # Make predictions using the fine-tuned model
        with torch.no_grad():
            outputs = fine_tuned_model(**inputs)
        predicted_class = torch.argmax(outputs.logits).item()

        if predicted_class == 1:  # Compliance breach detected
            print("Compliance Breach Detected!")
            print("Citation: User ID:", log.get("user_id"))
            print("Timestamp:", log.get("timestamp"))
            print("Department:", log.get("department"))
            print("Role:", log.get("role"))
            
            # Determine which rules are violated
            violated_rules = [rule for rule in compliance_documents if rule.lower() in input_text.lower()]
            print("Violated Rules:", violated_rules)
            
            # Generate actionable insights based on violated rules
            for rule in violated_rules:
                print("Actionable Insight for", rule, ": Fix the issue based on compliance standards.")
        else:
            print("No violation detected.")

def main():
    if len(sys.argv) < 3:
        print("Enter file paths.")
        exit()
    
    # Load compliance documents from a file
    compliance_documents = load_compliance_documents(sys.argv[1])
    
    # Load and parse log files
    logs = parse_log_file(sys.argv[2])
    
    # Load and preprocess compliance training data
    train_compliance_documents = load_compliance_documents(sys.argv[1])
    train_labels = [1] * len(train_compliance_documents)  # Assuming all are non-compliant
    train_dataset = list(zip(train_compliance_documents, train_labels))
    
    # Fine-tune the model on the compliance dataset
    fine_tuned_model = fine_tune_model(train_dataset)
    
    # Analyze logs and generate compliance breach reports
    analyze_logs(logs, fine_tuned_model, compliance_documents)

# Load the pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(model_name)


# Call the main function
if __name__ == "__main__":
    main()
