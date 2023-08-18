import torch, sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def load_compliance_documents(file_path):
    with open(file_path, "r") as f:
        compliance_documents = f.read().splitlines()
    return compliance_documents

def parse_log_file(file_path):
    logs = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            log_entry = {
                "timestamp": parts[0],
                "user_id": parts[1],
                "activity": parts[2],
                "details": parts[3],
                "ip_address": parts[4],
                "department": parts[5],
                "role": parts[6]
            }
            logs.append(log_entry)
    return logs

def analyze_logs(logs, compliance_model, compliance_documents):
    for log in logs:
        input_text = log["activity"] + " " + log["details"]  # Combine activity and details
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

        # Make predictions
        with torch.no_grad():
            outputs = compliance_model(**inputs)
        predicted_class = torch.argmax(outputs.logits).item()

        # Check compliance with your standards
        if any(rule.lower() in input_text.lower() for rule in compliance_documents):
            print("Compliance Breach Detected!")
            print("Citation: User ID:", log["user_id"])
            print("Timestamp:", log["timestamp"])
            print("Department:", log["department"])
            print("Role:", log["role"])
            print("Actionable Insight: Fix the issue based on compliance standards.")
        else:
            print("No violation detected.")


def main():
    if len(sys.argv)<3: 
        print("Enter file paths.")
        exit()
    
    # Load compliance documents from a file
    compliance_documents = load_compliance_documents(sys.argv[1])
    
    # Load and parse log files
    logs = parse_log_file(sys.argv[2])
    
    # Analyze logs and generate compliance breach reports
    analyze_logs(logs, model, compliance_documents)

# Call the main function
if __name__ == "__main__":
    main()
