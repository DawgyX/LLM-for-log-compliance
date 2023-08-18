import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the pre-trained model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Mock Compliance Documents (Ruleset)
password_policy = {
    "rule": "Password Policy",
    "requirements": ["at least 8 characters", "one uppercase letter", "one digit"]
}

access_control_policy = {
    "rule": "Access Control",
    "allowed_roles": ["Admin"]
}

# Mock Logs and System Policies
logs = [
    {"user_id": "john_doe", "action": "Failed login attempt", "message": "Invalid password."},
    {"user_id": "alice_smith", "action": "Successful login", "message": "Welcome, Alice!"}
]

system_policies = [
    {"role": "Admin", "access": "Sensitive data"},
    {"role": "User", "access": "Basic data"}
]

# Analysis and Reporting
for log in logs:
    input_text = log["message"]  # Use log message as input text

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_class = torch.argmax(outputs.logits).item()

    if "Failed login attempt" in log["action"]:
        # Check Password Policy
        if any(req not in log["message"] for req in password_policy["requirements"]):
            print("Compliance Breach Detected!")
            print("Violated Rule:", password_policy["rule"])
            print("Citation: Log File, Line:", logs.index(log) + 1)
            print("User ID:", log["user_id"])
            print("Actionable Insight: Improve password to meet policy requirements.")
        else:
            print("No violation detected.")

    print("Predicted Class:", predicted_class)

for policy in system_policies:
    input_text = policy["role"]  # Use policy role as input text

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_class = torch.argmax(outputs.logits).item()

    if policy["role"] not in access_control_policy["allowed_roles"]:
        print("Compliance Breach Detected!")
        print("Violated Rule:", access_control_policy["rule"])
        print("Citation: System Policy:", policy["role"])
        print("Actionable Insight: Users with Admin role should have access to sensitive data.")
    else:
        print("No violation detected.")

    print("Predicted Class:", predicted_class)
