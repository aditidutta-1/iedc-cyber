import pandas as pd

# Correct function for Excel files
data = pd.read_excel(r"C:\Users\User\Downloads\windows10_dataset.xlsx")

# Display the first few rows and columns for clarity
print(data.head())
print("Columns:", data.columns.tolist())

# Preprocess the data
# Convert timestamp to datetime if 'ts' column exists
if 'ts' in data.columns:
    data['ts'] = pd.to_datetime(data['ts'], unit='s')  # Assuming UNIX timestamp in seconds

# Remove rows with missing values
data.dropna(inplace=True)

# Define threat detection based on 'label' column
def detect_threats(df):
    threats = []

    # Check if 'label' column exists
    if 'label' in df.columns:
        suspicious_events = df[df['label'] == 'suspicious']  # change 'suspicious' if needed
        if not suspicious_events.empty:
            threats.append(f"Suspicious events detected: {len(suspicious_events)} rows.")
    else:
        threats.append("No 'label' column found in data.")

    return threats

# Run threat detection
threats_detected = detect_threats(data)

# Output the results
if threats_detected:
    for threat in threats_detected:
        print(threat)
else:
    print("No threats detected.")
    
    print(data['label'].unique())
    
    import matplotlib.pyplot as plt

# Count the occurrences of each label
label_counts = data['label'].value_counts()

# Plot as a bar chart
label_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Event Label Distribution")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

