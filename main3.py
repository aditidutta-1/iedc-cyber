import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv(r"C:\Users\User\Downloads\Global_Cybersecurity_Threats_2015-2024 (1).csv")

# Plot frequency of attack types
df['Attack Type'].value_counts().plot(kind='bar', color='tomato')
plt.title("Frequency of Attack Types")
plt.xlabel("Attack Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
