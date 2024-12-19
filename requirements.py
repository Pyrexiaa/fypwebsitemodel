import subprocess

# Run pip freeze and capture the output
result = subprocess.run(["pip", "freeze"], capture_output=True, text=True)
lines = result.stdout.splitlines()

# Filter out lines containing 'file://'
cleaned_lines = [line for line in lines if 'file://' not in line]

# Write to requirements.txt
with open("requirements.txt", "w") as file:
    file.write("\n".join(cleaned_lines))
