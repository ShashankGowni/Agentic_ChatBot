import os
import glob

# Directory of your project
project_directory = './src/lang_graph_chatbot'

# Get all Python files recursively in the directory
files = glob.glob(os.path.join(project_directory, '**', '*.py'), recursive=True)

# Combine all code into one file
with open('combined_code.py', 'w', encoding='utf-8') as output_file:
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            output_file.write(f"### Start of {file} ###\n")
            output_file.write(f.read())
            output_file.write(f"\n### End of {file} ###\n\n")

print("All code has been written to 'combined_code.py'.")
