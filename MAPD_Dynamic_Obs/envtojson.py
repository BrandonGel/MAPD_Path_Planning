import yaml
import json
import re

# Specify the file paths
yaml_file_path = 'Environments/environment.yaml_tmp'
json_file_path = 'Final/env.json'

def preprocess_yaml(yaml_content):
    # Remove !!python/tuple tags and replace with lists
    return re.sub(r'!!python/tuple ', '', yaml_content)

# Read YAML data from the file
with open(yaml_file_path, 'r') as yaml_file:
    yaml_content = yaml_file.read()

# Preprocess YAML content
clean_yaml_content = preprocess_yaml(yaml_content)

# Load YAML data from the cleaned content
data = yaml.safe_load(clean_yaml_content)

# Convert to JSON
json_data = json.dumps(data, indent=2)

# Write JSON data to a file
with open(json_file_path, 'w') as json_file:
    json_file.write(json_data)

# Load YAML file
with open('output.yaml', 'r') as yaml_file:
    yaml_data = yaml.safe_load(yaml_file)

# Convert to JSON
with open('Final/schedule.json', 'w') as json_file:
    json.dump(yaml_data, json_file, indent=4)

print("YAML to JSON conversion done")