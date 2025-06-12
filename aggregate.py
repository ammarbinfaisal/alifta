import json
import os
import re
from uuid import uuid4

def has_continuation(heading_text):
    """Check if the heading contains 'Continuation' (case-insensitive)."""
    return bool(re.search(r'continuation', heading_text, re.IGNORECASE))

def combine_continuation_sections(input_file, output_file):
    """Combine sections with 'Continuation' in heading into the previous non-continuation section."""
    # Read the input JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sections = data.get('sections', [])
    if not sections:
        return  # No sections to process
    
    new_sections = []
    current_main_section = None
    
    for section in sections:
        heading = section.get('heading', {})
        heading_text = heading.get('text', '')
        
        if not has_continuation(heading_text):
            # If this is a non-continuation section, add it as a new main section
            if current_main_section:
                new_sections.append(current_main_section)
            current_main_section = section.copy()
            current_main_section['content_items'] = section['content_items'].copy()
        else:
            # If this is a continuation section, append its content items to the current main section
            if current_main_section:
                current_main_section['content_items'].extend(section['content_items'].copy())
            else:
                # If there's no main section yet, treat this as a standalone section
                new_sections.append(section.copy())
    
    # Append the last main section if it exists
    if current_main_section:
        new_sections.append(current_main_section)
    
    # Update the data with the new sections
    data['sections'] = new_sections
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write the output JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def process_volume_directory(input_dir, output_dir):
    """Process all JSON files in the input directory structure vol/volX/volX.json."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Regular expression to match volX directories
    vol_pattern = re.compile(r'vol(\d+)')
    
    # Iterate through subdirectories in the input directory
    for dirname in os.listdir(input_dir):
        match = vol_pattern.match(dirname)
        if match:
            vol_num = match.group(1)
            input_path = os.path.join(input_dir, dirname, f'vol{vol_num}.json')
            output_path = os.path.join(output_dir, dirname, f'vol{vol_num}.json')
            
            if os.path.exists(input_path):
                print(f"Processing {input_path} -> {output_path}")
                combine_continuation_sections(input_path, output_path)
            else:
                print(f"File not found: {input_path}")

# Example usage
if __name__ == "__main__":
    input_directory = "vols"
    output_directory = "vol_combined"
    process_volume_directory(input_directory, output_directory)