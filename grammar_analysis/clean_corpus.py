import os
import re

def is_grammar_point_start(lines, index):
    line = lines[index].strip()
    # Check if line is just a number
    if not re.match(r'^\d+$', line):
        return False
    
    # Look ahead for "基礎 第"
    for j in range(index + 1, len(lines)):
        next_line = lines[j].strip()
        if not next_line:
            continue
        if next_line.startswith('基礎 第') or next_line.startswith('進階 第'):
            return True
        else:
            return False
    return False

def main():
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Step 1: Filter out lines starting with ---
    filtered_lines = [line for line in lines if not line.strip().startswith('---')]

    # Step 2: Insert // separator
    final_lines = []
    grammar_point_count = 0
    
    i = 0
    while i < len(filtered_lines):
        line = filtered_lines[i]
        
        if is_grammar_point_start(filtered_lines, i):
            grammar_point_count += 1
            if grammar_point_count > 1:
                # Check if we need to ensure newline before //
                # The user just said "independent line of //"
                # Let's just add it.
                # If the previous line was not empty, maybe add a newline?
                # But the file has lots of empty lines, so likely there is one.
                # Let's just append //\n
                final_lines.append('//\n')
        
        final_lines.append(line)
        i += 1

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(final_lines)

    print(f"Processed {len(lines)} lines into {len(final_lines)} lines.")
    print(f"Found {grammar_point_count} grammar points.")

if __name__ == '__main__':
    main()
