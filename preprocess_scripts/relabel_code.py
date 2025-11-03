import os

# Old and new class name lists
old_names = ['30C', '40C', 'DN_bleach', 'DN_dry_clean', 'DN_iron', 'DN_tumble_dry',
             'iron_low', 'iron_medium', 'normal_dry_clean_solvents', 'tumble_dry_low', 'z']

new_names = ['30C', '40C', 'DN_bleach', 'DN_dry_clean', 'DN_iron', 'DN_tumble_dry',
             'DN_wash', 'hand_wash', 'iron_low', 'iron_medium', 'non_chlorine_bleach',
             'normal_dry_clean_solvents', 'tumble_dry_low', 'tumble_dry_medium', 'z']

index_map = {old_idx: new_names.index(label) for old_idx, label in enumerate(old_names)}

labels_dir = "./labels"

for file in os.listdir(labels_dir):
    if not file.endswith(".txt"):
        continue

    file_path = os.path.join(labels_dir, file)
    updated_lines = []

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            old_class = int(parts[0])
            if old_class not in index_map:
                continue  # skip if class not in map
            new_class = index_map[old_class]
            updated_line = " ".join([str(new_class)] + parts[1:])
            updated_lines.append(updated_line)

    with open(file_path, 'w') as f:
        f.write("\n".join(updated_lines) + "\n")