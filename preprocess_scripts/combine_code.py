import os

def combine_text_files(input_folder, output_file):
    with open(output_file, 'w') as outfile:
        for filename in sorted(os.listdir(input_folder)):
            if filename.endswith(".txt"):
                file_path = os.path.join(input_folder, filename)
                with open(file_path, 'r') as infile:
                    outfile.write(f"{filename}\n")
                    outfile.write(infile.read())
                    outfile.write("\n\n")

combine_text_files('./labels', 'combined_output.txt')
