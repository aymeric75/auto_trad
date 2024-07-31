def process_line(line):
    values = line.strip().split()
    
    if len(values) > 11:
        return values[:10] + [values[-1]]
    return values

# # Exemple de ligne
# line = "1 0.34 0 0.05 0.32 0.44 -0.44 -1.21 -1.87 -1.85 -1.62 -0.39 -0.27 -0.17 -0.05 0.09 0"
# result = process_line(line)
# print(result)  # Output: ['1', '0.34', '0', '0.05', '0.32', '0']

# Pour traiter un fichier
def process_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            processed_line = process_line(line)
            outfile.write(' '.join(processed_line) + '\n')

# Utilisation
input_file = 'input.txt'
output_file = 'output.txt'
process_file(input_file, output_file)
