import csv

input_file_path = 'C:\\Users\\david\\OneDrive\\Desktop\\Stig App\\STIG APP 4\\privateGPT-main\\privateGPT-main\\source_documents\\the bigegest of bitches.csv'
output_file_path = 'C:\\Users\\david\\OneDrive\\Desktop\\Stig App\\STIG APP 4\\privateGPT-main\\privateGPT-main\\source_documents\\cleaned_file.csv'

with open(input_file_path, 'r', newline='', encoding='ISO-8859-1') as infile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames

    with open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            print(row)  # Add this line to see what the row looks like
            cleaned_row = {k: (v.strip() if isinstance(v, str) else '') for k, v in row.items() if k is not None}
            writer.writerow(cleaned_row)




