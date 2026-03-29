#!/usr/bin/env python3
import argparse


def trim_fasta_headers(input_file, output_file):
    """Trim FASTA headers to first part before '|' character."""
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            if line.startswith(">"):
                # Trim header to first part before '|'
                trimmed_header = ">" + line[1:].split("|")[0] + "\n"
                outfile.write(trimmed_header)
            else:
                outfile.write(line)


def main():
    parser = argparse.ArgumentParser(
        description='Trim FASTA headers to first part before "|" character'
    )
    parser.add_argument("--input", required=True, help="Input FASTA file")
    parser.add_argument("--output", required=True, help="Output FASTA file")

    args = parser.parse_args()

    trim_fasta_headers(args.input, args.output)
    print(f"Processed {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
