import os
import pandas

os.mkdir("clean")

filenames = sorted([f for f in os.listdir("raw")])


def process_file(filename):
    print(f"Processing {filename}...")
    # df = pandas.read_csv(filename, sep=",", header=None)

    new_filename = os.path.join("clean", filename + ".csv")
    # df.to_csv(new_filename, index=False)
