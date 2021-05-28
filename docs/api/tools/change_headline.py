import argparse


def replacement1(title):

    title = " ".join(("DEPREC:", "Full", "(legacy)", "API", title))
    return title


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="replace first two line")

    parser.add_argument("files", type=str, nargs="*")
    parsed_args = parser.parse_args()

    n_files = 0
    for rest_file in parsed_args.files:
        with open(rest_file) as f:
            first_word = f.readline().strip().split()[0]
            if first_word == "raredecay":
                first_word = r"foobar\." + replacement1(first_word)
            elif not "." in first_word:
                continue
            replacement = first_word.split(".")[-1]
            underline = f.readline()[0] * len(replacement)
            lower_file = f.read()
        with open(rest_file, "w") as f:
            f.write("\n".join((replacement, underline, lower_file)))
        n_files += 1

    print(f"finished sucessfully parsing {n_files} files")
