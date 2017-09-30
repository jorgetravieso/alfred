

with open("test3.tsv") as in_file:
    with open("test3.tsv", "w+") as out:
        for line in in_file.readlines():
            split = line.split("\t")
            label = 'n' if split[0] is '0' else 'y'
            out.write(split[1] + "\t" + split[2] + "\t" + label + "\n")