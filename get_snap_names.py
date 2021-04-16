import glob as glob

files = glob.glob("snapshots/*.*.data*")
files = [".".join(f.split(".")[:-1]) for f in files]
files = sorted(files, key=lambda x: int(x.split("-")[-1]))

with open("tmp", "w") as tmp:
    for f in files:
        tmp.write(f + "\n")
