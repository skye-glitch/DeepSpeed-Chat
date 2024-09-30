import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',
                        nargs='*',
                        default=['./'],
                        help='Path to save the hostfile')
    parser.add_argument('--slots',
                        type=int,
                        default=3,
                        help="slot number.",)
    args = parser.parse_args()
    return args
args = parse_args()
fname = args.output_dir[0]+"/hostfile"
destination= open(fname, "w")
slotstr = " slots=" + str(args.slots) + "\n"
with open("/tmp/hostfile", "r") as f:
    Lines = f.readlines()
    for line in Lines:
        line = line.rstrip()
        destination.write(line + slotstr)