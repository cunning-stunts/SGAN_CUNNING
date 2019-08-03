import os
import re
import sys

import imageio

from utils import get_random_string


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]


def main():
    run_id = sys.argv[1]
    max_duration = 30  # seconds
    min_time = 0.017  # seconds
    max_time = 0.2  # seconds
    overwrite = True
    if os.path.exists(f'{run_id}.gif'):
        overwrite = input("gif exists, overwrite? (y/n)")
        if overwrite.lower() != "y":
            overwrite = False

    if run_id is None or run_id == "":
        all_subdirs = [os.path.join("images", d) for d in os.listdir('images')]
        run_id = max(all_subdirs, key=os.path.getmtime)
    print(f"run_id: {run_id}")
    files = os.listdir(run_id)
    files.sort(key=natural_keys)
    [files.append(files[-1]) for _ in range(10)]
    print("loading images...")
    images = [imageio.imread(os.path.join(run_id, filename)) for filename in files]
    duration = min(max(max_duration / len(files), min_time), max_time)
    print(f"frame duration: {duration}")
    print(f"number of images: {len(files)}")
    print(f"gif length: {duration * len(files)} seconds")
    print("Saving gif...")
    if not overwrite:
        run_id = run_id + "_" + get_random_string(2)
    imageio.mimsave(f'{run_id}.gif', images, duration=duration)


if __name__ == '__main__':
    #    main("images/5NJ54IMP")
    main()
