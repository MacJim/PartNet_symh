# Reads MatLab `mat` file.

import os

import scipy.io


def print_mat_file_content(filename: str):
    content = scipy.io.loadmat(filename)
    print(content)


folders = ["ops", "part mesh indices", "boxes", "labels", "syms"]

for folder in folders:
    print("Folder ", folder, ":", sep="")
    all_filenames = os.listdir(folder)
    for filename in all_filenames:
        if (filename.endswith(".mat")):
            full_filename = os.path.join(folder, filename)
            print_mat_file_content(full_filename)

    print()
        
