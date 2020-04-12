# Reads MatLab `mat` file.

import scipy.io


filename = "part mesh indices/1.mat"
content = scipy.io.loadmat(filename)
print(content)
print(content["cell_boxs_correspond_objSerialNumber"])
