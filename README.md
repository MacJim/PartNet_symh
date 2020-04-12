# The PartNet Symmetry Hierarchy Dataset (PartNet-Symh)

## Introduction

The **PartNet-Symh dataset** augments the **PartNet dataset** by adding a **recursive hierarchical organization** of fine-grained parts for each shape. The PartNet dataset was originally proposed in the paper "*PartNet: A Large-scale Benchmark for Fine-grained and Hierarchical Part-level 3D Object Understanding*". The hierarchical organization follows the symmetry hierarchy defined in [Wang et al. 2011]. The symmetry hierarchies were used to train the **PartNet model** proposed in our paper "*PartNet: A Recursive Part Decomposition Network for Fine-grained and Hierarchical Shape Segmentation*". In general, PartNet-Symh can be used to train any model for encoding/decoding part-based structures based on Recursive Neural Networks, e.g., GRASS [Li et al. 2017].

## Basic Information

The dataset contains *22369* 3D shapes covering *24* shape categories. See ***Table 1*** for the statistics of the dataset.

|  category  |  Bag   |  Bed   |  Bottle   |  Bowl   |  Chair   |  Clock   |  Display   |  Door   |  Faucet   |  Hat   |  Keyboard   |  Knife   |  Lamp   |  Laptop   |  Microwave   |  Mug   |  Refrigerator   |  Scissors   |  Storage   |  Table   |  TrashCan   |  Vase   |  Dishwasher   |  Earphone   |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| # shapes |  154   |  147   |  511   |  102   |  6201   |  426   |  329   |  198   |  826   |  251   |  109   |  486   |  2603   |  92   |  81   |  231   |  164   |  112   |  2598   |  5868   |  296   |  180   |  135   |  269   |
| # parts |  343   |  3534   |  1432   |  213   |  38858   |  1152   |  1174   |  586   |  3986   |  588   |  5690   |  1571   |  12567   |  270   |  346   |  529   |  671   |  394   |  35507   |  30408   |  2580   |  447   |  928   |  1193   |
| max.# parts per shape |  4   |  89   |  5   |  3   |  30   |  8   |  5   |  9   |  18   |  3   |  64   |  5   |  128   |  3   |  8   |  4   |  8   |  5   |  101   |  50   |  43   |  4   |  8   |  8   |
| min.# parts per shape |  2   |  4   |  2   |  2   |  2   |  2   |  2   |  2   |  2   |  2   |  13   |  2   |  2   |  2   |  3   |  2   |  2   |  2   |  2   |  2   |  2   |  2   |  2   |  2   |

***Table 1. Statistics of the PartNet-Symh dataset.***

## Explaining the Dataset with an Example

Let us use a chair as an example to illustrate how our data is organized. We first show a figure to illustrate how to represent a part-based model with a symmetry hierarchy. We then explain the details of data organization.

### 1. Symmetry hierarchy

![image](./symh.png)
***Figure 1. A chair model is represented with a symmetry hierarchy which is a top-down recursive decomposition into its constituent parts.***

As shown in ***Figure 1***, a chair model is represented with a recursive symmetry hierarchy. Each leaf node represents a part. There are three types of nodes in the hierarchy: **Type 0 -- Leaf nodes** (e.g. *node 2*), **Type 1 -- Adjacency nodes** (e.g. *node 11*, indicating the proximity relations between two adjacent parts), and **Type 2 -- Symmetry nodes** (e.g. *node 9* or *node 12*, which represents either a **reflectional or a rotational symmetry** relations of multiple parts). A symmetry node stores the parameters (e.g., reflection axis) of the corresponding symmetry. Please refer to [Wang et al. 2011] and [Li et al. 2017] for more detailed definition of symmetry hierarchy.

We ensure all shapes belonging to the same shape category share the same high-level structure of symmetry hierarchy. This means that those shapes have consistency in the **top few levels** of their symmetry hierarchies [van Kaick et al. 2013]. These levels generally correspond to **semantically meaningful major parts**. For example, a chair model is composed of a back, a seat, a leg and an armrest, and the symmetry hierarchies are consistent at the level of these parts across all chairs.

### 2. Data organization

There are seven folders for each model.

#### ops

Each mat file in this folder stores the corresponding types of the nodes of a symmetry hierarchy. Taking the symmetry hierarchy in  ***Figure 1 (b)*** for example, ***Table 2*** gives the node types in a [depth-first traversing order with vertex postorderings](https://en.wikipedia.org/wiki/Depth-first_search).

|  node  | *node 7*  | *node 3* | *node 4*    |  *node 11*   | *node 12*  | *node 13*  | *node 6* | *node 1* | *node 2* | *node 8* | *node 9* | *node 10* | *node 14* | *node 5* | *node 15* |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| type | 0 | 0 | 0 | 1 | 2 | 1 | 0 | 0 | 0 | 1 | 2 | 1 | 1 | 0 | 1 |

***Table 2. Node type (0 -- leaf node, 1 -- adjacency node, and 2 -- symmetry node) of the nodes in Figure 1 (b).***

Files in this directory look like:

```python
{'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Tue Oct 15 15:09:20 2019', '__version__': '1.0', '__globals__': [], 'op': array([[0, 0, 0, 1, 2, 1, 0, 1]], dtype=uint8)}
{'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Tue Oct 15 15:09:20 2019', '__version__': '1.0', '__globals__': [], 'op': array([[0, 0, 0, 1, 2, 1, 0, 0, 1, 0, 0, 1, 2, 1, 1]], dtype=uint8)}
{'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Tue Oct 15 15:09:20 2019', '__version__': '1.0', '__globals__': [], 'op': array([[0, 0, 1, 0, 1]], dtype=uint8)}
```

#### part mesh indices

The mat file under this folder stores the **part mesh indices** corresponding to the **leaf** nodes of a symmetry hierarchy.
Taking the symmetry hierarchy in  ***Figure 1 (b)*** for example, ***Table 3*** gives the part mesh indices of leaf nodes in the same order as above.

|  leaf node  | *node 7*  | *node 3* |  *node 4* | *node 6* | *node 1* | *node 2* | *node 5* |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |---- |
| part mesh indices | 6 | 5 | 4 | 7 | 16 | 9 | 1 |

***Table 3. part mesh indices for leaf nodes.***

For example, you can find the sixth part mesh for ***node 7*** in 'objs' class from result_after_merging.json file for this shape. Note that this json file can be found in dataset from [[Mo et al 2019]](https://cs.stanford.edu/~kaichun/partnet/).

Files in this directory look like:

```python
{'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Tue Oct 15 15:09:20 2019', '__version__': '1.0', '__globals__': [], 'cell_boxs_correspond_objSerialNumber': array([[array([[2]], dtype=uint8), array([[3]], dtype=uint8),
        array([[6]], dtype=uint8), array([[1]], dtype=uint8)]],
      dtype=object), 'shapename': array(['1127'], dtype='<U4')}
{'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Tue Oct 15 15:09:20 2019', '__version__': '1.0', '__globals__': [], 'cell_boxs_correspond_objSerialNumber': array([[array([[8]], dtype=uint8), array([[18, 19, 20]], dtype=uint8),
        array([[15, 16, 17]], dtype=uint8), array([[1, 2]], dtype=uint8),
        array([[5]], dtype=uint8), array([[3]], dtype=uint8),
        array([[4]], dtype=uint8)]], dtype=object), 'shapename': array(['1277'], dtype='<U4')}
{'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Tue Oct 15 15:09:20 2019', '__version__': '1.0', '__globals__': [], 'cell_boxs_correspond_objSerialNumber': array([[array([[5, 6, 7, 8]], dtype=uint8),
        array([[ 9, 10]], dtype=uint8),
        array([[1, 2, 3, 4]], dtype=uint8)]], dtype=object), 'shapename': array(['1095'], dtype='<U4')}
```

#### boxes

The mat files under this folder stores the parameters of the **part bounding boxes** corresponding to the **leaf** nodes of a symmetry hierarchy.

Files in this directory look like:

```python
{'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Tue Oct 15 15:09:20 2019', '__version__': '1.0', '__globals__': [], 'box': array([[-1.59500e-04, -2.84819e-01, -2.77423e-01, -1.59500e-04],
       [-4.82040e-02, -3.57503e-01, -3.59754e-01,  4.35843e-01],
       [ 6.26015e-02, -5.17655e-02,  5.30526e-01, -2.07346e-01],
       [ 1.51730e-01,  5.40683e-01,  5.36180e-01,  8.64469e-01],
       [ 6.56509e-01,  4.27775e-01,  4.28064e-01,  1.16616e-01],
       [ 6.51043e-01,  8.17240e-02,  9.65160e-02,  6.51043e-01],
       [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
       [ 1.00000e+00,  1.00000e+00,  1.00000e+00,  1.00000e+00],
       [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
       [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
       [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00],
       [ 1.00000e+00,  1.00000e+00,  1.00000e+00,  1.00000e+00]])}
{'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Tue Oct 15 15:09:20 2019', '__version__': '1.0', '__globals__': [], 'box': array([[ 1.12725e-02, -2.26820e-01, -2.26820e-01, -2.10626e-02,
         3.51950e-03,  2.64826e-01,  1.36769e-01],
       [-1.50399e-01, -5.45476e-01, -5.45476e-01,  4.58107e-01,
         1.89732e-01,  1.89732e-01,  1.89732e-01],
       [ 8.99520e-02,  3.92704e-01, -1.02912e-01, -1.37087e-01,
        -1.91723e-01, -1.40932e-01, -1.79685e-01],
       [ 8.37250e-02,  7.06427e-01,  7.06427e-01,  2.17306e-01,
         5.96538e-01,  5.96538e-01,  5.96538e-01],
       [ 7.52022e-01,  9.41900e-02,  9.41900e-02,  4.67223e-01,
         5.19760e-02,  5.23280e-02,  5.22400e-02],
       [ 6.17469e-01,  9.94230e-02,  9.94230e-02,  6.88609e-01,
         5.19750e-02,  5.23280e-02,  5.22400e-02],
       [ 0.00000e+00,  0.00000e+00,  0.00000e+00, -2.14185e-17,
         0.00000e+00,  0.00000e+00,  0.00000e+00],
       [ 1.00000e+00,  1.00000e+00,  1.00000e+00,  1.00000e+00,
         1.00000e+00,  1.00000e+00,  1.00000e+00],
       [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,
         0.00000e+00,  0.00000e+00,  0.00000e+00],
       [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  1.04528e-01,
         0.00000e+00,  0.00000e+00,  0.00000e+00],
       [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  2.23884e-18,
         0.00000e+00,  0.00000e+00,  0.00000e+00],
       [ 1.00000e+00,  1.00000e+00,  1.00000e+00,  9.94522e-01,
         1.00000e+00,  1.00000e+00,  1.00000e+00]])}
{'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Tue Oct 15 15:09:20 2019', '__version__': '1.0', '__globals__': [], 'box': array([[ 0.0030865 ,  0.00419755,  0.0302842 ],
       [ 0.148677  , -0.422335  ,  0.488309  ],
       [ 0.0724135 ,  0.169623  , -0.315397  ],
       [ 0.223845  ,  0.523685  ,  0.781198  ],
       [ 0.805681  ,  1.15351   ,  0.305279  ],
       [ 0.890567  ,  0.898086  ,  0.89214   ],
       [ 0.        , -0.0121707 ,  0.0994548 ],
       [ 1.        ,  0.770206  ,  0.981659  ],
       [ 0.        , -0.637679  , -0.162646  ],
       [ 0.        , -0.00438424, -0.00601635],
       [ 0.        ,  0.637679  ,  0.164046  ],
       [ 1.        ,  0.77029   ,  0.986434  ]])}
```

#### labels

The mat file under this folder stores the semantic label for each leaf node. ***Table 4*** gives the node labels of a chair model, *node 5* is the back part of the chair, labeled as 0. *node 7* is the seat, labeled as 1. *node 1*, *node 2* and *node 6* are the leg parts labeled as 2. *node 3* and *node 4* represent the armrests labeled as 3.  

|  node  | *node 7*  | *node 3* |  *node 4* | *node 6* | *node 1* | *node 2* | *node 5* |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| label | 1 | 3 | 3 | 2 | 2 | 2 | 0 |

***Table 4. Node labels (0 -- back, 1 -- seat, 2 -- leg, and 3 -- armrest).***

Files in this directory look like:

```python
{'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Tue Oct 15 15:09:20 2019', '__version__': '1.0', '__globals__': [], 'label': array([[1, 2, 2, 0]], dtype=uint8)}
{'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Tue Oct 15 15:09:20 2019', '__version__': '1.0', '__globals__': [], 'label': array([[1, 2, 2, 0, 0, 0, 0]], dtype=uint8)}
{'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Tue Oct 15 15:09:20 2019', '__version__': '1.0', '__globals__': [], 'label': array([[1, 2, 0]], dtype=uint8)}
```

#### syms

The mat file in the syms folder stores the symmetry parameters for each symmetry node. The symmetry parameters defined for each symmetry type can be found in [Li et al. 2017].

Files in this directory look like:

```python
{'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Tue Oct 15 15:09:20 2019', '__version__': '1.0', '__globals__': [], 'shapename': array(['1127'], dtype='<U4'), 'sym': array([[ 0.00000e+00],
       [ 9.99830e-01],
       [-1.84221e-02],
       [-6.76412e-05],
       [-2.77250e-04],
       [-3.62745e-01],
       [-5.17847e-02],
       [ 0.00000e+00]])}
{'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Tue Oct 15 15:09:20 2019', '__version__': '1.0', '__globals__': [], 'shapename': array(['1277'], dtype='<U4'), 'sym': array([[ 0.        ,  0.        ],
       [ 1.        ,  0.999931  ],
       [-0.        ,  0.        ],
       [-0.        , -0.011787  ],
       [ 0.0112725 ,  0.00415425],
       [-0.545476  ,  0.189732  ],
       [ 0.392704  , -0.137859  ],
       [ 0.        ,  0.        ]])}
{'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Tue Oct 15 15:09:20 2019', '__version__': '1.0', '__globals__': [], 'shapename': array(['1095'], dtype='<U4'), 'sym': array([[0],
       [0],
       [0],
       [0],
       [0],
       [0],
       [0],
       [0]], dtype=uint8)}
```

#### models

The models folder stores the 3D mesh models in .obj format.

#### obbs

The obbs folder stores the whole shape obb for each model, which contains original part obb, adjacent part relations and symmetric parameters.

## Downloading

The dataset can be downloaded from [here](https://www.dropbox.com/sh/el63rv14d01mk89/AAANX5fxfZ5vV7ygTmj-I_ema?dl=0).

## Read data

Code for reading symmetry hierarchies and part bounding boxes can be found [here](https://github.com/PeppaZhu/grass).

Code for sampling point cloud from part mesh, reading symmetry hierarchies and part point clouds can be found [here](https://github.com/FoggYu/PartNet).

## Reference

**[Wang et al. 2011]** Yanzhen Wang, Kai Xu, Jun Li, Hao Zhang, Ariel Shamir, Ligang Liu, Zhi-Quan Cheng, and Yueshan Xiong, "Symmetry Hierarchy of Man-Made Objects", Computer Graphics Forum (Special Issue of Eurographics 2011), 30(2): 287-296.

**[van Kaick et al. 2013]** Oliver van Kaick, Kai Xu, Hao Zhang, Yanzhen Wang, Shuyang Sun, Ariel Shamir and Daniel Cohen-Or, "Co-Hierarchical Analysis of Shape Structures", ACM Transactions on Graphics (SIGGRAPH 2013), 32(4).

**[Li et al. 2017]** Jun Li, Kai Xu, Siddhartha Chaudhuri, Ersin Yumer, Hao Zhang and Leonidas Guibas, "GRASS: Generative Recursive Autoencoders for Shape Structures", ACM Transactions on Graphics (SIGGRAPH 2017), 36(4).

## Citation

If you use this dataset, please cite the following papers.
```
@InProceedings{Yu_2019_CVPR,
    title = {{PartNet}: A Recursive Part Decomposition Network for Fine-grained and Hierarchical Shape Segmentation},
    author = {Fenggen Yu and Kun Liu and Yan Zhang and Chenyang Zhu and Kai Xu},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    pages = {to appear},
    month = {June},
    year = {2019}
}
```

```
@InProceedings{Mo_2019_CVPR,
    author = {Mo, Kaichun and Zhu, Shilin and Chang, Angel X. and Yi, Li and Tripathi, Subarna and Guibas, Leonidas J. and Su, Hao},
    title = {{PartNet}: A Large-Scale Benchmark for Fine-Grained and Hierarchical Part-Level {3D} Object Understanding},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2019}
}
```
