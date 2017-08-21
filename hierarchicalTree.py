from struct import unpack
import numpy as np

sizeof_int = 4
sizeof_uint64 = 8
sizeof_dimImage = 3
sizeof_unsigned_int = 4
sizeof_unsigned_short_int = 2

class SuperVoxel(object):
    """docstring for SuperVoxel"""
    def __init__(self, f):
        super(SuperVoxel, self).__init__()
        self.TM              = unpack('<i', f.read(sizeof_int))[0]
        tmp = f.read(sizeof_uint64)
        self.dataSizeInBytes = unpack('<q', tmp)[0]
        self.dataDims = []
        for dim in range(3):
            self.dataDims.append(unpack('<q', f.read(sizeof_uint64))[0])
        self.ll              = unpack('<i', f.read(sizeof_unsigned_int))[0]
        self.PixelIdxList = []
        for l in range(self.ll):
            self.PixelIdxList.append(unpack('<q', f.read(sizeof_uint64))[0])
    

class nodeHierarchicalSegmentation(object):
    """docstring for nodeHierarchicalSegmentation"""
    def __init__(self, f, basicRegionsVec):
        super(nodeHierarchicalSegmentation, self).__init__()
        self.thrTau = unpack('<H', f.read(sizeof_unsigned_short_int))[0]
        ii = unpack('<i', f.read(sizeof_int))[0]
        if ii == -1:
            self.svPtr = -1
        else:
            self.svPtr = basicRegionsVec[ii]
        
class hierarchicalTree(object):
    """docstring for hierarchicalTree"""
    def __init__(self, file):
        if type(file) == str:
            f = open(file)
        else:
            f = file

        self.numBasicRegions = unpack('<i', f.read(sizeof_int))[0]
        self.basicRegionsVec = []

        for i in range(self.numBasicRegions):
            self.basicRegionsVec.append(SuperVoxel(f))

        self.numNodes = unpack('<i', f.read(sizeof_int))[0]
        self.parentIdx = []
        self.nodes = []

        for i in range(self.numNodes):
            self.nodes.append(nodeHierarchicalSegmentation(f, self.basicRegionsVec))
            self.parentIdx.append(unpack('<i', f.read(sizeof_int))[0])

        for i in range(1, self.numNodes):
            self.nodes[i].parent = self.nodes[self.parentIdx[i]]
            if not hasattr(self.nodes[self.parentIdx[i]], 'left'):
                self.nodes[self.parentIdx[i]].left = self.nodes[i]
            elif not hasattr(self.nodes[self.parentIdx[i]], 'right'):
                self.nodes[self.parentIdx[i]].right = self.nodes[i]
            else:
                print 'oupsies ...'

        f.close()

        self.nodes_by_Tau = {}
        for i, n in enumerate(self.nodes):
            self.nodes_by_Tau.setdefault(n.thrTau, []).append(n)


def read_svb(file):
    if type(file) == str:
        f = open(file)
    else:
        f = file

    numSv = np.fromfile(f, dtype=np.int32, count=1)
    SuperVoxels = []

    for i in range(numSv):
        SuperVoxels.append(SuperVoxel(f).PixelIdxList)

    return SuperVoxels
