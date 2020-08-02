import meshio
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import spilu
from numpy.linalg import solve, norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Hex:

    W = [1.0, 1.0]
    gaussPoint = [-1/np.sqrt(3), 1/np.sqrt(3)]

    def __init__(self, elNodes, elType="hexahedron", order=1):
        # self.GNM = None
        # self.jacobian = None
        self.elNodes = elNodes
        self.integralSum = 0

    def N1(self, g, h, r): return (1-g)*(1-h)*(1-r)/8
    def N2(self, g, h, r): return (1+g)*(1-h)*(1-r)/8
    def N3(self, g, h, r): return (1+g)*(1+h)*(1-r)/8
    def N4(self, g, h, r): return (1-g)*(1+h)*(1-r)/8
    def N5(self, g, h, r): return (1-g)*(1-h)*(1+r)/8
    def N6(self, g, h, r): return (1+g)*(1-h)*(1+r)/8
    def N7(self, g, h, r): return (1+g)*(1+h)*(1+r)/8
    def N8(self, g, h, r): return (1-g)*(1+h)*(1+r)/8

    def GN1(self, g, h, r): return np.array(
        [(-(1-h)*(1-r), -(1-g)*(1-r), -(1-g)*(1-h))])/8

    def GN2(self, g, h, r): return np.array(
        [((1-h)*(1-r), -(1+g)*(1-r), -(1+g)*(1-h))])/8

    def GN3(self, g, h, r): return np.array(
        [((1+h)*(1-r), (1+g)*(1-r), -(1+g)*(1+h))])/8

    def GN4(self, g, h, r): return np.array(
        [(-(1+h)*(1-r), (1-g)*(1-r), -(1-g)*(1+h))])/8

    def GN5(self, g, h, r): return np.array(
        [(-(1-h)*(1+r), -(1-g)*(1+r), (1-g)*(1-h))])/8

    def GN6(self, g, h, r): return np.array(
        [((1-h)*(1+r), -(1+g)*(1+r), (1+g)*(1-h))])/8

    def GN7(self, g, h, r): return np.array(
        [((1+h)*(1+r), (1+g)*(1+r), (1+g)*(1+h))])/8

    def GN8(self, g, h, r): return np.array(
        [(-(1+h)*(1+r), (1-g)*(1+r), (1-g)*(1+h))])/8

    def getGNM(self, g, h, r):
        GNM = np.concatenate((self.GN1(g, h, r), self.GN2(g, h, r), self.GN3(g, h, r), self.GN4(g, h, r),
                                   self.GN5(g, h, r), self.GN6(g, h, r), self.GN7(g, h, r), self.GN8(g, h, r)), axis=0).transpose()
        return GNM

    def getJacobian(self, GNM):
        jacobian = GNM.dot(self.elNodes)
        return jacobian

    def getDetJ(self, jacobian):
        detJ = np.linalg.det(jacobian)
        return detJ

    def getStrainDispM(self, GNM, jacobian):
        GNxyz = np.linalg.inv(jacobian).dot(GNM)

        strainDispM = np.zeros((6, 24))
        for i in range(GNxyz.shape[1]):
            Nx, Ny, Nz = GNxyz[:, i]
            strainDispM[:, i*3:(i+1)*3] = [ [Nx, 0, 0],
                                            [0, Ny, 0],
                                            [0, 0, Nz],
                                            [Ny, Nx, 0],
                                            [0, Nz, Ny],
                                            [Nz, 0, Nx]]

        return strainDispM

    def integralElem(self, D):
        integralSum = np.zeros((24,24))
        for i, g in enumerate(Hex.gaussPoint):
            for j, h in enumerate(Hex.gaussPoint):
                for k, r in enumerate(Hex.gaussPoint):
                    GNM = self.getGNM(g, h, r)
                    jacobian = self.getJacobian(GNM)
                    detJ = self.getDetJ(jacobian)
                    strainDispM = self.getStrainDispM(GNM, jacobian)

                    integralSum += Hex.W[i]*Hex.W[j]*Hex.W[k]*detJ * \
                        strainDispM.transpose().dot(D).dot(strainDispM)

        self.integralSum = integralSum
        return self.integralSum


class Problem:
    def __init__(self, meshFilePath, modulus, poissonRatio, name=""):
        self.meshFilePath = meshFilePath
        self.modulus = modulus
        self.poissonRatio = poissonRatio

        self.unitConstitutiveM = None
        self.constitutiveM = None

        self.loadMesh()
        self.loadBoundary()
        # self.printProblem()

    def loadMesh(self):
        self.mesh = meshio.read(self.meshFilePath)
        self.bbox = np.array(
            (np.min(self.mesh.points, axis=0), np.max(self.mesh.points, axis=0)))

    def loadBoundary(self):
        self.forceVector = [0.0, -8000.0, 0.0]
        self.fixedNode = self.mesh.point_sets["fix"]
        self.forceNode = self.mesh.point_sets["pull"]

    def getUnitConstitutiveMatrix(self):
        v = self.poissonRatio
        C0 = np.array([(1-v, v, v, 0, 0, 0),
                       (v, 1-v, v, 0, 0, 0),
                       (v, v, 1-v, 0, 0, 0),
                       (0, 0, 0, (1-2*v)/2, 0, 0),
                       (0, 0, 0, 0, (1-2*v)/2, 0),
                       (0, 0, 0, 0, 0, (1-2*v)/2)], dtype=float)

        self.unitConstitutiveM = C0/((1+v)*(1-2*v))
        return self.unitConstitutiveM

    def getConstitutiveMatrix(self):
        if self.unitConstitutiveM is None:
            self.getUnitConstitutiveMatrix()
        self.constitutiveM = self.modulus*self.unitConstitutiveM
        return self.constitutiveM

    def printProblem(self):
        print(self.bbox)
        print(self.mesh.points[0])
        print(self.mesh.cells[0].data[0])
        # print(self.mesh.cells[0].index)
        # print(self.mesh.cells[0].count)
        print(self.mesh.cells[0].type)
        # print(dir(self.mesh))
        # print(self.mesh.point_sets["fix"])
        # print(self.mesh.point_sets["pull"])
        print(self.fixedNode)
        print(self.forceNode)


class FEAModel:
    def __init__(self, problem):
        self.problem = problem
        self.nodesM = problem.mesh.points
        self.elemM = p1.mesh.cells[0].data
        self.numN = len(self.nodesM)
        self.K = None # stiffness matrix
        self.F = None # force
        self.U = None # displacement
        self.KF = None # K for free nodes
        self.FF = None # # F for free nodes
        self.freeNodeIdList = None

        self.KeList = []
        self.LeList = []

    def elemCalculation(self):
        for nodeConnect in self.elemM:
            elNodes = self.nodesM[nodeConnect]
            elemObj = []
            if len(nodeConnect) == 8:
                elemObj = Hex(elNodes)

            # element stiffness matrix
            Ke = elemObj.integralElem(self.problem.getConstitutiveMatrix())
            # element garther matrix
            Le = lil_matrix((3*len(nodeConnect), self.numN*3))
            for i, nodeId in enumerate(nodeConnect):
                Le[i*3, nodeId*3] = 1
                Le[i*3+1, nodeId*3+1] = 1
                Le[i*3+2, nodeId*3+2] = 1
            self.KeList.append(Ke)
            self.LeList.append(Le)

    def assembleK(self):
        self.K = csr_matrix((self.numN*3, self.numN*3))
        for i, Ke in enumerate(self.KeList):
            Ke = csr_matrix(Ke)
            Le = csr_matrix(self.LeList[i])
            self.K += csr_matrix(Le.transpose()).dot(Ke).dot(Le)
            


    def getF(self):
        self.F = lil_matrix((self.numN*3,1))
        for nodeId in self.problem.forceNode:
            self.F[nodeId*3,0] = self.problem.forceVector[0]
            self.F[nodeId*3+1,0] = self.problem.forceVector[1]
            self.F[nodeId*3+2,0] = self.problem.forceVector[2]

        # print(self.F)

    def eliminateFixedNode(self):
        freeNodeIdList = list(set(range(self.numN))-set(self.problem.fixedNode))
        freeNodeIdList.sort()
        self.freeNodeIdList = freeNodeIdList
        freeDofs = []
        for nodeId in freeNodeIdList:
            freeDofs.append(nodeId*3)
            freeDofs.append(nodeId*3+1)
            freeDofs.append(nodeId*3+2)

        self.KF = lil_matrix(self.K)[freeDofs,:][:,freeDofs] # K for free nodes
        self.FF = self.F[freeDofs,:] # F for free nodes


    def solve(self):
        print(type(self.KF))
        print(type(self.KF))
        print(type(self.FF))

        self.KF = csr_matrix(self.KF)
        self.FF = csr_matrix(self.FF)

        # luKF = spilu(self.KF)
        # self.UF = luKF.solve(self.FF.toarray())

        self.UF = spsolve(self.KF, self.FF)
        # print(self.UF)
        print(np.max(self.UF))
        UF = self.UF
        UF = UF.reshape(self.numN-len(self.problem.fixedNode), 3)
        self.U = np.zeros((self.numN, 3))
        for i,nodeDisp in enumerate(UF):
            nodeId = self.freeNodeIdList[i]
            self.U[nodeId] = nodeDisp
        print(self.U.shape)
        UMagnitude = [np.linalg.norm(d) for d in self.U]
        print(np.argmax(UMagnitude))
        print(np.max(UMagnitude))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(self.nodesM[:,0], self.nodesM[:,1], self.nodesM[:,2], c=UMagnitude)
        axisEqual3D(ax)
        plt.show()

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


if __name__ == "__main__":

    meshFile = "beamHex1_2.inp"
    modulus = 113000.0  # MPa
    poissonRatio = 0.23

    p1 = Problem(meshFile, modulus, poissonRatio)

    mod = FEAModel(p1)

    mod.elemCalculation()
    mod.assembleK()
    mod.getF()
    mod.eliminateFixedNode()
    mod.solve()
