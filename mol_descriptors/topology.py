# Imports
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Chem import GraphDescriptors as GD
import pandas as pd
import numpy
import scipy


periodicTable = rdchem.GetPeriodicTable()


class TopologyDescriptors:

    def __init__(self):

        self._Topology = {'W': self.CalculateWeiner,
                        'AW': self.CalculateMeanWeiner,
                        'J': self.CalculateBalaban,
                        'Xu': self.CalculateXuIndex,
                        'GMTI': self.CalculateGutmanTopo,
                        'Pol': self.CalculatePolarityNumber,
                        'DZ': self.CalculatePoglianiIndex,
                        'Ipc': self.CalculateIpc,
                        'BertzCT': self.CalculateBertzCT,
                        'Thara': self.CalculateHarary,
                        'Tsch': self.CalculateSchiultz,
                        'ZM1': self.CalculateZagreb1,
                        'ZM2': self.CalculateZagreb2,
                        'MZM1': self.CalculateMZagreb1,
                        'MZM2': self.CalculateMZagreb2,
                        'Qindex': self.CalculateQuadratic,
                        'Platt': self.CalculatePlatt,
                        'diametert': self.CalculateDiameter,
                        'radiust': self.CalculateRadius,
                        'petitjeant': self.CalculatePetitjean,
                        'Sito': self.CalculateSimpleTopoIndex,
                        'Hato': self.CalculateHarmonicTopoIndex,
                        'Geto': self.CalculateGeometricTopoIndex,
                        'Arto': self.CalculateArithmeticTopoIndex,
                        'Tigdi': self.CalculateGraphDistance
        }




    def _GetPrincipalQuantumNumber(self, atNum):
        """
        *Internal Use Only*
        Get the principal quantum number of atom with atomic
        number equal to atNum 
            Parameters:
                atNum: int
            Returns:
                PrincipalQuantumNumber: int
        """
        if atNum <= 2:
            return 1
        elif atNum <= 10:
            return 2
        elif atNum <= 18:
            return 3
        elif atNum <= 36:
            return 4
        elif atNum <= 54:
            return 5
        elif atNum <= 86:
            return 6
        else:
            return 7


    def CalculateWeiner(self, mol):
        """
        Weiner index (W) is the entries of distance matrix D from
        H-depleted molecular graph.
        Usage: 
            Parameters:
                mol: RDKit molecule object
            Returns:
            W: Weiner index
        """
        return 1.0 / 2 * sum(sum(Chem.GetDistanceMatrix(mol)))


    def CalculateMeanWeiner(self, mol):
        """
        Average Weiner index (AW)
            Parameters:
                mol: RDKit molecule object
            Returns:
                AW: Average Weiner index
        """
        N = mol.GetNumAtoms()
        WeinerNumber = self.CalculateWeiner(mol)
        return 2.0 * WeinerNumber / (N * (N - 1))


    def CalculateBalaban(self, mol):
        """
        Calculation of Balaban's J index (J) for a molecule
            Parameters:
                mol: RDKit molecule object
            Returns:
                J: Balaban’s J index
        Usage: 
            result=CalculateBalaban(mol)
            Input: mol is a molecule object
            Output: result is a numeric value
        """
        return Chem.GraphDescriptors.BalabanJ(mol)


    def CalculateDiameter(self, mol):
        """
        Largest value in the distance matrix.
            Parameters:
                mol: RDKit molecule object
            Returns:
                diametert: Largest value in the distance matrix
        """
        Distance = Chem.GetDistanceMatrix(mol)
        return Distance.max()


    def CalculateRadius(self, mol):
        """
        Radius based on topology.
            Parameters:
                mol: RDKit molecule object
            Returns:
                radiust: Radius based on topology
        """
        Distance = Chem.GetDistanceMatrix(mol)
        temp = []
        for i in Distance:
            temp.append(max(i))
        return min(temp)


    def CalculatePetitjean(self, mol):
        """
        Petitjean based on topology (petitjeant)
        Usage:
            Parameters:
                mol: RDKit molecule object
            Returns:
                petitjeant: Petitjean based on topology
        """
        diameter = self.CalculateDiameter(mol)
        radius = self.CalculateRadius(mol)
        return 1 - radius / float(diameter)


    def CalculateXuIndex(self, mol):
        """
        Xu index: based on the adjacency matrix and distance matrix
            Parameters:
                mol: RDKit molecule object
            Returns:
                Xu: Xu Index
        """
        nAT = mol.GetNumAtoms()
        deltas = [x.GetDegree() for x in mol.GetAtoms()]
        Distance = Chem.GetDistanceMatrix(mol)
        sigma = scipy.sum(Distance, axis=1)
        temp1 = 0.0
        temp2 = 0.0
        for i in range(nAT):
            temp1 = temp1 + deltas[i] * ((sigma[i]) ** 2)
            temp2 = temp2 + deltas[i] * (sigma[i])
        Xu = numpy.sqrt(nAT) * numpy.log(temp1 / temp2)
        return Xu


    def CalculateGutmanTopo(self, mol):
        """
        Gutman molecular topological index (GMTI) based on
        simple vertex degree.
            Parameters:
                mol: RDKit molecule object
            Returns:
                GMTI: Gutman molecular topological index
        """
        nAT = mol.GetNumAtoms()
        deltas = [x.GetDegree() for x in mol.GetAtoms()]
        Distance = Chem.GetDistanceMatrix(mol)
        res = 0.0
        for i in range(nAT):
            for j in range(i + 1, nAT):
                res = res + deltas[i] * deltas[j] * Distance[i, j]
        return numpy.log10(res)


    def CalculatePolarityNumber(self, mol):
        """
        Polarity number (Pol): It is the number of pairs of vertexes
        at distance matrix equal to 3
            Parameters:
                mol: RDKit molecule object
            Returns:
                Pol: Gutman molecular topological index
        """
        Distance = Chem.GetDistanceMatrix(mol)
        res = 1. / 2 * sum(sum(Distance == 3))
        return res


    def CalculatePoglianiIndex(self, mol):
        """
        Calculation of Poglicani index
        The Pogliani index (Dz) is the sum over all non-hydrogen atoms
        of a modified vertex degree calculated as the ratio
        of the number of valence electrons over the principal
        quantum number of an atom
            Parameters:
                mol: RDKit molecule object
            Returns:
                Dz: Pogliani index
        """
        res = 0.0
        for atom in mol.GetAtoms():
            n = atom.GetAtomicNum()
            nV = periodicTable.GetNOuterElecs(n)
            mP = self._GetPrincipalQuantumNumber(n)
            res = res + (nV + 0.0) / mP
        return res


    def CalculateIpc(self, mol):
        """
        Ipc index is the information for polynomial coefficients
        based information theory.
            Parameters:
                mol: RDKit molecule object
            Returns:
                Ipc: Ipc index
        """
        temp = GD.Ipc(mol)
        if temp > 0:
            return numpy.log10(temp)
        else:
            return 0


    def CalculateBertzCT(self, mol):
        """
        BertzCT index meant to quantify "complexity" of molecules.
            Parameters:
                mol: RDKit molecule object
            Returns:
                BertzCT: BertzCT index
        """
        temp = GD.BertzCT(mol)
        if temp > 0:
            return numpy.log10(temp)
        else:
            return 0


    def CalculateHarary(self, mol):
        """
        The Harary index is a molecular topological index derived
        from the reciprocal distance matrix
            Parameters:
                mol: RDKit molecule object
            Returns:
                Thara: Thara number
        """
        Distance = numpy.array(Chem.GetDistanceMatrix(mol), 'd')
        return 1.0 / 2 * (sum(1.0 / Distance[Distance != 0]))


    def CalculateSchiultz(self, mol):
        """
        Calculation of Schiultz number
            Parameters:
                mol: RDKit molecule object
            Returns:
                Tsch: Thara number
        """
        Distance = numpy.array(Chem.GetDistanceMatrix(mol), 'd')
        Adjacent = numpy.array(Chem.GetAdjacencyMatrix(mol), 'd')
        VertexDegree = sum(Adjacent)
        return sum(scipy.dot((Distance + Adjacent), VertexDegree))


    def CalculateZagreb1(self, mol):
        """
        Calculation of Zagreb index with order 1 in a molecule
            Parameters:
                mol: RDKit molecule object
            Returns:
                ZM1: Zagreb index with order 1
        """
        deltas = [x.GetDegree() for x in mol.GetAtoms()]
        return sum(numpy.array(deltas) ** 2)


    def CalculateZagreb2(self, mol):
        """
        Calculation of Zagreb index with order 2 in a molecule
            Parameters:
                mol: RDKit molecule object
            Returns:
                ZM2: Zagreb index with order 2
        """
        ke = [x.GetBeginAtom().GetDegree() * x.GetEndAtom().GetDegree() for x in mol.GetBonds()]
        return sum(ke)


    def CalculateMZagreb1(self, mol):
        """
        Calculation of Modified Zagreb index with order 1 in a molecule
            Parameters:
                mol: RDKit molecule object
            Returns:
                ZM1: Modified Zagreb index with order 1
        """
        deltas = [x.GetDegree() for x in mol.GetAtoms()]
        while 0 in deltas:
            deltas.remove(0)
        deltas = numpy.array(deltas, 'd')
        res = sum((1. / deltas) ** 2)
        return res


    def CalculateMZagreb2(self, mol):
        """
        Calculation of Modified Zagreb index with order 2 in a molecule
            Parameters:
                mol: RDKit molecule object
            Returns:
                ZM2: Zagreb index with order 2
        """
        cc = [x.GetBeginAtom().GetDegree() * x.GetEndAtom().GetDegree() for x in mol.GetBonds()]
        while 0 in cc:
            cc.remove(0)
        cc = numpy.array(cc, 'd')
        res = sum((1. / cc) ** 2)
        return res


    def CalculateQuadratic(self, mol):
        """
        Calculation of Quadratic index in a molecule
            Parameters:
                mol: RDKit molecule object
            Returns:
                Qindex: Quadratic index
        """
        M = self.CalculateZagreb1(mol)
        N = mol.GetNumAtoms()
        return 3 - 2 * N + M / 2.0


    def CalculatePlatt(self, mol):
        """
        Calculation of Platt number in a molecule
            Parameters:
                mol: RDKit molecule object
            Returns:
                Platt: Platt number
        """
        cc = [x.GetBeginAtom().GetDegree() + x.GetEndAtom().GetDegree() - 2 for x in mol.GetBonds()]
        return sum(cc)


    def CalculateSimpleTopoIndex(self, mol):
        """
        Calculation of the logarithm of the simple topological index by Narumi,
        which is defined as the product of the vertex degree.
            Parameters:
                mol: RDKit molecule object
            Returns:
                Sito: log of simple topological index
        """
        deltas = [x.GetDegree() for x in mol.GetAtoms()]
        while 0 in deltas:
            deltas.remove(0)
        deltas = numpy.array(deltas, 'd')

        res = numpy.prod(deltas)
        if res > 0:
            return numpy.log10(res)
        else:
            return 0


    def CalculateHarmonicTopoIndex(self, mol):
        """
        Calculation of harmonic topological index proposed by Narnumi.
            Parameters:
                mol: RDKit molecule object
            Returns:
                Hamo: harmonic tological index
        """
        deltas = [x.GetDegree() for x in mol.GetAtoms()]
        while 0 in deltas:
            deltas.remove(0)
        deltas = numpy.array(deltas, 'd')
        nAtoms = mol.GetNumAtoms()
        if len(deltas) != 0 and all(deltas) != 0:
            res = nAtoms / sum(1. / deltas)
        else:
            return 0
        return res


    def CalculateGeometricTopoIndex(self, mol):
        """
        Geometric topological index by Narumi
            Parameters:
                mol: RDKit molecule object
            Returns:
                Geto: Geometric topological index
        """
        nAtoms = mol.GetNumAtoms()
        deltas = [x.GetDegree() for x in mol.GetAtoms()]
        while 0 in deltas:
            deltas.remove(0)
        deltas = numpy.array(deltas, 'd')
        temp = numpy.prod(deltas)
        res = numpy.power(temp, 1. / nAtoms)
        return res


    def CalculateGraphDistance(self, mol):
        """
        Calculation of graph distance index
            Parameters:
                mol: RDKit molecule object
            Returns:
                Tigdi: graph topological index

        """
        Distance = Chem.GetDistanceMatrix(mol)
        n = int(Distance.max())
        if n == 1e8:
            return 0

        else:
            res = 0.0
            for i in range(n):
                temp = 1. / 2 * sum(sum(Distance == i + 1))
                res = res + temp ** 2
            return numpy.log10(res)


    def CalculateArithmeticTopoIndex(self, mol):
        """
        Arithmetic topological index by Narumi
            Parameters:
                mol: RDKit molecule object
            Returns:
                Arto: Arithmatic topological index
        """
        nAtoms = mol.GetNumAtoms()
        nBonds = mol.GetNumBonds()
        res = 2. * nBonds / nAtoms
        return res


    def GetTopologyofMol(self, mol):
        """
        Get the dictionary of constitutional descriptors for
        a given molecule
            Parameters:
                mol: RDKit molecule object
            Returns:
                result: Dict with descriptor name as key and
                        value as the descriptor value
        """
        result = {}
        for DesLabel in self._Topology.keys():
            result[DesLabel] = self._Topology[DesLabel](mol)
        return result


    def getTopology(self, df_x):
        """
        Calculates all topology descriptors for the dataset
            Parameters:
                df_x: pandas.DataFrame
                    SMILES DataFrame
            Returns:
                topology_descriptors: pandas.DataFrame
                    Topology Descriptors DataFrame
        """
        r = {}
        for key in self._Topology.keys():
            r[key] = []
        for m in df_x['Smiles']:
            mol = Chem.MolFromSmiles(m)
            res = self.GetTopologyofMol(mol)
            for key in self._Topology.keys():
                r[key].append(res[key])
        topology_descriptors = pd.DataFrame(r).round(3)
        return pd.DataFrame(topology_descriptors)

