from rdkit import Chem
from rdkit.Chem import rdchem
import numpy
import pandas as pd
periodicTable = rdchem.GetPeriodicTable()


class ConnectivityDescriptors:


    def __init__(self):

        self._connectivity = {'Chi0': self.CalculateChi0,
                            'Chi1': self.CalculateChi1,
                            'mChi1': self.CalculateMeanRandic,
                            'Chi2': self.CalculateChi2,
                            'Chi3': self.CalculateChi3p,
                            'Chi4': self.CalculateChi4p,
                            'Chi5': self.CalculateChi5p,
                            'Chi6': self.CalculateChi6p,
                            'Chi7': self.CalculateChi7p,
                            'Chi8': self.CalculateChi8p,
                            'Chi9': self.CalculateChi9p,
                            'Chi10': self.CalculateChi10p,
                            'Chi3c': self.CalculateChi3c,
                            'Chi4c': self.CalculateChi4c,
                            'Chi4pc': self.CalculateChi4pc,
                            'Chi3ch': self.CalculateChi3ch,
                            'Chi4ch': self.CalculateChi4ch,
                            'Chi5ch': self.CalculateChi5ch,
                            'Chi6ch': self.CalculateChi6ch,
                            'knotp': self.CalculateDeltaChi3c4pc,
                            'Chiv0': self.CalculateChiv0,
                            'Chiv1': self.CalculateChiv1,
                            'Chiv2': self.CalculateChiv2,
                            'Chiv3': self.CalculateChiv3p,
                            'Chiv4': self.CalculateChiv4p,
                            'Chiv5': self.CalculateChiv5p,
                            'Chiv6': self.CalculateChiv6p,
                            'Chiv7': self.CalculateChiv7p,
                            'Chiv8': self.CalculateChiv8p,
                            'Chiv9': self.CalculateChiv9p,
                            'Chiv10': self.CalculateChiv10p,
                            'dchi0': self.CalculateDeltaChi0,
                            'dchi1': self.CalculateDeltaChi1,
                            'dchi2': self.CalculateDeltaChi2,
                            'dchi3': self.CalculateDeltaChi3,
                            'dchi4': self.CalculateDeltaChi4,
                            'Chiv3c': self.CalculateChiv3c,
                            'Chiv4c': self.CalculateChiv4c,
                            'Chiv4pc': self.CalculateChiv4pc,
                            'Chiv3ch': self.CalculateChiv3ch,
                            'Chiv4ch': self.CalculateChiv4ch,
                            'Chiv5ch': self.CalculateChiv5ch,
                            'Chiv6ch': self.CalculateChiv6ch,
                            'knotpv': self.CalculateDeltaChiv3c4pc
                            }



    def CalculateChi0(self, mol):
        """
        Calculation of molecular connectivity chi index for path order 0
        """
        deltas = [x.GetDegree() for x in mol.GetAtoms()]
        while 0 in deltas:
            deltas.remove(0)
        deltas = numpy.array(deltas, 'd')
        res = sum(numpy.sqrt(1. / deltas))
        return res


    def CalculateChi1(self, mol):
        """
        Calculation of molecular connectivity chi index for path order 1
        (i.e.,Radich)
        """
        cc = [x.GetBeginAtom().GetDegree() * x.GetEndAtom().GetDegree() for x in mol.GetBonds()]
        while 0 in cc:
            cc.remove(0)
        cc = numpy.array(cc, 'd')
        res = sum(numpy.sqrt(1. / cc))
        return res


    def CalculateMeanRandic(self, mol):
        """
        Calculation of mean chi1 (Randic) connectivity index.
        """
        cc = [x.GetBeginAtom().GetDegree() * x.GetEndAtom().GetDegree() for x in mol.GetBonds()]
        while 0 in cc:
            cc.remove(0)
        cc = numpy.array(cc, 'd')
        res = numpy.mean(numpy.sqrt(1. / cc))

        return res


    def _CalculateChinp(self, mol, NumPath=2):
        """
        **Internal used only**
        Calculation of molecular connectivity chi index for path order 2
        """
        accum = 0.0
        deltas = [x.GetDegree() for x in mol.GetAtoms()]
        for path in Chem.FindAllPathsOfLengthN(mol, NumPath + 1, useBonds=0):
            cAccum = 1.0
            for idx in path:
                cAccum *= deltas[idx]
            if cAccum:
                accum += 1. / numpy.sqrt(cAccum)
        return accum


    def CalculateChi2(self, mol):
        """
        Calculation of molecular connectivity chi index for path order 2
        """
        return self._CalculateChinp(mol, NumPath=2)


    def CalculateChi3p(self, mol):
        """
        Calculation of molecular connectivity chi index for path order 3
        """
        return self._CalculateChinp(mol, NumPath=3)


    def CalculateChi4p(self, mol):
        """
        Calculation of molecular connectivity chi index for path order 4
        """
        return self._CalculateChinp(mol, NumPath=4)


    def CalculateChi5p(self, mol):
        """
        Calculation of molecular connectivity chi index for path order 5
        """
        return self._CalculateChinp(mol, NumPath=5)


    def CalculateChi6p(self, mol):
        """
        Calculation of molecular connectivity chi index for path order 6
        """
        return self._CalculateChinp(mol, NumPath=6)


    def CalculateChi7p(self, mol):
        """
        Calculation of molecular connectivity chi index for path order 7
        """
        return self._CalculateChinp(mol, NumPath=7)


    def CalculateChi8p(self, mol):
        """
        Calculation of molecular connectivity chi index for path order 8
        """
        return self._CalculateChinp(mol, NumPath=8)


    def CalculateChi9p(self, mol):
        """
        Calculation of molecular connectivity chi index for path order 9
        """
        return self._CalculateChinp(mol, NumPath=9)


    def CalculateChi10p(self, mol):
        """
        Calculation of molecular connectivity chi index for path order 10
        """
        return self._CalculateChinp(mol, NumPath=10)


    def CalculateChi3c(self, mol):
        """
        Calculation of molecular connectivity chi index for cluster
        """

        accum = 0.0
        deltas = [x.GetDegree() for x in mol.GetAtoms()]
        patt = Chem.MolFromSmarts('*~*(~*)~*')
        HPatt = mol.GetSubstructMatches(patt)
        for cluster in HPatt:
            deltas = [mol.GetAtomWithIdx(x).GetDegree() for x in cluster]
            while 0 in deltas:
                deltas.remove(0)
            if deltas != []:
                deltas1 = numpy.array(deltas, numpy.float)
                accum = accum + 1. / numpy.sqrt(deltas1.prod())
        return accum


    def CalculateChi4c(self, mol):
        """
        Calculation of molecular connectivity chi index for cluster
        """

        accum = 0.0
        deltas = [x.GetDegree() for x in mol.GetAtoms()]
        patt = Chem.MolFromSmarts('*~*(~*)(~*)~*')
        HPatt = mol.GetSubstructMatches(patt)
        for cluster in HPatt:
            deltas = [mol.GetAtomWithIdx(x).GetDegree() for x in cluster]
            while 0 in deltas:
                deltas.remove(0)
            if deltas != []:
                deltas1 = numpy.array(deltas, numpy.float)
                accum = accum + 1. / numpy.sqrt(deltas1.prod())
        return accum


    def CalculateChi4pc(self, mol):
        """
        Calculation of molecular connectivity chi index for path/cluster
        """

        accum = 0.0
        deltas = [x.GetDegree() for x in mol.GetAtoms()]
        patt = Chem.MolFromSmarts('*~*(~*)~*~*')
        HPatt = mol.GetSubstructMatches(patt)
        for cluster in HPatt:
            deltas = [mol.GetAtomWithIdx(x).GetDegree() for x in cluster]
            while 0 in deltas:
                deltas.remove(0)
            if deltas != []:
                deltas1 = numpy.array(deltas, numpy.float)
                accum = accum + 1. / numpy.sqrt(deltas1.prod())
        return accum


    def CalculateDeltaChi3c4pc(self, mol):
        """
        Calculation of the difference between chi3c and chi4pc
        """
        return abs(self.CalculateChi3c(mol) - self.CalculateChi4pc(mol))


    def _CalculateChinch(self, mol, NumCycle=3):
        """
        **Internal used only**
        Calculation of molecular connectivity chi index for cycles of n
        """
        accum = 0.0
        deltas = [x.GetDegree() for x in mol.GetAtoms()]
        for tup in mol.GetRingInfo().AtomRings():
            cAccum = 1.0
            if len(tup) == NumCycle:
                for idx in tup:
                    cAccum *= deltas[idx]
                if cAccum:
                    accum += 1. / numpy.sqrt(cAccum)

        return accum


    def CalculateChi3ch(self, mol):
        """
        Calculation of molecular connectivity chi index for cycles of 3
        """

        return self._CalculateChinch(mol, NumCycle=3)


    def CalculateChi4ch(self, mol):
        """
        Calculation of molecular connectivity chi index for cycles of 4
        """
        return self._CalculateChinch(mol, NumCycle=4)


    def CalculateChi5ch(self, mol):
        """
        Calculation of molecular connectivity chi index for cycles of 5
        """

        return self._CalculateChinch(mol, NumCycle=5)


    def CalculateChi6ch(self, mol):
        """
        Calculation of molecular connectivity chi index for cycles of 6
        """
        return self._CalculateChinch(mol, NumCycle=6)


    def _HKDeltas(self, mol, skipHs=1):
        """
        *Internal Use Only*
        Calculation of modified delta value for a molecule
        """
        global periodicTable
        res = []
        for atom in mol.GetAtoms():
            n = atom.GetAtomicNum()
            if n > 1:
                nV = periodicTable.GetNOuterElecs(n)
                nHs = atom.GetTotalNumHs()
                if n < 10:
                    res.append(float(nV - nHs))
                else:
                    res.append(float(nV - nHs) / float(n - nV - 1))
            elif not skipHs:
                res.append(0.0)
        return res


    def CalculateChiv0(self, mol):
        """
        Calculation of valence molecular connectivity chi index for
        path order 0
        """
        deltas = self._HKDeltas(mol, skipHs=0)
        while 0 in deltas:
            deltas.remove(0)
        deltas = numpy.array(deltas, 'd')
        res = sum(numpy.sqrt(1. / deltas))
        return res


    def _CalculateChivnp(self, mol, NumPath=1):
        """
        **Internal used only**
        Calculation of valence molecular connectivity chi index for path order 1
        """

        accum = 0.0
        deltas = self._HKDeltas(mol, skipHs=0)
        for path in Chem.FindAllPathsOfLengthN(mol, NumPath + 1, useBonds=0):
            cAccum = 1.0
            for idx in path:
                cAccum *= deltas[idx]
            if cAccum:
                accum += 1. / numpy.sqrt(cAccum)
        return accum


    def CalculateChiv1(self, mol):
        """
        Calculation of valence molecular connectivity chi index for
        path order 1
        """
        return self._CalculateChivnp(mol, NumPath=1)


    def CalculateChiv2(self, mol):
        """
        Calculation of valence molecular connectivity chi index for
        path order 2
        """
        return self._CalculateChivnp(mol, NumPath=2)


    def CalculateChiv3p(self, mol):
        """
        Calculation of valence molecular connectivity chi index for
        path order 3
        """

        return self._CalculateChivnp(mol, NumPath=3)


    def CalculateChiv4p(self, mol):
        """
        Calculation of valence molecular connectivity chi index for
        path order 4
        """

        return self._CalculateChivnp(mol, NumPath=4)


    def CalculateChiv5p(self, mol):
        """
        Calculation of valence molecular connectivity chi index for
        path order 5
        """

        return self._CalculateChivnp(mol, NumPath=5)


    def CalculateChiv6p(self, mol):
        """
        Calculation of valence molecular connectivity chi index for
        path order 6
        """

        return self._CalculateChivnp(mol, NumPath=6)


    def CalculateChiv7p(self, mol):
        """
        Calculation of valence molecular connectivity chi index for
        path order 7
        """

        return self._CalculateChivnp(mol, NumPath=7)


    def CalculateChiv8p(self, mol):
        """
        Calculation of valence molecular connectivity chi index for
        path order 8
        """

        return self._CalculateChivnp(mol, NumPath=8)


    def CalculateChiv9p(self, mol):
        """
        Calculation of valence molecular connectivity chi index for
        path order 9
        """

        return self._CalculateChivnp(mol, NumPath=9)


    def CalculateChiv10p(self, mol):
        """
        Calculation of valence molecular connectivity chi index for
        path order 10
        """

        return self._CalculateChivnp(mol, NumPath=10)


    def CalculateDeltaChi0(self, mol):
        """
        Calculation of the difference between chi0v and chi0
        """
        return abs(self.CalculateChiv0(mol) - self.CalculateChi0(mol))


    def CalculateDeltaChi1(self, mol):
        """
        Calculation of the difference between chi1v and chi1
        """
        return abs(self.CalculateChiv1(mol) - self.CalculateChi1(mol))


    def CalculateDeltaChi2(self, mol):
        """
        Calculation of the difference between chi2v and chi2
        """
        return abs(self._CalculateChivnp(mol, NumPath=2) - self._CalculateChinp(mol, NumPath=2))


    def CalculateDeltaChi3(self, mol):
        """
        Calculation of the difference between chi3v and chi3
        """
        return abs(self._CalculateChivnp(mol, NumPath=3) - self._CalculateChinp(mol, NumPath=3))


    def CalculateDeltaChi4(self, mol):
        """
        Calculation of the difference between chi4v and chi4
        """
        return abs(self._CalculateChivnp(mol, NumPath=4) - self._CalculateChinp(mol, NumPath=4))


    def _AtomHKDeltas(self, atom, skipHs=0):
        """
        *Internal Use Only*
        Calculation of modified delta value for a molecule
        """
        global periodicTable
        res = []
        n = atom.GetAtomicNum()
        if n > 1:
            nV = periodicTable.GetNOuterElecs(n)
            nHs = atom.GetTotalNumHs()
            if n < 10:
                res.append(float(nV - nHs))
            else:
                res.append(float(nV - nHs) / float(n - nV - 1))
        elif not skipHs:
            res.append(0.0)
        return res


    def CalculateChiv3c(self, mol):
        """
        Calculation of valence molecular connectivity chi index for cluster
        """
        accum = 0.0
        deltas = [x.GetDegree() for x in mol.GetAtoms()]
        patt = Chem.MolFromSmarts('*~*(~*)~*')
        HPatt = mol.GetSubstructMatches(patt)
        for cluster in HPatt:
            deltas = [self._AtomHKDeltas(mol.GetAtomWithIdx(x)) for x in cluster]
            while 0 in deltas:
                deltas.remove(0)
            if deltas != []:
                deltas1 = numpy.array(deltas, numpy.float)
                accum = accum + 1. / numpy.sqrt(deltas1.prod())
        return accum


    def CalculateChiv4c(self, mol):
        """
        Calculation of valence molecular connectivity chi index for cluster
        """
        accum = 0.0
        deltas = [x.GetDegree() for x in mol.GetAtoms()]
        patt = Chem.MolFromSmarts('*~*(~*)(~*)~*')
        HPatt = mol.GetSubstructMatches(patt)
        for cluster in HPatt:
            deltas = [self._AtomHKDeltas(mol.GetAtomWithIdx(x)) for x in cluster]
            while 0 in deltas:
                deltas.remove(0)
            if deltas != []:
                deltas1 = numpy.array(deltas, numpy.float)
                accum = accum + 1. / numpy.sqrt(deltas1.prod())
        return accum


    def CalculateChiv4pc(self, mol):
        """
        Calculation of valence molecular connectivity chi index for
        path/cluster
        """
        accum = 0.0
        deltas = [x.GetDegree() for x in mol.GetAtoms()]
        patt = Chem.MolFromSmarts('*~*(~*)~*~*')
        HPatt = mol.GetSubstructMatches(patt)
        for cluster in HPatt:
            deltas = [self._AtomHKDeltas(mol.GetAtomWithIdx(x)) for x in cluster]
            while 0 in deltas:
                deltas.remove(0)
            if deltas != []:
                deltas1 = numpy.array(deltas, numpy.float)
                accum = accum + 1. / numpy.sqrt(deltas1.prod())
        return accum


    def CalculateDeltaChiv3c4pc(self, mol):
        """
        Calculation of the difference between chiv3c and chiv4pc
        """
        return abs(self.CalculateChiv3c(mol) - self.CalculateChiv4pc(mol))


    def _CalculateChivnch(self, mol, NumCyc=3):
        """
        **Internal used only**
        Calculation of valence molecular connectivity chi index for cycles of n
        """
        accum = 0.0
        deltas = self._HKDeltas(mol, skipHs=0)
        for tup in mol.GetRingInfo().AtomRings():
            cAccum = 1.0
            if len(tup) == NumCyc:
                for idx in tup:
                    cAccum *= deltas[idx]
                if cAccum:
                    accum += 1. / numpy.sqrt(cAccum)

        return accum


    def CalculateChiv3ch(self, mol):
        """
        Calculation of valence molecular connectivity chi index
        for cycles of 3
        """
        return self._CalculateChivnch(mol, NumCyc=3)


    def CalculateChiv4ch(self, mol):
        """
        Calculation of valence molecular connectivity chi index for
        cycles of 4
        """
        return self._CalculateChivnch(mol, NumCyc=4)


    def CalculateChiv5ch(self, mol):
        """
        Calculation of valence molecular connectivity chi index for
        cycles of 5
        """
        return self._CalculateChivnch(mol, NumCyc=5)


    def CalculateChiv6ch(self, mol):
        """
        Calculation of valence molecular connectivity chi index for
        cycles of 6
        """
        return self._CalculateChivnch(mol, NumCyc=6)


    def GetConnectivityforMol(self, mol):
        """
        Get the dictionary of connectivity descriptors for given molecule mol
        """
        result = {}
        for DesLabel in self._connectivity.keys():
            result[DesLabel] = round(self._connectivity[DesLabel](mol), 3)
        return result

    def getConnectivity(self, df_x):
        """
        Calculates all Connectivity descriptors for the dataset
            Parameters:
                df_x: pandas.DataFrame
                    SMILES DataFrame
            Returns:
                connectivity_descriptors: pandas.DataFrame
                    Connectivity Descriptors DataFrame
        """
        r = {}
        for key in self._connectivity.keys():
            r[key] = []
        for m in df_x['Smiles']:
            mol = Chem.MolFromSmiles(m)
            res = self.GetConnectivityforMol(mol)
            for key in self._connectivity.keys():
                r[key].append(res[key])
        connectivity_descriptors = pd.DataFrame(r).round(3)
        return pd.DataFrame(connectivity_descriptors)


