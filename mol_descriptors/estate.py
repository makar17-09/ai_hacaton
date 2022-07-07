
from rdkit.Chem.EState import Fingerprinter  as ESFP
from rdkit import Chem
import pandas as pd
from mol_descriptors.AtomTypes import AtomTypesDescriptors
import numpy



class EstateDescriptors:

    def __init__(self):

        self.labels = []
        for i in range(1,79):
            self.labels.append('S' + str(i))
            self.labels.append('Smax' + str(i))
            self.labels.append('Smin' + str(i))
        self.labels.append('S79')

    def _CalculateEState(self, mol, skipH=1):
        """
        **Internal used only**
        Get the EState value of each atom in a molecule
        """
        mol = Chem.AddHs(mol)
        if skipH == 1:
            mol = Chem.RemoveHs(mol)
        tb1 = Chem.GetPeriodicTable()
        nAtoms = mol.GetNumAtoms()
        Is = numpy.zeros(nAtoms, numpy.float)
        for i in range(nAtoms):
            at = mol.GetAtomWithIdx(i)
            atNum = at.GetAtomicNum()
            d = at.GetDegree()
            if d > 0:
                h = at.GetTotalNumHs()
                dv = tb1.GetNOuterElecs(atNum) - h
                # dv=numpy.array(_AtomHKDeltas(at),'d')
                N = self._GetPrincipleQuantumNumber(atNum)
                Is[i] = (4.0 / (N * N) * dv + 1) / d
        dists = Chem.GetDistanceMatrix(mol, useBO=0, useAtomWts=0)
        dists += 1
        accum = numpy.zeros(nAtoms, numpy.float)
        for i in range(nAtoms):
            for j in range(i + 1, nAtoms):
                p = dists[i, j]
                if p < 1e6:
                    temp = (Is[i] - Is[j]) / (p * p)
                    accum[i] += temp
                    accum[j] -= temp
        res = accum + Is
        return res


    def _GetPrincipleQuantumNumber(self, atNum):
        """
        *Internal Use Only*
        Get the principle quantum number of atom with atomic
        number equal to atNum
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


    def CalculateEstateFingerprint(self, mol):
        """
        The Calculation of EState Fingerprints.
        It is the number of times each possible atom type is hit.
        """
        temp = ESFP.FingerprintMol(mol)
        res = {}
        for i, j in enumerate(temp[0]):
            res['Sfinger' + str(i + 1)] = j

        return res


    def CalculateEstateValue(self, mol):
        """
        The Calculate of EState Values.
        It is the sum of the Estate indices for atoms of each type.
        """
        temp = ESFP.FingerprintMol(mol)
        res = {}
        for i, j in enumerate(temp[1]):
            res['S' + str(i + 1)] = round(j, 3)

        return res


    def CalculateMaxAtomTypeEState(self, mol):
        """
        Calculation of maximum of E-State value of specified atom type
        """
        AT = AtomTypesDescriptors().GetAtomLabel(mol)
        Estate = self._CalculateEState(mol)
        res = []
        for i in AT:
            if i == []:
                res.append(0)
            else:
                res.append(max([Estate[k] for k in i]))
        ESresult = {}
        for n, es in enumerate(res):
            ESresult['Smax' + str(n)] = round(es, 3)

        return ESresult


    def CalculateMinAtomTypeEState(self, mol):
        """
        Calculation of minimum of E-State value of specified atom type
        """
        AT = AtomTypesDescriptors().GetAtomLabel(mol)
        Estate = self._CalculateEState(mol)
        res = []
        for i in AT:
            if i == []:
                res.append(0)
            else:
                res.append(min([Estate[k] for k in i]))
        ESresult = {}
        for n, es in enumerate(res):
            ESresult['Smin' + str(n)] = round(es, 3)

        return ESresult


    def _GetEstateforMol(self, mol):
        """
        Obtain all Estate descriptors except Estate fingerprints .
        """
        result = {}
        result.update(self.CalculateEstateValue(mol))
        result.update(self.CalculateMaxAtomTypeEState(mol))
        result.update(self.CalculateMinAtomTypeEState(mol))

        return result


    def getEstate(self, df_x):
        """
        Calculates all Estate descriptors for the dataset
            Parameters:
                df_x: pandas.DataFrame
                    SMILES DataFrame
            Returns:
                estate_descriptors: pandas.DataFrame
                    Estate Descriptors DataFrame
        """

        r = {}
        for key in self.labels:
            r[key] = []
        for m in df_x['Smiles']:
            mol = Chem.MolFromSmiles(m)
            res = self._GetEstateforMol(mol)
            for key in self.labels:
                r[key].append(res[key])
        estate_descriptors = pd.DataFrame(r).round(3)
        return pd.DataFrame(estate_descriptors)

