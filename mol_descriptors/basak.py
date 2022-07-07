from rdkit import Chem
import numpy
import copy
import pandas as pd


class BasakDescriptors:

    def __init__(self):

        self._basak = {'CIC0': self.CalculateBasakCIC0,
                        'CIC1': self.CalculateBasakCIC1,
                        'CIC2': self.CalculateBasakCIC2,
                        'CIC3': self.CalculateBasakCIC3,
                        'CIC4': self.CalculateBasakCIC4,
                        'CIC5': self.CalculateBasakCIC5,
                        'CIC6': self.CalculateBasakCIC6,
                        'SIC0': self.CalculateBasakSIC0,
                        'SIC1': self.CalculateBasakSIC1,
                        'SIC2': self.CalculateBasakSIC2,
                        'SIC3': self.CalculateBasakSIC3,
                        'SIC4': self.CalculateBasakSIC4,
                        'SIC5': self.CalculateBasakSIC5,
                        'SIC6': self.CalculateBasakSIC6,
                        'IC0': self.CalculateBasakIC0,
                        'IC1': self.CalculateBasakIC1,
                        'IC2': self.CalculateBasakIC2,
                        'IC3': self.CalculateBasakIC3,
                        'IC4': self.CalculateBasakIC4,
                        'IC5': self.CalculateBasakIC5,
                        'IC6': self.CalculateBasakIC6
        }



    def _CalculateEntropy(self, Probability):
        """
        **Internal used only**
        Calculation of entropy (Information content) for probability given
        """
        res = 0.0
        for i in Probability:
            if i != 0:
                res = res - i * numpy.log2(i)

        return res


    def CalculateBasakIC0(self, mol):
        """
        Obtain the information content with order 0 proposed by Basak
        """

        BasakIC = 0.0
        Hmol = Chem.AddHs(mol)
        nAtoms = Hmol.GetNumAtoms()
        IC = []
        for i in range(nAtoms):
            at = Hmol.GetAtomWithIdx(i)
            IC.append(at.GetAtomicNum())
        Unique = numpy.unique(IC)
        NAtomType = len(Unique)
        NTAtomType = numpy.zeros(NAtomType, numpy.float)
        for i in range(NAtomType):
            NTAtomType[i] = IC.count(Unique[i])

        if nAtoms != 0:
            BasakIC = self._CalculateEntropy(NTAtomType / nAtoms)
        else:
            BasakIC = 0.0

        return BasakIC


    def CalculateBasakSIC0(self, mol):
        """
        Obtain the structural information content with order 0
        proposed by Basak
        """

        Hmol = Chem.AddHs(mol)
        nAtoms = Hmol.GetNumAtoms()
        IC = self.CalculateBasakIC0(mol)
        if nAtoms <= 1:
            BasakSIC = 0.0
        else:
            BasakSIC = IC / numpy.log2(nAtoms)

        return BasakSIC


    def CalculateBasakCIC0(self, mol):
        """
        Obtain the complementary information content with order 0
        proposed by Basak
        """
        Hmol = Chem.AddHs(mol)
        nAtoms = Hmol.GetNumAtoms()
        IC = self.CalculateBasakIC0(mol)
        if nAtoms <= 1:
            BasakCIC = 0.0
        else:
            BasakCIC = numpy.log2(nAtoms) - IC

        return BasakCIC


    def _CalculateBasakICn(self, mol, NumPath=1):
        """
        **internal used only**
        Obtain the information content with order n proposed by Basak
        """
        Hmol = Chem.AddHs(mol)
        nAtoms = Hmol.GetNumAtoms()
        TotalPath = Chem.FindAllPathsOfLengthN(Hmol, NumPath, useBonds=0, useHs=1)
        if len(TotalPath) == 0:
            BasakIC = 0.0
        else:
            IC = {}
            for i in range(nAtoms):
                temp = []
                at = Hmol.GetAtomWithIdx(i)
                temp.append(at.GetAtomicNum())
                for index in TotalPath:
                    if i == index[0]:
                        temp.append([Hmol.GetAtomWithIdx(kk).GetAtomicNum() for kk in index[1:]])
                    if i == index[-1]:
                        cds = list(index)
                        cds.reverse()
                        temp.append([Hmol.GetAtomWithIdx(kk).GetAtomicNum() for kk in cds[1:]])

                IC[str(i)] = temp
            cds = []
            for value in IC.values():
                for i in range(len(value)):
                    if isinstance(value[i],list):
                        value[i] = value[i][0]
                cds.append(sorted(list(value)))
            kkk = list(range(len(cds)))
            aaa = copy.deepcopy(kkk)
            res = []
            for i in aaa:
                if i in kkk:
                    jishu = 0
                    kong = []
                    temp1 = cds[i]
                    for j in aaa:
                        if cds[j] == temp1:
                            jishu = jishu + 1
                            kong.append(j)
                    for ks in kong:
                        kkk.remove(ks)
                    res.append(jishu)

            BasakIC = self._CalculateEntropy(numpy.array(res, numpy.float) / sum(res))

        return BasakIC


    def CalculateBasakIC1(self, mol):
        """
        Obtain the information content with order 1 proposed by Basak
        """
        return self._CalculateBasakICn(mol, NumPath=2)


    def CalculateBasakIC2(self, mol):
        """
        Obtain the information content with order 2 proposed by Basak
        """
        return self._CalculateBasakICn(mol, NumPath=3)


    def CalculateBasakIC3(self, mol):
        """
        Obtain the information content with order 3 proposed by Basak
        """
        return self._CalculateBasakICn(mol, NumPath=4)


    def CalculateBasakIC4(self, mol):
        """
        Obtain the information content with order 4 proposed by Basak
        """
        return self._CalculateBasakICn(mol, NumPath=5)


    def CalculateBasakIC5(self, mol):
        """
        Obtain the information content with order 5 proposed by Basak
        """
        return self._CalculateBasakICn(mol, NumPath=6)


    def CalculateBasakIC6(self, mol):
        """
        Obtain the information content with order 6 proposed by Basak
        """
        return self._CalculateBasakICn(mol, NumPath=7)


    def CalculateBasakSIC1(self, mol):
        """
        Obtain the structural information content with order 1
        proposed by Basak.
        """
        Hmol = Chem.AddHs(mol)
        nAtoms = Hmol.GetNumAtoms()
        IC = self.CalculateBasakIC1(mol)
        if nAtoms <= 1:
            BasakSIC = 0.0
        else:
            BasakSIC = IC / numpy.log2(nAtoms)

        return BasakSIC


    def CalculateBasakSIC2(self, mol):
        """
        Obtain the structural information content with order 2 proposed
        by Basak.
        """
        Hmol = Chem.AddHs(mol)
        nAtoms = Hmol.GetNumAtoms()
        IC = self.CalculateBasakIC2(mol)
        if nAtoms <= 1:
            BasakSIC = 0.0
        else:
            BasakSIC = IC / numpy.log2(nAtoms)

        return BasakSIC


    def CalculateBasakSIC3(self, mol):
        """
        Obtain the structural information content with order 3 proposed
        by Basak.
        """
        Hmol = Chem.AddHs(mol)
        nAtoms = Hmol.GetNumAtoms()
        IC = self.CalculateBasakIC3(mol)
        if nAtoms <= 1:
            BasakSIC = 0.0
        else:
            BasakSIC = IC / numpy.log2(nAtoms)

        return BasakSIC


    def CalculateBasakSIC4(self, mol):
        """
        Obtain the structural information content with order 4 proposed
        by Basak.
        """
        Hmol = Chem.AddHs(mol)
        nAtoms = Hmol.GetNumAtoms()
        IC = self.CalculateBasakIC4(mol)
        if nAtoms <= 1:
            BasakSIC = 0.0
        else:
            BasakSIC = IC / numpy.log2(nAtoms)

        return BasakSIC


    def CalculateBasakSIC5(self, mol):
        """
        Obtain the structural information content with order 5 proposed
        by Basak.
        """
        Hmol = Chem.AddHs(mol)
        nAtoms = Hmol.GetNumAtoms()
        IC = self.CalculateBasakIC5(mol)
        if nAtoms <= 1:
            BasakSIC = 0.0
        else:
            BasakSIC = IC / numpy.log2(nAtoms)

        return BasakSIC


    def CalculateBasakSIC6(self, mol):
        """
        Obtain the structural information content with order 6 proposed
        by Basak.
        """
        Hmol = Chem.AddHs(mol)
        nAtoms = Hmol.GetNumAtoms()
        IC = self.CalculateBasakIC6(mol)
        if nAtoms <= 1:
            BasakSIC = 0.0
        else:
            BasakSIC = IC / numpy.log2(nAtoms)

        return BasakSIC


    def CalculateBasakCIC1(self, mol):
        """
        Obtain the complementary information content with order 1 proposed
        by Basak.
        """
        Hmol = Chem.AddHs(mol)
        nAtoms = Hmol.GetNumAtoms()
        IC = self.CalculateBasakIC1(mol)
        if nAtoms <= 1:
            BasakCIC = 0.0
        else:
            BasakCIC = numpy.log2(nAtoms) - IC
        return BasakCIC


    def CalculateBasakCIC2(self, mol):
        """
        Obtain the complementary information content with order 2 proposed
        by Basak.
        """
        Hmol = Chem.AddHs(mol)
        nAtoms = Hmol.GetNumAtoms()
        IC = self.CalculateBasakIC2(mol)
        if nAtoms <= 1:
            BasakCIC = 0.0
        else:
            BasakCIC = numpy.log2(nAtoms) - IC

        return BasakCIC


    def CalculateBasakCIC3(self, mol):
        """
        Obtain the complementary information content with order 3 proposed
        by Basak.
        """
        Hmol = Chem.AddHs(mol)
        nAtoms = Hmol.GetNumAtoms()
        IC = self.CalculateBasakIC3(mol)
        if nAtoms <= 1:
            BasakCIC = 0.0
        else:
            BasakCIC = numpy.log2(nAtoms) - IC

        return BasakCIC


    def CalculateBasakCIC4(self, mol):
        """
        Obtain the complementary information content with order 4 proposed
        by Basak.
        """
        Hmol = Chem.AddHs(mol)
        nAtoms = Hmol.GetNumAtoms()
        IC = self.CalculateBasakIC4(mol)
        if nAtoms <= 1:
            BasakCIC = 0.0
        else:
            BasakCIC = numpy.log2(nAtoms) - IC

        return BasakCIC


    def CalculateBasakCIC5(self, mol):
        """
        Obtain the complementary information content with order 5 proposed
        by Basak.
        """
        Hmol = Chem.AddHs(mol)
        nAtoms = Hmol.GetNumAtoms()
        IC = self.CalculateBasakIC5(mol)
        if nAtoms <= 1:
            BasakCIC = 0.0
        else:
            BasakCIC = numpy.log2(nAtoms) - IC

        return BasakCIC


    def CalculateBasakCIC6(self, mol):
        """
        Obtain the complementary information content with order 6 proposed
        by Basak.
        """
        Hmol = Chem.AddHs(mol)
        nAtoms = Hmol.GetNumAtoms()
        IC = self.CalculateBasakIC6(mol)
        if nAtoms <= 1:
            BasakCIC = 0.0
        else:
            BasakCIC = numpy.log2(nAtoms) - IC

        return BasakCIC


    def GetBasakofMol(self, mol):
        """
        Get the dictionary of basak descriptors for given moelcule mol
        """
        result = {}
        for DesLabel in self._basak.keys():
            result[DesLabel] = round(self._basak[DesLabel](mol), 3)
        return result


    def getBasak(self, df_x):
        """
        Calculates all Basak descriptors for the dataset
            Parameters:
                df_x: pandas.DataFrame
                    SMILES DataFrame
            Returns:
                basak_descriptors: pandas.DataFrame
                    Basak Descriptors DataFrame
        """
        r = {}
        for key in self._basak:
            r[key] = []
        for m in df_x['Smiles']:
            mol = Chem.MolFromSmiles(m)
            res = self.GetBasakofMol(mol)
            for key in self._basak:
                r[key].append(res[key])
        basak_descriptors = pd.DataFrame(r).round(3)
        return pd.DataFrame(basak_descriptors)


