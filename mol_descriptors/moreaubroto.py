
from rdkit import Chem
from mol_descriptors.AtomProperty import AtomPropertyDescriptors
import pandas as pd
import numpy


class MoreaubrotoDescriptors:

    def _CalculateMoreauBrotoAutocorrelation(self, mol, lag=1, propertylabel='m'):
        """
        **Internal used only**
        Calculation of Moreau-Broto autocorrelation descriptors based on
        different property weights.
        """

        Natom = mol.GetNumAtoms()

        GetDistanceMatrix = Chem.GetDistanceMatrix(mol)
        res = 0.0
        for i in range(Natom):
            for j in range(Natom):
                if GetDistanceMatrix[i, j] == lag:
                    atom1 = mol.GetAtomWithIdx(i)
                    atom2 = mol.GetAtomWithIdx(j)
                    temp1 = AtomPropertyDescriptors().GetRelativeAtomicProperty(element=atom1.GetSymbol(), propertyname=propertylabel)
                    temp2 = AtomPropertyDescriptors().GetRelativeAtomicProperty(element=atom2.GetSymbol(), propertyname=propertylabel)
                    res = res + temp1 * temp2
                else:
                    res = res + 0.0

        return round(numpy.log(res / 2 + 1), 3)


    def CalculateMoreauBrotoAutoMass(self, mol):
        """
        Calculation of Moreau-Broto autocorrelation descriptors based on
        carbon-scaled atomic mass.
        """
        res = {}

        for i in range(8):
            res['ATSm' + str(i + 1)] = self._CalculateMoreauBrotoAutocorrelation(mol, lag=i + 1, propertylabel='m')

        return res


    def CalculateMoreauBrotoAutoVolume(self, mol):
        """
        Calculation of Moreau-Broto autocorrelation descriptors based on
        carbon-scaled atomic van der Waals volume.
        """
        res = {}

        for i in range(8):
            res['ATSv' + str(i + 1)] = self._CalculateMoreauBrotoAutocorrelation(mol, lag=i + 1, propertylabel='V')

        return res


    def CalculateMoreauBrotoAutoElectronegativity(self, mol):
        """
        Calculation of Moreau-Broto autocorrelation descriptors based on
        carbon-scaled atomic Sanderson electronegativity.
        """
        res = {}

        for i in range(8):
            res['ATSe' + str(i + 1)] = self._CalculateMoreauBrotoAutocorrelation(mol, lag=i + 1, propertylabel='En')

        return res


    def CalculateMoreauBrotoAutoPolarizability(self, mol):
        """
        Calculation of Moreau-Broto autocorrelation descriptors based on
        carbon-scaled atomic polarizability.
        """
        res = {}

        for i in range(8):
            res['ATSp' + str(i + 1)] = self._CalculateMoreauBrotoAutocorrelation(mol, lag=i + 1, propertylabel='alapha')

        return res


    def GetMoreauBrotoAutoofMol(self, mol):
        """
        Calcualate all Moreau-Broto autocorrelation descriptors.
        """
        res = {}
        res.update(self.CalculateMoreauBrotoAutoMass(mol))
        res.update(self.CalculateMoreauBrotoAutoVolume(mol))
        res.update(self.CalculateMoreauBrotoAutoElectronegativity(mol))
        res.update(self.CalculateMoreauBrotoAutoPolarizability(mol))

        return res


    def getMoreauBrotoAuto(self, df_x):
        """
        Calculates all MoreauBroto Auto-correlation descriptors for the dataset
            Parameters:
                df_x: pandas.DataFrame
                    SMILES DataFrame
            Returns:
                mb_descriptors: pandas.DataFrame
                    MoreauBroto Auto-correlation Descriptors DataFrame
        """
        r = {}
        labels = []
        for i in range(8):
            labels.append('ATSm' + str(i + 1))
            labels.append('ATSv' + str(i + 1))
            labels.append('ATSe' + str(i + 1))
            labels.append('ATSp' + str(i + 1))
        for key in labels:
            r[key] = []
        i = 0
        for m in df_x['Smiles']:
            i = i+1
            mol = Chem.MolFromSmiles(m)
            res = self.GetMoreauBrotoAutoofMol(mol)
            for key in labels:
                r[key].append(res[key])
        mb_descriptors = pd.DataFrame(r).round(3)
        return pd.DataFrame(mb_descriptors)


