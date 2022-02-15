from rdkit import Chem
from rdkit.Chem import Lipinski as LPK
import pandas as pd



class ConstitutionDescriptors:


    def __init__(self):

        self._constitutional = {'Weight': self.CalculateMolWeight,
                                'AWeight': self.CalculateAverageMolWeight,
                                'nhyd': self.CalculateHydrogenNumber,
                                'nhal': self.CalculateHalogenNumber,
                                'nhet': self.CalculateHeteroNumber,
                                'nhev': self.CalculateHeavyAtomNumber,
                                'ncof': self.CalculateFlorineNumber,
                                'ncocl': self.CalculateChlorineNumber,
                                'ncobr': self.CalculateBromineNumber,
                                'ncoi': self.CalculateIodineNumber,
                                'ncarb': self.CalculateCarbonNumber,
                                'nphos': self.CalculatePhosphorNumber,
                                'nsulph': self.CalculateOxygenNumber,
                                'noxy': self.CalculateOxygenNumber,
                                'nnitro': self.CalculateNitrogenNumber,
                                'nring': self.CalculateRingNumber,
                                'nrot': self.CalculateRotationBondNumber,
                                'ndonr': self.CalculateHdonorNumber,
                                'naccr': self.CalculateHacceptorNumber,
                                'nsb': self.CalculateSingleBondNumber,
                                'ndb': self.CalculateDoubleBondNumber,
                                'naro': self.CalculateAromaticBondNumber,
                                'ntb': self.CalculateTripleBondNumber,
                                'nta': self.CalculateAllAtomNumber,
                                'PC1': self.CalculatePath1,
                                'PC2': self.CalculatePath2,
                                'PC3': self.CalculatePath3,
                                'PC4': self.CalculatePath4,
                                'PC5': self.CalculatePath5,
                                'PC6': self.CalculatePath6
        }



    def CalculateMolWeight(self, mol):
        """
        Calculation of molecular weight. Note that not including H
            Parameters:
                mol: rdkit molecule
            Returns:
                MolWeight: Molecular weight
        """
        MolWeight = 0
        for atom in mol.GetAtoms():
            MolWeight = MolWeight + atom.GetMass()

        return MolWeight


    def CalculateAverageMolWeight(self, mol):
        """
        Calculation of average molecular weight. Note that not including H
            Parameters:
                mol: rdkit molecule
            Returns:
                AvgMolWeight: Average Molecular weight
        """
        MolWeight = 0
        for atom in mol.GetAtoms():
            MolWeight = MolWeight + atom.GetMass()
        return MolWeight / mol.GetNumAtoms()


    def CalculateHydrogenNumber(self, mol):
        """
        Calculation of Number of Hydrogen in a molecule
            Parameters:
                mol: rdkit molecule
            Returns:
                HydrogenNumber
        """
        i = 0
        Hmol = Chem.AddHs(mol)
        for atom in Hmol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                i = i + 1
        return i


    def CalculateHalogenNumber(self, mol):
        """
        Calculation of Halogen counts in a molecule
            Parameters:
                mol: rdkit molecule
            Returns:
                HalogenNumber
        """
        i = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 9 or atom.GetAtomicNum() == 17 or atom.GetAtomicNum() == 35 or atom.GetAtomicNum() == 53:
                i = i + 1
        return i


    def CalculateHeteroNumber(self, mol):
        """
        Calculation of Hetero counts in a molecule
            Parameters:
                mol: rdkit molecule
            Returns:
                HeteroNumber
        """
        i = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 6 or atom.GetAtomicNum() == 1:
                i = i + 1
        return mol.GetNumAtoms() - i


    def CalculateHeavyAtomNumber(self, mol):
        """
        Calculation of Heavy atom counts in a molecule
            Parameters:
                mol: rdkit molecule
            Returns:
                Heavy Atom Number
        """
        return mol.GetNumHeavyAtoms()


    def _CalculateElementNumber(self, mol, AtomicNumber=6):
        """
        **Internal used only**
        Calculation of element counts with atomic number equal to n in a molecule
        """
        i = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == AtomicNumber:
                i = i + 1
        return i


    def CalculateFlorineNumber(self, mol):
        """
        Calculation of Florine count in a molecule
            Parameters:
                mol: rdkit molecule
            Returns:
                Florine Number
        """
        return self._CalculateElementNumber(mol, AtomicNumber=9)


    def CalculateChlorineNumber(self, mol):
        """
        Calculation of Chlorine count in a molecule
            Parameters:
                mol: rdkit molecule
            Returns:
                Chlorine Number
        """

        return self._CalculateElementNumber(mol, AtomicNumber=17)


    def CalculateBromineNumber(self, mol):
        """
        Calculation of Bromine counts in a molecule
            Parameters:
                mol: rdkit molecule
            Returns:
                Bromine Number
        """
        return self._CalculateElementNumber(mol, AtomicNumber=35)


    def CalculateIodineNumber(self, mol):
        """
        Calculation of Iodine counts in a molecule
            Parameters:
                mol: rdkit molecule
            Returns:
                Iodine Number
        """
        return self._CalculateElementNumber(mol, AtomicNumber=53)


    def CalculateCarbonNumber(self, mol):
        """
        Calculation of Carbon number in a molecule
            Parameters:
                mol: rdkit molecule
            Returns:
                Carbon Number
        """
        return self._CalculateElementNumber(mol, AtomicNumber=6)


    def CalculatePhosphorNumber(self, mol):
        """
        Calculation of Phosphorus number in a molecule
            Parameters:
                mol: rdkit molecule
            Returns:
                Heavy Atom Number
        """
        return self._CalculateElementNumber(mol, AtomicNumber=15)


    def CalculateSulfurNumber(self, mol):
        """
        Calculation of Sulfur count in a molecule
            Parameters:
                mol: rdkit molecule
            Returns:
                Sulfur Number
        """
        return self._CalculateElementNumber(mol, AtomicNumber=16)


    def CalculateOxygenNumber(self, mol):
        """
        Calculation of Oxygen count in a molecule
            Parameters:
                mol: rdkit molecule
            Returns:
                Oxygen Number
        """
        return self._CalculateElementNumber(mol, AtomicNumber=8)


    def CalculateNitrogenNumber(self, mol):
        """
        Calculation of Nitrogen count in a molecule
            Parameters:
                mol: rdkit molecule
            Returns:
                Nitrogen Number
        """
        return self._CalculateElementNumber(mol, AtomicNumber=7)


    def CalculateRingNumber(self, mol):
        """
        Calculation of ring counts in a molecule
            Parameters:
                mol: rdkit molecule
            Returns:
                Ring Number
        """
        return Chem.GetSSSR(mol)


    def CalculateRotationBondNumber(self, mol):
        """
        Calculation of rotation bonds count in a molecule
            Parameters:
                mol: rdkit molecule
            Returns:
                Rotation Bond Number
        """
        return LPK.NumRotatableBonds(mol)


    def CalculateHdonorNumber(self, mol):
        """
        Calculation of Hydrongen bond donor count in a molecule
            Parameters:
                mol: rdkit molecule
            Returns:
                Hdonor Number
        """
        return LPK.NumHDonors(mol)


    def CalculateHacceptorNumber(self, mol):
        """
        Calculation of Hydrogen bond acceptor count in a molecule
            Parameters:
                mol: rdkit molecule
            Returns:
                Hacceptor Number
        """
        return LPK.NumHAcceptors(mol)


    def CalculateSingleBondNumber(self, mol):
        """
        Calculation of single bond counts in a molecule
            Parameters:
                mol: rdkit molecule
            Returns:
                Single Bond Number
        """
        i = 0;
        for bond in mol.GetBonds():
            if bond.GetBondType().name == 'SINGLE':
                i = i + 1
        return i


    def CalculateDoubleBondNumber(self, mol):
        """
        Calculation of double bond counts in a molecule
            Parameters:
                mol: rdkit molecule
            Returns:
                Double Bond Number
        """
        i = 0;
        for bond in mol.GetBonds():
            if bond.GetBondType().name == 'DOUBLE':
                i = i + 1
        return i


    def CalculateTripleBondNumber(self, mol):
        """
        Calculation of triple bond counts in a molecule
            Parameters:
                mol: rdkit molecule
            Returns:
                Triple Bond Number
        """
        i = 0;
        for bond in mol.GetBonds():
            if bond.GetBondType().name == 'TRIPLE':
                i = i + 1
        return i


    def CalculateAromaticBondNumber(self, mol):
        """
        Calculation of aromatic bond counts in a molecule
            Parameters:
                mol: rdkit molecule
            Returns:
                Aromatic Bond Number
        """
        i = 0;
        for bond in mol.GetBonds():
            if bond.GetBondType().name == 'AROMATIC':
                i = i + 1
        return i


    def CalculateAllAtomNumber(self, mol):
        """
        Calculation of all atom counts in a molecule
            Parameters:
                mol: rdkit molecule
            Returns:
                All Atom Count
        """
        return Chem.AddHs(mol).GetNumAtoms()


    def _CalculatePathN(self, mol, PathLength=2):
        """
        *Internal Use Only*
        Calculation of the counts of path length N for a molecule
        """
        return len(Chem.FindAllPathsOfLengthN(mol, PathLength, useBonds=1))


    def CalculatePath1(self, mol):
        """
        Calculation of the counts of path length 1 for a molecule
        """
        return self._CalculatePathN(mol, 1)


    def CalculatePath2(self, mol):
        """
        Calculation of the counts of path length 2 for a molecule
        """
        return self._CalculatePathN(mol, 2)


    def CalculatePath3(self, mol):
        """
        Calculation of the counts of path length 3 for a molecule
        """
        return self._CalculatePathN(mol, 3)


    def CalculatePath4(self, mol):
        """
        Calculation of the counts of path length 4 for a molecule
        """
        return self._CalculatePathN(mol, 4)


    def CalculatePath5(self, mol):
        """
        Calculation of the counts of path length 5 for a molecule
        """
        return self._CalculatePathN(mol, 5)


    def CalculatePath6(self, mol):
        """
        Calculation of the counts of path length 6 for a molecule
        """
        return self._CalculatePathN(mol, 6)


    def GetConstitutionalofMol(self, mol):
        """
        Get the dictionary of constitutional descriptors for given molecule mol
            Parameters:
                mol: rdkit molecule
            Returns:
                constitution descriptors: dict
        """
        result = {}
        for DesLabel in self._constitutional.keys():
            result[DesLabel] = round(self._constitutional[DesLabel](mol), 3)
        return result

    def getConstitutional(self, df_x):
        """
        Calculates all constitutional descriptors for the dataset
            Parameters:
                df_x: pandas.DataFrame
                    SMILES DataFrame
            Returns:
                constitutional_descriptors: pandas.DataFrame
                    Constitutional Descriptors DataFrame
        """

        r = {}
        for key in self._constitutional.keys():
            r[key] = []
        for m in df_x['Smiles']:
            mol = Chem.MolFromSmiles(m)
            res = self.GetConstitutionalofMol(mol)
            for key in self._constitutional.keys():
                r[key].append(res[key])
        constitutional_descriptors = pd.DataFrame(r).round(3)
        return pd.DataFrame(constitutional_descriptors)

