import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdFMCS
from rdkit.Chem import AllChem
from rdkit import DataStructs
from matplotlib.colors import LinearSegmentedColormap
import base64

# Function: Find and visualize similar substructures
def visualize_similar_substructures(target_mol, substructures, threshold=0.4):
    target_fp = AllChem.GetMorganFingerprint(target_mol, 2)
    max_similarity = 0
    most_similar_substructure = None
    most_similar_substructure_smiles = None

    # Find the most similar substructure
    for substructure_smiles in substructures:
        substructure_mol = Chem.MolFromSmiles(substructure_smiles)
        substructure_fp = AllChem.GetMorganFingerprint(substructure_mol, 2)
        similarity = DataStructs.TanimotoSimilarity(target_fp, substructure_fp)
        if similarity > max_similarity and similarity >= threshold:
            max_similarity = similarity
            most_similar_substructure = substructure_mol
            most_similar_substructure_smiles = substructure_smiles

    # If similar substructures are found, visualize them
    if most_similar_substructure:
        mcs = rdFMCS.FindMCS([target_mol, most_similar_substructure])
        common_substructure = Chem.MolFromSmarts(mcs.smartsString)
        match = target_mol.GetSubstructMatch(common_substructure)

        # Create a color map
        cmap = LinearSegmentedColormap.from_list('custom_red', [(1,1,1), (1,0,0)], N=256)
        colors = [cmap(int(255 * (max_similarity ** (1/2))))] * len(match)

        # Mapping molecule
        d = rdMolDraw2D.MolDraw2DCairo(300, 300)  # or MolDraw2DSVG for SVG output
        rdMolDraw2D.PrepareAndDrawMolecule(d, target_mol, highlightAtoms=match, highlightAtomColors={m: colors[i] for i, m in enumerate(match)})
        d.FinishDrawing()
        png = d.GetDrawingText()
        png_base64 = base64.b64encode(png).decode('utf-8')
        return png_base64, max_similarity, most_similar_substructure_smiles
    return None, None, None

# Introduce gardenia compounds to SMILES
gardenia_smiles_file = 'SMILES of hepatotoxic compounds.xlsx'
gardenia_df = pd.read_excel(gardenia_smiles_file)
gardenia_smiles = gardenia_df.iloc[:, 1].dropna().tolist()

# SMILES introducing toxic compounds
toxic_smiles_file = '临时_CAS数据.xlsx'
toxic_df = pd.read_excel(toxic_smiles_file)
toxic_smiles = toxic_df['SMILES'].dropna().tolist()

# Prepare a DataFrame to store the results
results_df = pd.DataFrame(columns=['Index', 'Gardenia_SMILES', 'Similarity', 'Toxic_Substructure_SMILES', 'Image_Base64'])

# For each gardenia compound, the substructure similar to the toxic compound was found and visualized
results_list = []  # Create an empty list to store the results

for i, gardenia_smile in enumerate(gardenia_smiles):
    gardenia_mol = Chem.MolFromSmiles(gardenia_smile)
    if gardenia_mol:
        image_base64, similarity, toxic_substructure_smiles = visualize_similar_substructures(gardenia_mol, toxic_smiles)
        if image_base64:
            print(f'Gardenia molecule {i+1} with similarity {similarity:.2f} visualized.')
            results_list.append({
                'Index': i+1,
                'Gardenia_SMILES': gardenia_smile,
                'Similarity': similarity,
                'Toxic_Substructure_SMILES': toxic_substructure_smiles,
                'Image_Base64': image_base64
            })
        else:
            print(f'Gardenia molecule {i+1} has no similar substructures above the threshold.')

# Convert the list of results to a DataFrame
results_df = pd.DataFrame(results_list)

# Save the results to an Excel file
results_df.to_excel('.xlsx', index=False)
