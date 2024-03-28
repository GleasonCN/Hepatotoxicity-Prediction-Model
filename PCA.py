import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.express as px

# Define a function to calculate the properties of a compound
def calculate_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    molwt = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    psa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    return [molwt, logp, psa, hbd, hba]

# Read the database of toxic compounds
toxic_df = pd.read_excel('（SMILES of hepatotoxicity）.xlsx', engine='openpyxl')
toxic_df['Properties'] = toxic_df['SMILES'].apply(calculate_properties)
toxic_df = toxic_df.dropna().reset_index(drop=True)

# The attributes were separated and PCA analysis was performed
toxic_properties = pd.DataFrame(toxic_df['Properties'].tolist(), columns=['MW', 'LogP', 'PSA', 'HBD', 'HBA'])
scaler = StandardScaler()
toxic_properties_scaled = scaler.fit_transform(toxic_properties)
pca = PCA(n_components=3)
toxic_pca = pca.fit_transform(toxic_properties_scaled)

# Read the gardenia chemical composition database
gardenia_df = pd.read_excel('SMILES of hepatotoxic compounds.xlsx', engine='openpyxl')
gardenia_df['Properties'] = gardenia_df['SMILES'].apply(calculate_properties)
gardenia_df = gardenia_df.dropna().reset_index(drop=True)

# The attributes were separated and PCA analysis was performed
gardenia_properties = pd.DataFrame(gardenia_df['Properties'].tolist(), columns=['MW', 'LogP', 'PSA', 'HBD', 'HBA'])
gardenia_properties_scaled = scaler.transform(gardenia_properties)
gardenia_pca = pca.transform(gardenia_properties_scaled)

# Calculate PCA center points for toxic compounds
toxic_center = np.mean(toxic_pca, axis=0)

# Calculate the distance between the chemical composition of gardenia gardenia and the central point of the toxic compound
gardenia_df['Distance_to_Toxic_Center'] = np.linalg.norm(gardenia_pca - toxic_center, axis=1)

# The chemical components of gardenia were sorted by distance
gardenia_df_sorted = gardenia_df.sort_values(by='Distance_to_Toxic_Center')

# Output to Excel
gardenia_df_sorted[['SMILES', 'Distance_to_Toxic_Center']].to_excel('sorted_gardenia_compounds.xlsx', index=False)

# Create a DataFrame that contains all the data points and labels
all_pca_data = np.vstack((toxic_pca, gardenia_pca))
labels = ['Toxic'] * len(toxic_pca) + ['Gardenia'] * len(gardenia_pca)

# Create a DataFrame for visualization
pca_df = pd.DataFrame(all_pca_data, columns=['PC1', 'PC2', 'PC3'])
pca_df['Label'] = labels

# Create interactive 3D scatter plots with plotly
fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='Label', symbol='Label',
                    size_max=10, opacity=0.7, title="PCA 3D Scatter Plot")

# The labeling style was updated to make the chemical composition of gardenia more obvious
fig.update_traces(marker=dict(size=12, line=dict(width=1)),
                  selector=dict(mode='markers'))

# Print
fig.show()
