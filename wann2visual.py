# %% [markdown]
# ### This code is made easily to reveal hopping parameters from wannier90 output.
# ### Author: DH Kiem (kiem.dohoon@gmail.com)
# ### Data: 2025-Apr-3
# ### Copyright: MIT
# ### Use: python THISCODE

# %%
# package import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bisect
from matplotlib.backends.backend_pdf import PdfPages



# %%
# User input block
# Criterion: length
# User inputs: lattice_vector, atomic position, orbital names for each atom.

########### USER INPUT START ############
hamfile = "wannier90_hr.dat"
skiplines = 28

# printing results up to this length.
length_criterion = 10.0 

# Range of visualization 
xlim = [-10,10]
ylim = [-10,10]
zlim = [0,20]

# You can rotate the lattice vector to see the lattice structure at the PDF.
lattice_vec = np.matrix([
 [  2.4890159,     0.0000000 ,    0.0000000], 
 [ -1.2445079,     2.1555510 ,    0.0000000],
 [  0.0000000,     0.0000000 ,   20.0000000] 
])                   

# atomic positions are given in the fractional coordinates.
projected_atompos_frac = np.matrix([
  [0.0, 0.0, 0.5],
  [-0.3333, 0.3333, 0.5],
])

# The number of groups for orbital info should be same with projected_atompos_cartesian.  
# e.g. ["dxz", "dyz", "dxy"],
orbital_info = [
  ["pz"], 
  ["pz"],
]

########### USER INPUT END ############


# %%
NumAtom =len(projected_atompos_frac)

assert NumAtom == len(orbital_info), "The atomic position is different to orbital infomration."
total_orb_num = 0
atom_pos_cart = projected_atompos_frac * lattice_vec

print("Lattice vector: \n",lattice_vec)

for (n,orb_per_atom) in enumerate(orbital_info):
  print("atom",n+1, ": ", orb_per_atom)
  total_orb_num += len(orb_per_atom)

print("Total number of orbitals: ", total_orb_num)

print("Atomic position (in Cartesian): ")
print(atom_pos_cart)
print(f"Length criterion: {length_criterion} (Å)")


# %%


# %%
# wannier90_har.dat to csv file
hamdata = np.loadtxt(hamfile,skiprows=skiplines)
df = pd.DataFrame(hamdata, columns=['Rx', 'Ry', 'Rz', 'index1', 'index2', 'ReH', 'ImH'])
df['Rx'] = df['Rx'].astype(int)
df['Ry'] = df['Ry'].astype(int)
df['Rz'] = df['Rz'].astype(int)
df['index1'] = df['index1'].astype(int)
df['index2'] = df['index2'].astype(int)
#df.to_csv(hamfile+".csv", index=False)  # Wannier90 format out


# %%
# assign_orb: As the orbital index is given, return orbital name

class OrbitalIndexer:
    def __init__(self, orbital_info):
        self.orbital_info = orbital_info
        self.group_start_indices = self._compute_start_indices(orbital_info)

    def _compute_start_indices(self, orbital_info):
        start_indices = [1]
        for group in orbital_info:
            start_indices.append(start_indices[-1] + len(group))
        return start_indices

    def get_atom_num(self, index):
        if index < 1 or index > self.group_start_indices[-1] + len(self.orbital_info[-1]) - 1:
            raise ValueError("Index out of range")
        
        return bisect.bisect_right(self.group_start_indices, index) # binary search for finding group index
    
    def get_orb_name(self, index):
        group = self.get_atom_num(index)
        idx_in_group = index-self.group_start_indices[group-1]
        return self.orbital_info[group-1][idx_in_group]
    
indexer = OrbitalIndexer(orbital_info)




# %%
df_position = pd.DataFrame()

for ix in range(df['Rx'].min(), df['Rx'].max()+1):
    for iy in range(df['Ry'].min(), df['Ry'].max()+1):
        for iz in range(df['Rz'].min(), df['Rz'].max()+1):
            df_temp = pd.DataFrame(atom_pos_cart)
            df_temp.columns = ['x','y','z']
            df_temp['Rx'] = ix
            df_temp['Ry'] = iy
            df_temp['Rz'] = iz
            df_temp[['x','y','z']] += df_temp['Rx'].to_numpy().reshape(-1,1) * lattice_vec[0,:] 
            df_temp[['x','y','z']] += df_temp['Ry'].to_numpy().reshape(-1,1) * lattice_vec[1,:] 
            df_temp[['x','y','z']] += df_temp['Rz'].to_numpy().reshape(-1,1) * lattice_vec[2,:] 
            df_position = pd.concat([df_position, df_temp],axis=0)

#print(df_position)            

# %%
df['Atom1'] = df['index1'].apply(lambda x: indexer.get_atom_num(x))
df['Atom2'] = df['index2'].apply(lambda x: indexer.get_atom_num(x))
df['Orb1']  = df['index1'].apply(lambda x: indexer.get_orb_name(x))
df['Orb2']  = df['index2'].apply(lambda x: indexer.get_orb_name(x))



# %%
cart_distance = atom_pos_cart[df['Atom1'].to_numpy()-1,:] - atom_pos_cart[df['Atom2'].to_numpy()-1,:] 
cart_distance += df['Rx'].to_numpy().reshape(-1,1) * lattice_vec[0,:] 
cart_distance += df['Ry'].to_numpy().reshape(-1,1) * lattice_vec[1,:] 
cart_distance += df['Rz'].to_numpy().reshape(-1,1) * lattice_vec[2,:] 

#print(cart_distance)
df = pd.concat([df, pd.DataFrame(cart_distance, columns=['rx','ry','rz'])], axis=1)
df['Distance'] = np.linalg.norm(df[['rx','ry','rz']].values, axis=1)
df = df[df['Distance'] < length_criterion]
df = df.sort_values(by=['Distance','Atom1','Atom2'])

# %%
df.insert(0, 'Distance', df.pop('Distance'))
df.insert(4, 'Atom1', df.pop('Atom1'))
df.insert(5, 'Atom2', df.pop('Atom2'))
df.insert(6, 'Orb1', df.pop('Orb1'))
df.insert(7, 'Orb2', df.pop('Orb2'))

df.pop('index1')
df.pop('index2')
df['ReH'] = df['ReH']*1e3
df['ImH'] = df['ImH']*1e3


# %%
df.to_csv(hamfile+".csv", index=False)  # Wannier90 format out


# %%

grouped = df.groupby(['Distance', 'Rx', 'Ry', 'Rz', 'Atom1', 'Atom2', 'rx','ry','rz'])

#for key, group in grouped:
#    print(f"\n--- Group: Distance={key[0]}, (Rx,Ry,Rz)=({key[1]},{key[2]},{key[3]}), (Atom1,Atom2)=({key[4]},{key[5]}), ---")
#    pivot = group.pivot(index='Orb1', columns='Orb2', values='ReH')
#    print(pivot)



# %%
pdf_filename = "orbital_matrices.pdf"
with PdfPages(pdf_filename) as pdf:
    for key, group in grouped:
        pivot = group.pivot_table(index='Orb1', columns='Orb2', values='ReH')

        fig, ax = plt.subplots(2,2,figsize=(8,4.5))

        # Title (right, upper)
        ax[0, 1].axis("off")
        title_str = (
            f"Distance = {round(key[0],3)} Å\n"
            f"Cell vector (Rx, Ry, Rz) = ({key[1]}, {key[2]}, {key[3]})\n"
            f"Cartesian vector (rx, ry, rz) = ({round(key[6],3)}, {round(key[7],3)}, {round(key[8],3)})\n"
            f"Atomic index (A1, A2) = ({key[4]}, {key[5]})\n"
            f"from atom {key[5]} to atom {key[4]}\n"
            f"Hamiltonian Unit: meV\n"
        )
        ax[0, 1].text(0.5, 0.5, title_str,
                       ha='center', 
                       va='center',
                       fontsize=10, 
                       #weight='bold',
                       )

        #Table (right,lower)
        ax[1,1].axis('tight')
        ax[1,1].axis('off')
        table = ax[1,1].table(
            cellText=pivot.round(3).values,
            rowLabels=pivot.index,
            colLabels=pivot.columns,
            cellLoc='center',
            loc='center'
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(0.9, 1.3)

        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(0.3)
            if row == 0 or col == -1: 
                cell.set_fontsize(9)
                cell.set_text_props()
            cell.set_edgecolor('gray')


        # top view with arrow
        topview = ax[0,0]
        scattertop = topview.scatter(df_position['x'],df_position['y'], s=2 )
        topview.set_aspect('equal')
        topview.set_xlim(xlim)
        topview.set_ylim(ylim)
        topview.set_xlabel("X (Å)")
        topview.set_ylabel("Y (Å)")
        points_x = [atom_pos_cart[key[5]-1,0], atom_pos_cart[key[4]-1,0]+ key[1]*lattice_vec[0,0]+ key[2]*lattice_vec[1,0]+ key[3]*lattice_vec[2,0]]
        points_y = [atom_pos_cart[key[5]-1,1], atom_pos_cart[key[4]-1,1]+ key[1]*lattice_vec[0,1]+ key[2]*lattice_vec[1,1]+ key[3]*lattice_vec[2,1]]
        topview.scatter(points_x, points_y, color='black', s=7)
        topview.annotate(
            '',                          
            #xy=(atom_pos_cart[key[4]-1,0]+key[6], atom_pos_cart[key[4]-1,1]+key[7] ),                   
            xy=(atom_pos_cart[key[4]-1,0]+ key[1]*lattice_vec[0,0]+ key[2]*lattice_vec[1,0]+ key[3]*lattice_vec[2,0] ,
                atom_pos_cart[key[4]-1,1]+ key[1]*lattice_vec[0,1]+ key[2]*lattice_vec[1,1]+ key[3]*lattice_vec[2,1] ),#,           ),
            xytext=(atom_pos_cart[key[5]-1,0], atom_pos_cart[key[5]-1,1] ),               # 화살표의 시작 점 (기준)
            arrowprops=dict(
                arrowstyle='->',         
                color='red',             
                lw=2                     
            )   
        )

        # side view with arrow
        sideview = ax[1,0]
        scatterside = sideview.scatter(df_position['x'],df_position['z'], s=2 )
        sideview.set_aspect('equal')
        sideview.set_xlim(xlim)
        sideview.set_ylim(zlim)
        sideview.set_xlabel("X (Å)")
        sideview.set_ylabel("Z (Å)")
        points_z = [atom_pos_cart[key[5]-1,2], atom_pos_cart[key[4]-1,2]+ key[1]*lattice_vec[0,2]+ key[2]*lattice_vec[1,2]+ key[3]*lattice_vec[2,2]]
        sideview.scatter(points_x, points_z, color='black', s=7)           
        sideview.annotate(
            '',                          
            #xy=(atom_pos_cart[key[4]-1,0]+key[6], atom_pos_cart[key[4]-1,1]+key[7] ),                   
            xy=(atom_pos_cart[key[4]-1,0]+ key[1]*lattice_vec[0,0]+ key[2]*lattice_vec[1,0]+ key[3]*lattice_vec[2,0] ,
                atom_pos_cart[key[4]-1,2]+ key[1]*lattice_vec[0,2]+ key[2]*lattice_vec[1,2]+ key[3]*lattice_vec[2,2] ),#,           ),
            xytext=(atom_pos_cart[key[5]-1,0], atom_pos_cart[key[5]-1,2] ),               
            arrowprops=dict(
                arrowstyle='->',         
                color='red',             
                lw=2                     
            )   
        )

        # save pdf
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

print(f"The result has been saved as {pdf_filename}")
