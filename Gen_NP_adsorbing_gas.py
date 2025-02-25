
import os
import ase
from ase.cluster import wulff_construction
from ase.cluster import Octahedron
from ase.io import read, write
import numpy as np
from ase import Atoms
from itertools import combinations
from ase.geometry import get_duplicate_atoms
from ase.neighborlist import NeighborList, natural_cutoffs


def gen_bare_NP():
    octa = Octahedron(symbol='Au', length=5, cutoff=2, latticeconstant=4.1, alloy=False)
    # wulf = wulff_construction(symbol='Au',surfaces=[(1, 0, 0), (1, 1, 0), (1, 1, 1)],energies=[1.0, 0.7, 0.9],size=147,latticeconstant=4.1,structure='fcc')
    write('NP.xyz', octa)
    return octa

def get_surface_atoms_cn( cutoff_multiplier=None, cn_threshold=None):
    structure = gen_bare_NP()
    atoms = structure

    # Déterminer les cutoffs naturels pour les atomes et les multiplier par le facteur
    cutoffs = natural_cutoffs(atoms)
    cutoffs = [cutoff * cutoff_multiplier for cutoff in cutoffs]

    # Créer une liste des voisins en utilisant les cutoffs
    neighbor_list = NeighborList(cutoffs, self_interaction=False, bothways=True)
    neighbor_list.update(atoms)
    # Liste pour stocker les indices des atomes de surface
    surface_atom_indices = []
    core_atom_indices = []
    indice_111 , indice_100, indice_arret,indice_sommet= [],[],[],[]

    # Calculer le nombre de coordination (CN) pour chaque atome
    for i in range(len(atoms)):
        indice, offset = neighbor_list.get_neighbors(i)
        cn = len(indice)

        # Vérifier si le CN est inférieur au seuil défini pour les atomes de surface
        if cn < cn_threshold:
            surface_atom_indices.append(i)
        if cn == cn_threshold:
            core_atom_indices.append(i)

        if cn == 8:
            indice_100.append(i)
        if cn == 9:
            indice_111.append(i)
        if cn == 7:
            indice_arret.append(i)
        if cn == 6:
            indice_sommet.append(i)

    return atoms, surface_atom_indices, core_atom_indices, indice_111 , indice_100, indice_arret,indice_sommet

def adsorb_mols(top=None, hcp=None, fcc = None, brg= None, height = None):
    atoms, surface_indices, core_indices,indice_111 , indice_100, indice_arret,indice_sommet = get_surface_atoms_cn(cutoff_multiplier=1., cn_threshold=12)
    print("Indices des atomes de surface (", len(surface_indices), "en total) : ", surface_indices)

    center_of_mass = atoms.get_center_of_mass()
    core_positions = atoms[core_indices].get_positions()

    bond_lengths = atoms.get_all_distances()
    mask = bond_lengths != 0
    bond_length = np.min(bond_lengths[mask])
    bond_length = np.around(bond_length,decimals=2)
    print("M-M bond length = ", bond_length)
    latticeconstant = np.sqrt(2)*bond_length
    latticeconstant = np.around(latticeconstant,decimals=2)
    print("Lattice constant = ",latticeconstant)
    print('sqrt3',np.sqrt(3)*bond_length)


    hcp_sites , fcc_sites= [],[]
    for triplet in combinations(indice_111+indice_arret+indice_sommet,3):

        dis1 = round(atoms.get_distance(triplet[0], triplet[1]),2)
        dis2 = round(atoms.get_distance(triplet[0], triplet[2]),2)
        dis3 = round(atoms.get_distance(triplet[1], triplet[2]),2)

        pos1 = atoms.positions[triplet[0]]
        pos2 = atoms.positions[triplet[1]]
        pos3 = atoms.positions[triplet[2]]

        centroid = np.mean([pos1, pos2, pos3], axis=0)
        direction_vector = centroid - center_of_mass
        min_dis_to_sublayer = round((np.linalg.norm(centroid - core_positions, axis=1)).min(),2)

        vec1 = pos1 - pos2
        vec2 = pos2 - pos3
        normal = np.cross(vec1, vec2)
        normal /= np.linalg.norm(normal)
        if np.dot(normal, direction_vector) < 0:
            normal = -normal
        if dis1 == dis2 == dis3 == bond_length:
            if not (bond_length == min_dis_to_sublayer):
                hcp_site = centroid + normal * height
                hcp_sites.append(hcp_site)
            if (bond_length == min_dis_to_sublayer):
                fcc_site = centroid + normal * height
                fcc_sites.append(fcc_site)

    if hcp:
        for site in hcp_sites:
            atoms += (Atoms('H', positions=[site]))

    if fcc:
        for site in fcc_sites:
            atoms += (Atoms('H', positions=[site]))

    if brg:
        brg_sites = []
        for triplet in combinations(indice_100+indice_arret+indice_sommet,3):
            dis1 = round(atoms.get_distance(triplet[0], triplet[1]), 2)
            dis2 = round(atoms.get_distance(triplet[0], triplet[2]), 2)
            dis3 = round(atoms.get_distance(triplet[1], triplet[2]), 2)

            pos1 = atoms.positions[triplet[0]]
            pos2 = atoms.positions[triplet[1]]
            pos3 = atoms.positions[triplet[2]]

            brg1 = (pos1 + pos2) / 2
            brg2 = (pos1 + pos3) / 2

            direction_vector1 = brg1 - center_of_mass
            direction_vector2 = brg2 - center_of_mass

            vec1 = pos1 - pos2
            vec2 = pos2 - pos3
            normal = np.cross(vec1, vec2)
            normal /= np.linalg.norm(normal)

            if np.dot(normal, direction_vector1) < 0:
                normal = -normal

            if np.dot(normal, direction_vector2) < 0:
                normal = -normal

            if dis1 == dis2 == bond_length and dis3 == latticeconstant:
                brg_site1 = brg1 + normal * height
                brg_site2 = brg2 + normal * height
                brg_sites.append(brg_site1)
                brg_sites.append(brg_site2)

        for site in brg_sites:
            atoms += (Atoms('H', positions=[site]))

        get_duplicate_atoms(atoms, cutoff=0.1, delete=True)

    if top:
        top_sites = []
        for atom in (surface_indices):
            pos = atoms.positions[atom]
            direct =  center_of_mass - pos
            direct /= np.linalg.norm(direct)
            top_site = pos - direct * height * 1.5
            top_sites.append(top_site)

            for site in top_sites:
                atoms += (Atoms('H', positions=[site]))

    write('adsorbed_NP.xyz', atoms)
    return

# utilisateur peut modifier:
adsorb_mols(fcc=True,brg=True,height=1)
