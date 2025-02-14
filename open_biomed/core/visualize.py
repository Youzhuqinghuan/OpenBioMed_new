from typing import Optional

from pymol import cmd
from datetime import datetime
import os
import shutil
import imageio
from rdkit.Chem import Draw, rdDepictor

from open_biomed.data import Molecule, Protein

def convert_png2gif(png_dir, gif_file, duration=0.1):
    """
    generate gif from png
    :param png_file
    :param gif_file
    """
    frames = [imageio.imread(os.path.join(png_dir, f)) for f in sorted(os.listdir(png_dir)) if f.endswith(".png")]
    imageio.mimsave(gif_file, frames, duration=duration)

def visualize_complex_3D(
    file: str, 
    protein_file: Optional[str]=None, 
    ligand_file: Optional[str]=None,
    rotate=False,
    width=800, 
    height=600, 
    dpi=300, 
    num_frames=50
):
    cmd.reinitialize()
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    
    if protein_file is not None:
        cmd.load(protein_file, "protein")
        cmd.hide("everything", "protein")
        cmd.show("cartoon", "protein")
        cmd.color("gray", "protein")
        cmd.set("transparency", 0.6, "protein")
        cmd.orient("protein")
    if ligand_file is not None:
        cmd.load(ligand_file, "ligand")
        cmd.show("sticks", "ligand")
        cmd.orient("ligand")
    cmd.zoom("all")
    
    if rotate:
        cmd.mset(f"1 x{num_frames}")
        cmd.util.mroll(1, num_frames, 360 // num_frames)

        name_dir = os.path.dirname(file)
        name_base = os.path.basename(file)
        name_time = datetime.now()
        name_temp = os.path.join(name_dir, "rotate_png_"+name_time.strftime("%Y%m%d_%H%M%S")+"_"+name_base[:-4])
        if not os.path.exists(name_temp):
            os.makedirs(name_temp)
        cmd.mpng(f"{name_temp}/", width=width, height=height)

        convert_png2gif(name_temp, file)
        if os.path.exists(name_temp):
            shutil.rmtree(name_temp)
    else:
        cmd.png(file, width=width, height=height, dpi=dpi)

class MoleculeVisualizer():
    def __init__(self) -> None:
        pass

    def run(self, molecule: Molecule, mode: str="2D", rotate: bool=False) -> str:
        timestamp = int(datetime.now().timestamp() * 1000)
        img_file_type = "gif" if rotate else "png"
        img_file = f"./tmp/mol_{timestamp}.{img_file_type}"
        if mode == "3D":
            sdf_file = f"./tmp/mol_{timestamp}.sdf"
            molecule.save_sdf(sdf_file)
            visualize_complex_3D(img_file, ligand_file=sdf_file, rotate=rotate)
        if mode == "2D":
            molecule._add_rdmol()
            rdDepictor.Compute2DCoords(molecule.rdmol)
            Draw.MolToImageFile(molecule.rdmol, img_file, size=(800, 600))

        return os.path.abspath(img_file)

class ProteinVisualizer():
    def __init__(self) -> None:
        pass

    def run(self, protein: Protein, rotate: bool=False) -> str:
        timestamp = int(datetime.now().timestamp() * 1000)
        pdb_file = f"./tmp/protein_{timestamp}.pdb"
        img_file_type = "gif" if rotate else "png"
        img_file = f"./tmp/protein_{timestamp}.{img_file_type}"
        protein.save_pdb(pdb_file)
        visualize_complex_3D(img_file, protein_file=pdb_file, rotate=rotate)

        return os.path.abspath(img_file)

class ComplexVisualizer():
    def __init__(self) -> None:
        pass

    def run(self, molecule: Molecule, protein: Protein, rotate: bool=True) -> str:
        timestamp = int(datetime.now().timestamp() * 1000)
        sdf_file = f"./tmp/mol_{timestamp}.sdf"
        pdb_file = f"./tmp/protein_{timestamp}.pdb"
        img_file_type = "gif" if rotate else "png"
        img_file = f"./tmp/complex_{timestamp}.{img_file_type}"
        molecule.save_sdf(sdf_file)
        protein.save_pdb(pdb_file)
        visualize_complex_3D(img_file, ligand_file=sdf_file, protein_file=pdb_file, rotate=rotate)

        return os.path.abspath(img_file)
"""
if __name__ == "__main__":
    # conda env: pymol

    # 小分子可视化
    SDF_FILE = "./files/8C7Y_TXV_ligand.sdf"
    PNG_FILE = "ligand.png"
    visualize_ligand(SDF_FILE, PNG_FILE)

    # 蛋白质可视化
    PDB_FILE = "./files/8C7Y_TXV_protein.pdb"
    PNG_FILE = "protein.png"
    visualize_protein(PDB_FILE, PNG_FILE)

    # 复合物动态可视化
    PROT_FILE = "./files/8C7Y_TXV_protein.pdb"
    LIG_FILE = "./files/8C7Y_TXV_ligand.sdf"
    GIF_FILE = "complex.gif"
    visualize_complex(PROT_FILE, LIG_FILE, GIF_FILE)
"""