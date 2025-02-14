#!bin/bash
export CUDA_VISIBLE_DEVICES=$1

python open_biomed/test.py --task text_based_molecule_editing \
    --additional_config_file configs/model/moleculestm.yaml --dataset_name fs_mol_edit --dataset_path ./datasets/text_based_molecule_editing/fs_mol_edit