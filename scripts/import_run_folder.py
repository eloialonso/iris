#! /usr/bin/python3

import argparse
from pathlib import Path
import shutil
import subprocess
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('src_folder', type=str, help='Folder to import. Must be formatted as: USERNAME@HOSTNAME:PATH_TO_OUTPUTS_FOLDER/DATE/TIME')
    parser.add_argument('-k', '--from-key', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    if args.from_key:
        key_file = Path('path_runs.yaml')
        assert key_file.is_file()
        with key_file.open('r') as f:
            runs = yaml.load(f, Loader=yaml.loader.SafeLoader)
        src_folder = runs[args.src_folder]
    else:
        src_folder = args.src_folder

    day, time = src_folder.split('/')[-2:]
    host = src_folder.split('@')[1].split(':')[0]
    dst_folder = Path(host) / day / time
    print(dst_folder)
    if dst_folder.is_dir():
        if input(f'{dst_folder} exists, remove it ? [Y/n] ').lower() != 'n':
            shutil.rmtree(dst_folder)
        else:
            print('Bye.')
            return

    dst_folder.mkdir(exist_ok=False, parents=True)

    # Make symbolic link from key to folder
    if args.from_key:
        subprocess.run(f'ln -s {str(dst_folder)} {args.src_folder}', shell=True, check=True)

    download_media = args.verbose and input('Download media/ ? [y/N] ').lower() == 'y'
    download_last = not args.verbose or input('Download checkpoints/last.pt ? [Y/n] ').lower() != 'n'

    folders = ['src', 'config', 'scripts']
    if download_media:
        folders.append('media')

    for folder in folders:
        subprocess.run(f'scp -r {src_folder}/{folder} {dst_folder}', shell=True, check=True)

    if download_last:
        checkpoint_folder = dst_folder / 'checkpoints'
        checkpoint_folder.mkdir(exist_ok=False, parents=False)
        subprocess.run(f'scp {src_folder}/checkpoints/last.pt {checkpoint_folder}', shell=True, check=True)


if __name__ == '__main__':
    main()
