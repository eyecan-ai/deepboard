from pathlib import Path
import click
import numpy as np
import cv2
import torch
from orione.modules.keypoints.heatmaps import HeatmapsUtils


@click.command("Generate dataset files")
@click.option("--input_folder", required=True, help="input folder")
@click.option("--radius", required=True, type=int, help="radius of gaussians drawn on heatmaps")
@click.option("--output_folder", required=True, help="output folder")
def generate(input_folder, radius, output_folder):
    images = sorted(Path(input_folder).glob('*background.jpg'))
    vertices_files = sorted(Path(input_folder).glob('*pixels.txt'))
    ratio_files = sorted(Path(input_folder).glob('*ratio.txt'))
    output_folder = Path(output_folder)

    for img_f, v_f, r_f in zip(images, vertices_files, ratio_files):
        img = cv2.imread(str(img_f))
        h, w = img.shape[:2]
        vertices = np.loadtxt(v_f).astype(np.int32)
        ratio = np.loadtxt(r_f)

        heatmap = torch.zeros((h, w))
        for v in vertices:
            heatmap = HeatmapsUtils.draw_gaussian_with_radius(heatmap, v, radius)

        heatmap = heatmap.numpy()
        cv2.imwrite(str(output_folder / img_f.stem.replace('background', 'heatmap.exr')), heatmap)
        np.savetxt(output_folder / (r_f.stem + '.txt'), [ratio])

if __name__ == "__main__":
    generate()
