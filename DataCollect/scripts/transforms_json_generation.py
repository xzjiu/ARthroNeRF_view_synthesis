from camera_stream_ndi_tracking import main
from extrinsic_mapping import csv_add
import argparse

parser = argparse.ArgumentParser(description='Collect images and tracking information')
parser.add_argument('--row_path', type=str, default="C:/Users/camp/Documents/Xinrui Zou/tracking_array_v3.rom",
                    help='tracking pattern description')
parser.add_argument('--img_path', type=str, default='./images_carbox/',
                    help='output image path')
parser.add_argument('--camera_file', type=str, default="transforms.json",
                    help='output path for json file')
parser.add_argument('--transforms_matrix', type=str, default="transforms_matrix.csv",
                    help='camera to tracking center')
args = parser.parse_args()

streams = main(args.row_path, args.img_path)
csv_add(streams, args.transforms_matrix, args.camera_file)
