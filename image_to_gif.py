import glob
import moviepy.editor as mpy
import argparse

parser = argparse.ArgumentParser('Image to GIF')
parser.add_argument('--dynamics', type=str,
                    choices=['heat', 'gene', 'mutualistic',], default='heat')
parser.add_argument('--network', type=str,
                    choices=['grid', 'random', 'power_law', 'small_world', 'community'], default='grid')
parser.add_argument('--model', type=str,
                    choices=['tru', 'differential_gcn', 'no_embedding', 'no_control', 'no_graph'], default='differential_gcn')
args = parser.parse_args()
dynamics = args.dynamics
network = args.network
model = args.model

gif_name = dynamics + "_" + network + "_" + model
fps = 3
all_file_list = glob.glob(r'C:\Users\zangc\Desktop\fig\{}\{}\*.png'.format(dynamics, network)) # Get all the pngs in the current directory
file_list = [x for x in all_file_list if model in x]
file_list.sort( key=lambda x: int(x.split('\\')[-1].split('-')[0])) # Sort the images by #, this may need to be tweaked for your use case
clip = mpy.ImageSequenceClip(file_list, fps=fps)
clip.write_gif('{}.gif'.format(gif_name), fps=fps)