import os
import argparse

from sqa_gen.data_extraction import extract_data_from_file
from sqa_gen.sqa_gen_engine import sqa_gen_engine

parser = argparse.ArgumentParser()
# path
parser.add_argument('--datapath', type=str, default='dataset/opportunity')
parser.add_argument('--question_file', type=str, default='question_family.json')
parser.add_argument('--save_folder', type=str, default='sqa_data')
parser.add_argument('--save_model_folder', type=str, default='trained_models/')

# SQA settings
parser.add_argument('--stride', type=int, default=400)
parser.add_argument('--window_size', type=int, default=1500)
parser.add_argument('--dim', type=int, default=77)
parser.add_argument('--win_len', type=int, default=1)
parser.add_argument('--num_class1', type=int, default=18)
parser.add_argument('--num_class2', type=int, default=5)
parser.add_argument('--feature', type=int, default=64)
parser.add_argument('--drop_rate', type=float, default=0.3)
parser.add_argument('--epochs', type=int, default=200)

args = parser.parse_args()

def gen_main():
    _, label_list, _ = extract_data_from_file('S1-ADL5.dat',  args.datapath, plot_option = False, show_other = False)

    file_list = [ i for i in os.listdir(args.datapath) if '.dat' in i]
    save_file = 's1234_' + str(args.window_size) + '_' + str(args.stride)

    generated_data = sqa_gen_engine(file_list,
                                    args.datapath,
                                    label_list,
                                    data_split = 'Test', # either train or test
                                    save_folder = args.save_folder,
                                    save_model_folder = args.save_model_folder,
                                    sqa_data_name = save_file,
                                    window_size = args.window_size, stride = args.stride,
                                    question_family_file = 'sqa_data_gen/question_family.json',
                                    show_other = False  # not include "other" in activity list scene_list
                                    )

if __name__ == '__main__':
    gen_main()