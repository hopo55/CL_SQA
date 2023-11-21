import os
import numpy as np
from tqdm import tqdm
from datetime import date

import torch

from sqa_gen.data_extraction import extract_data_from_file, data_split_tools, series2graph
from sqa_gen.train_model import train_opp_model
from models.ConvLSTM import ConvLSTM
from sqa_gen.question_generation import question_generator

def sqa_gen_engine(args,
                   file_list,
                   datapath,
                   label_list,
                   data_split, # either train or test
                   save_folder,
                   save_model_folder,
                   sqa_data_name,
                   window_size = 1800, stride = 900,
                   question_family_file = 'question_family.json',
                   show_other = False
                   ):

    model_name = 'single_1'
    save_path1 = save_model_folder+ 'opp_model/'+ model_name+'.pt'
    model_name = 'single_2'
    save_path2 = save_model_folder + 'opp_model/'+ model_name+'.pt'

    if os.path.isfile(save_path1) and os.path.isfile(save_path1):
        print('Load models')
        trained_model_1 = ConvLSTM(dim=args.dim, win_len=args.win_len, num_classes_1=args.num_class1, num_feat_map=args.feature, dropout_rate=args.drop_rate)
        trained_model_1.load_state_dict(torch.load(save_path1))

        trained_model_2 = ConvLSTM(dim=args.dim, win_len=args.win_len, num_classes_1=args.num_class2, num_feat_map=args.feature, dropout_rate=args.drop_rate)
        trained_model_2.load_state_dict(torch.load(save_path2))
    else:
        # train model
        train_opp_model(args, datapath)
    
    today = date.today()
    date_str = today.strftime("%m/%d/%y") # mm/dd/y

    # structure of question family
    gen_sqa_data = {}
    gen_sqa_data['info'] = {}
    gen_sqa_data['questions'] = []

    gen_sqa_data['info']['date'] = date_str
    gen_sqa_data['info']['license'] = 'Creative Commons Attribution (CC BY 4.0)'
    gen_sqa_data['info']['split'] = data_split
    gen_sqa_data['info']['version'] = '1.0'

    context_counter = 0
    
    trained_model_1 = trained_model_1.to(args.device)
    trained_model_2 = trained_model_2.to(args.device)
    trained_model_1.eval()
    trained_model_2.eval()

    for source_file_i in file_list:
        label_y, _, data_x = extract_data_from_file(source_file_i, 
                                                    datapath = datapath,
                                                    plot_option = False, 
                                                    show_other = show_other,
                                                    )
        ## Whether using "other" as one activity? No ... (3 places)
        print('Extracting %s file....'%(source_file_i))

        # generate context and questions using sliding window
        for startpoint_i in tqdm(range(0, data_x.shape[0]-window_size, stride)):
            # the sampling rate for opportunity is 30HZ, window is 60s
            seg_x ,seg_y_list, startpoint = data_split_tools(data_x, 
                                                             label_y, 
                                                             window_size,
                                                             startpoint = startpoint_i)
            
            # label_list[0] : mid / label_list[1] : loc / label_list[2] : high
            # series2graph???
            scene_list_1 = series2graph(seg_y_list[0], label_list[0], show_graph = False, show_other = show_other)
            scene_list_2 = series2graph(seg_y_list[1], label_list[1], show_graph = False)
            scene_lists = [scene_list_1, scene_list_2]

            # =====  get predicted scene_list using pre-trained source classifier.=====
            # need to reshape the input X (n, h, w, c)
            # expanding dims twice means expanding h and w
            seg_x = np.expand_dims(seg_x, axis=1)
            seg_x = np.expand_dims(seg_x, axis=1)
            seg_x = torch.tensor(seg_x, dtype=torch.float32).to(args.device)

            with torch.no_grad():  # Ensure no gradients are calculated
                # Get predictions from the models
                seg_y1_pred = trained_model_1(seg_x)
                seg_y2_pred = trained_model_2(seg_x)
                seg_y1_pred = torch.argmax(seg_y1_pred, dim=1)
                seg_y2_pred = torch.argmax(seg_y2_pred, dim=1)
            
                seg_y_list_pred = [seg_y1_pred, seg_y2_pred]

                scene_list_1_pred = series2graph(seg_y_list_pred[0], label_list[0], show_graph = False, show_other = show_other)
                scene_list_2_pred = series2graph(seg_y_list_pred[1], label_list[1], show_graph = False)

                scene_lists_pred = [scene_list_1_pred, scene_list_2_pred]
            
            # modify question generator: it takes 2 sets of scene_list (real and predicted), and 2 answers.
            question_family_index, question_nl, answer_nl, answer_nl_p, question_struct = question_generator(scene_lists, 
                                                                                                             scene_lists_pred,
                                                                                                             question_family_file,
                                                                                                             label_list,
                                                                                                             show_other = show_other,
                                                                                                             question_validation = True,
                                                                                                             diagnose = False)