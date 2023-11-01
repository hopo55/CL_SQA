import numpy as np
from tqdm import tqdm
from datetime import date

import torch

from sqa_gen.data_extraction import extract_data_from_file, data_split_tools, series2graph

def sqa_gen_engine(file_list,
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
    save_path = save_model_folder+ 'opp_model/'+ model_name+'.hdf5'
    trained_model_1 = torch.load(save_path)

    model_name = 'single_2'
    save_path = save_model_folder + 'opp_model/'+ model_name+'.hdf5'
    trained_model_2 = torch.load(save_path)

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
            scene_list_1 = series2graph(seg_y_list[0], label_list[0], show_graph = False, show_other = show_other)
            scene_list_2 = series2graph(seg_y_list[1], label_list[1], show_graph = False)
            scene_lists = [scene_list_1, scene_list_2]

            # =====  get predicted scene_list using pre-trained source classifier.=====
            # need to reshape the input X
            seg_x = np.expand_dims(seg_x, axis=-1)
            seg_x = np.expand_dims(seg_x, axis=-1)
            
            seg_y_list_pred = [np.argmax(trained_model_1.predict(seg_x), axis=1), 
                                np.argmax(trained_model_2.predict(seg_x), axis=1)]
            scene_list_1_pred = series2graph(seg_y_list_pred[0], label_list[0], show_graph = False, show_other = show_other)
            scene_list_2_pred = series2graph(seg_y_list_pred[1], label_list[1], show_graph = False)
            scene_lists_pred = [scene_list_1_pred, scene_list_2_pred]