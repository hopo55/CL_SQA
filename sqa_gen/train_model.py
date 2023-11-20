import os
import numpy as np

from sqa_gen.data_extraction import extract_data_from_file
from models.ConvLSTM import ConvLSTM

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset

def train(args, model, model_name, train_x, train_y):
    ############## training ##############
    model_directory = args.save_model_folder + '/opp_model'
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    save_path = model_directory + model_name + '.pt'

    # Creating DataLoader for PyTorch
    # If train_x and train_y are already tensors, use them directly
    if isinstance(train_x, torch.Tensor) and isinstance(train_y, torch.Tensor):
        dataset = TensorDataset(train_x.float(), train_y.long())
    else:
        # If they are NumPy arrays or lists, convert them to tensors
        dataset = TensorDataset(torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_y, dtype=torch.long))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            outputs = model(data)
            
            loss = criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(args.device), target.to(args.device)
                outputs = model(data)
                val_loss += criterion(outputs, target).item()
                
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
        val_accuracy = correct / total
        print(f"Epoch: {epoch+1}, Training Loss: {loss.item()}, Validation Loss: {val_loss/len(val_loader)}, Validation Accuracy: {val_accuracy}")

        # Save the best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            # torch.save(model, save_path)
            print(f"Model saved to {save_path}")


def train_opp_model(args, datapath):
    
    # preparing training data
    train_list = ['S1-ADL1.dat', 'S1-ADL2.dat', 'S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat',
                 'S2-ADL1.dat', 'S2-ADL2.dat', 'S2-ADL3.dat', 'S2-ADL4.dat', 'S2-ADL5.dat', 'S2-Drill.dat',
                 'S3-ADL1.dat', 'S3-ADL2.dat', 'S3-ADL3.dat', 'S3-ADL4.dat', 'S3-ADL5.dat', 'S3-Drill.dat'
                ]

    label1 = np.zeros([0,1])
    label2 = np.zeros([0,1])
    data = np.zeros([0, args.dim])

    for file_i in train_list:
        label_y, label_list, data_x = extract_data_from_file(file_i, datapath,
                                                             plot_option = False,
                                                             show_other = False)
        
        label1 = np.concatenate((label1, label_y[0].reshape([-1, 1])), axis = 0)
        label2 = np.concatenate((label2, label_y[1].reshape([-1, 1])), axis = 0)
        data = np.concatenate((data, data_x), axis = 0)

    train_x = data
    # Squeezing to remove the redundant dimension
    train_y1 = torch.tensor(label1, dtype=torch.float32).squeeze() - 1
    train_y2 = torch.tensor(label2, dtype=torch.float32).squeeze() - 1
    train_x = np.expand_dims(train_x, axis=1)
    train_x = np.expand_dims(train_x, axis=1)

    # train model
    print('model training ...')
    model_1 = ConvLSTM(dim=args.dim, win_len=args.win_len, num_classes_1=args.num_class1, num_feat_map=args.feature, dropout_rate=args.drop_rate)
    model_2 = ConvLSTM(dim=args.dim, win_len=args.win_len, num_classes_1=args.num_class2, num_feat_map=args.feature, dropout_rate=args.drop_rate)

    model_1.to(args.device)
    model_2.to(args.device)

    # train mid
    train(args, model_1, 'single_1', train_x, train_y1)
    # train loc
    train(args, model_2, 'single_2', train_x, train_y2)


def test_opp_model():
    # preparing test data
    test_list = ['S4-ADL1.dat', 'S4-ADL2.dat', 'S4-ADL3.dat', 'S4-ADL4.dat', 'S4-ADL5.dat', 'S4-Drill.dat']

    label1 = np.zeros([0,1])
    label2 = np.zeros([0,1])
    data = np.zeros([0,77])

    for file_i in test_list:
        label_y, label_list, data_x = extract_data_from_file(file_i, datapath,
                                                             plot_option = False, 
                                                             show_other = False)
        label1 = np.concatenate((label1, label_y[0].reshape([-1, 1])), axis = 0)
        label2 = np.concatenate((label2, label_y[1].reshape([-1, 1])), axis = 0)
        data = np.concatenate((data, data_x), axis = 0)

    test_x = data
    # Squeezing to remove the redundant dimension
    test_y1 = torch.tensor(label1, dtype=torch.int64).squeeze() - 1
    test_y2 = torch.tensor(label2, dtype=torch.int64).squeeze() - 1
    test_x = np.expand_dims(test_x, axis=-1)
    test_x = np.expand_dims(test_x, axis=-1)