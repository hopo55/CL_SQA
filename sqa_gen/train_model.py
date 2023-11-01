import numpy as np

from data_extraction import extract_data_from_file

import torch
import torch.nn as nn
import torch.optim as optim

def train(args, model, model_name, train_x, train_y):
    ############## training ##############   
    save_path = 'opp_model/' + model_name + '.pt'

    # Creating DataLoader for PyTorch
    dataset = TensorDataset(torch.tensor(train_x, dtype=torch.float32), torch.tensor(train_y, dtype=torch.long))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=300, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=300, shuffle=False)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(args.epochs):
        # Training
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
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
            print(f"Model saved to {save_path}")


def train_model(args, datapath):
    
    # preparing training data
    train_list = ['S1-ADL1.dat', 'S1-ADL2.dat', 'S1-ADL3.dat', 'S1-ADL4.dat', 'S1-ADL5.dat', 'S1-Drill.dat',
                 'S2-ADL1.dat', 'S2-ADL2.dat', 'S2-ADL3.dat', 'S2-ADL4.dat', 'S2-ADL5.dat', 'S2-Drill.dat',
                 'S3-ADL1.dat', 'S3-ADL2.dat', 'S3-ADL3.dat', 'S3-ADL4.dat', 'S3-ADL5.dat', 'S3-Drill.dat'
                ]

    label1 = np.zeros([0,1])
    label2 = np.zeros([0,1])
    data = np.zeros([0,77])

    for file_i in train_list:
        label_y, label_list, data_x = extract_data_from_file(file_i, datapath,
                                                             plot_option = False,
                                                             show_other = False)
        label1 = np.concatenate((label1, label_y[0].reshape([-1, 1])), axis = 0)
        label2 = np.concatenate((label2, label_y[1].reshape([-1, 1])), axis = 0)

        data = np.concatenate((data, data_x), axis = 0)

    train_x = data
    train_y1 = label1
    train_y2 = label2

    train_y1 = to_categorical(train_y1-1, num_classes=len(label_list[0] ) )
    train_y2 = to_categorical(train_y2-1, num_classes=len(label_list[1] ) )
    train_x = np.expand_dims(train_x, axis=-1)
    train_x = np.expand_dims(train_x, axis=-1)

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
    test_y1 = label1
    test_y2 = label2

    test_y1 = to_categorical(test_y1-1, num_classes=len(label_list[0] ) )
    test_y2 = to_categorical(test_y2-1, num_classes=len(label_list[1] ) )
    test_x = np.expand_dims(test_x, axis=-1)
    test_x = np.expand_dims(test_x, axis=-1)

    train()



