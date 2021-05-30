from transformers import BertForSequenceClassification, Wav2Vec2CTCTokenizer, BertConfig
from dogDataset import Dogdataset
from torch.utils.data import DataLoader
import torch
from torch import optim
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    # tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('bert-base-uncase
    train_data = Dogdataset('./dataset/train/', './dataset/meta_train.csv', True)
    train_dataloader = DataLoader(train_data, batch_size=10, shuffle=True)

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6).to(device)

    print(model)
    # 定義optimizer和loss_function
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    # number of epochs to train the model
    n_epochs = 80

    valid_loss_min = np.Inf  # track change in validation loss

    for epoch in range(1, n_epochs + 1):

        # keep track of training and validation loss
        train_correct = 0
        valid_correct = 0
        train_loss = 0.0
        valid_loss = 0.0
        print('running epoch: {}'.format(epoch))
        ###################
        # train the model #
        ###################
        model.train()
        for data, target in tqdm(train_dataloader):
            # move tensors to GPU if CUDA is available
            print(data, target)
            data, target = torch.tensor(data).to(device).long(), target.cuda()
            print(data.shape)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)

            # select the class with highest probability
            _, pred = output.max(1)

            # if the model predicts the same results as the true
            # label, then the correct counter will plus 1
            train_correct += pred.eq(target).sum().item()

            # calculate the batch loss
            loss = criterion(output, target)

            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)

        ######################
        # validate the model #
        ######################
        model.eval()
        with torch.no_grad():
            for data, target in tqdm(train_dataloader):
                # move tensors to GPU if CUDA is available

                data, target = data.cuda(), target.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)

                # select the class with highest probability
                _, pred = output.max(1)

                # if the model predicts the same results as the true
                # label, then the correct counter will plus 1
                valid_correct += pred.eq(target).sum().item()

                # calculate the batch loss
                loss = criterion(output, target)
                # update average validation loss
                valid_loss += loss.item() * data.size(0)

        # calculate average losses
        # train_losses.append(train_loss/len(train_loader.dataset))
        # valid_losses.append(valid_loss.item()/len(valid_loader.dataset)
        train_loss = train_loss / len(train_dataloader)
        valid_loss = valid_loss / len(train_dataloader)
        train_correct = 100. * train_correct / len(train_dataloader)
        valid_correct = 100. * valid_correct / len(train_dataloader)

        # print training/validation statistics
        print(
            '\tTraining Acc: {:.6f} \tTraining Loss: {:.6f} \tValidation Acc: {:.6f} \tValidation Loss: {:.6f}'.format(
                train_correct,
                train_loss, valid_correct, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), 'model_CNN.pth')
            valid_loss_min = valid_loss

    print('Finished Training')