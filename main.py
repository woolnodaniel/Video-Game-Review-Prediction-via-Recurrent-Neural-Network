#!/usr/bin/env python3
"""
hw2main.py

UNSW COMP9444 Neural Networks and Deep Learning

DO NOT MODIFY THIS FILE
"""

import torch
from torchtext import data

import model

def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.device = device
    print("Using device: {}"
          "\n".format(str(device)))

    # Load the training dataset, and create a dataloader to generate a batch.
    textField = data.Field(lower=True, include_lengths=True, batch_first=True,
                           preprocessing=model.preprocessing,
                           postprocessing=model.postprocessing,
                           stop_words=model.stopWords)
    labelField = data.Field(sequential=False, use_vocab=False, is_target=True)

    dataset = data.TabularDataset('train.json', 'json',
                                 {'reviewText': ('reviewText', textField),
                                  'rating': ('rating', labelField)})
    #print(dataset[0].reviewText)

    textField.build_vocab(dataset, vectors=model.wordVectors)

    # Allow training on the entire dataset, or split it for training and validation.
    if model.trainValSplit == 1:
        trainLoader = data.BucketIterator(dataset, shuffle=True,
                                          batch_size=model.batchSize,
                                          sort_key=lambda x: len(x.reviewText),
                                          sort_within_batch=True)
    else:
        train, validate = dataset.split(split_ratio=model.trainValSplit,
                                        stratified=True, strata_field='rating')

        trainLoader, valLoader = data.BucketIterator.splits(
            (train, validate), shuffle=True, batch_size=model.batchSize,
             sort_key=lambda x: len(x.reviewText), sort_within_batch=True)

    # Get model and optimiser from model.
    net = model.net.to(device)
    criterion = model.lossFunc
    optimiser = model.optimiser

    # Train.
    for epoch in range(model.epochs):
        runningLoss = 0

        for i, batch in enumerate(trainLoader):
            # Get a batch and potentially send it to GPU memory.
            inputs = textField.vocab.vectors[batch.reviewText[0]].to(device)
            length = batch.reviewText[1].to(device)
            labels = batch.rating.type(torch.FloatTensor).to(device)

            # PyTorch calculates gradients by accumulating contributions
            # to them (useful for RNNs).
            # Hence we must manually set them to zero before calculating them.
            optimiser.zero_grad()

            # Forward pass through the network.
            output = net(inputs, length)
            loss = criterion(output, model.convertLabel(labels))

            # Calculate gradients.
            loss.backward()

            # Minimise the loss according to the gradient.
            optimiser.step()

            runningLoss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f"
                      % (epoch + 1, i + 1, runningLoss / 32))
                runningLoss = 0


        if model.trainValSplit != 1:
            net.eval()

            closeness = [0 for _ in range(5)]
            with torch.no_grad():
                for batch in valLoader:
                    # Get a batch and potentially send it to GPU memory.
                    inputs = textField.vocab.vectors[batch.reviewText[0]].to(device)
                    length = batch.reviewText[1].to(device)
                    labels = batch.rating.type(torch.FloatTensor).to(device)

                    # Convert network output to integer values.
                    outputs = model.convertNetOutput(net(inputs, length)).flatten()

                    for i in range(5):
                        closeness[i] += torch.sum(abs(labels - outputs) == i).item()

            accuracy = [x / len(validate) for x in closeness]
            score = 100 * (accuracy[0] + 0.4 * accuracy[1])

            print("\n"
                  "Correct predictions: {:.2%}\n"
                  "One star away: {:.2%}\n"
                  "Two stars away: {:.2%}\n"
                  "Three stars away: {:.2%}\n"
                  "Four stars away: {:.2%}\n"
                  "\n"
                  "Weighted score: {:.2f}".format(*accuracy, score))

    # Save model.
    torch.save(net.state_dict(), 'savedModel.pth')
    print("\n"
          "Model saved to savedModel.pth")

    # Test on validation data if it exists.
    if model.trainValSplit != 1:
        net.eval()

        closeness = [0 for _ in range(5)]
        with torch.no_grad():
            for batch in valLoader:
                # Get a batch and potentially send it to GPU memory.
                inputs = textField.vocab.vectors[batch.reviewText[0]].to(device)
                length = batch.reviewText[1].to(device)
                labels = batch.rating.type(torch.FloatTensor).to(device)

                # Convert network output to integer values.
                outputs = model.convertNetOutput(net(inputs, length)).flatten()
                print(outputs)

                for i in range(5):
                    closeness[i] += torch.sum(abs(labels - outputs) == i).item()

        accuracy = [x / len(validate) for x in closeness]
        score = 100 * (accuracy[0] + 0.4 * accuracy[1])

        print("\n"
              "Correct predictions: {:.2%}\n"
              "One star away: {:.2%}\n"
              "Two stars away: {:.2%}\n"
              "Three stars away: {:.2%}\n"
              "Four stars away: {:.2%}\n"
              "\n"
              "Weighted score: {:.2f}".format(*accuracy, score))

    print(f'Best epoch: {best_epoch}, best score: {best_loss}')

if __name__ == '__main__':
    main()
