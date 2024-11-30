from train import sample_evaluation
from config import parse_option
import os
from utils import set_loader, set_model, set_optimizer, adjust_learning_rate
import torch
import numpy as np

def main():
    opt = parse_option()

    # Build data loader
    train_loader, test_loader = set_loader(opt)

    # Build two models
    model1, _ = set_model(opt)
    opt.model = opt.model_2
    model2, _ = set_model(opt)

    print(opt.model_1_path)
    check1 = torch.load(opt.model_1_path, map_location=torch.device('cpu'))

    model1.load_state_dict(check1['model'])
    model1.eval()
    check2 = torch.load(opt.model_2_path, map_location=torch.device('cpu'))
    model2.load_state_dict(check2['model'])
    model2.eval()

    # Set devices
    model1 = model1.to(opt.device)
    model2 = model2.to(opt.device)

    # Evaluate on test set
    out_list = []
    label_list = []

    with torch.no_grad():
        for idx, (image, bio_tensor) in enumerate(test_loader):
            # Prepare input and labels
            images = image.float().to(opt.device)
            labels = bio_tensor.float()
            label_list.append(labels.squeeze().detach().cpu().numpy())

            # Forward pass through both models
            output1 = torch.sigmoid(model1(images))  # Model1 probabilities
            output2 = torch.sigmoid(model2(images))  # Model2 probabilities

            # Ensemble predictions (weighted average)
            weight1 = 0.6
            weight2 = 0.4
            ensemble_output = (weight1 * output1 + weight2 * output2)
            ensemble_output = torch.round(ensemble_output)  # Binary predictions

            out_list.append(ensemble_output.squeeze().detach().cpu().numpy())

    # Concatenate predictions and labels
    label_array = np.concatenate(label_list, axis=0)
    out_array = np.concatenate(out_list, axis=0)

    # Evaluate ensemble predictions
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(label_array, out_array)
    print(f"Ensemble Accuracy: {accuracy:.2%}")


if __name__ == '__main__':
    main()
