from train import sample_evaluation
from config import parse_option
import os
from utils import set_loader, set_model, set_optimizer, adjust_learning_rate
import torch
import numpy as np
from sklearn.metrics import accuracy_score,f1_score

def main():
    opt = parse_option()
    opt2 = parse_option()

    # Build data loader
    train_loader, test_loader = set_loader(opt)

    # Build two models
    model1, _ = set_model(opt)
    opt2.model = opt.model_2
    train_loader2, test_loader2 = set_loader(opt2)
    model2, _ = set_model(opt2)

    print(opt.model_1_path)
    check1 = torch.load(opt.model_1_path, map_location=torch.device('cpu'))

    model1.load_state_dict(check1['model'])
    model1.eval()
    check2 = torch.load(opt2.model_2_path, map_location=torch.device('cpu'))
    model2.load_state_dict(check2['model'])
    model2.eval()

    # Set devices
    model1 = model1.to(opt.device)
    model2 = model2.to(opt2.device)

    # Evaluate on test set
    out_list = []
    label_list = []

    with torch.no_grad():
        for (image, bio_tensor), (image2, _) in zip(test_loader, test_loader2):
            # Prepare input and labels
            images = image.float().to(opt.device)
            images2 = image2.float().to(opt.device)
            labels = bio_tensor.float()
            label_list.append(labels.squeeze().detach().cpu().numpy())
            # Forward pass through both models
            # Model1 probabilities
            output1 = model1(images)  
            # Model2 probabilities
            output2 = model2(images2) 
            # Ensemble predictions (weighted average)
            weight1 = 0.6
            weight2 = 0.4
            ensemble_output = torch.sigmoid(weight1 * output1 + weight2 * output2)
            ensemble_output = torch.round(ensemble_output)  # Binary predictions
            # print(output1 == ensemble_output)
            out_list.append(ensemble_output.squeeze().detach().cpu().numpy())

    # Evaluate ensemble predictions

    accuracy = accuracy_score(label_list, out_list)
    print(f"Ensemble Accuracy: {accuracy:.2%}")
    f = f1_score(label_list,out_list,average='macro')
    print(f"F1 Score (macro): {f:.4f}")

def get_all_results():
    opt = parse_option()
    opt2 = parse_option()
    opt2.model = opt2.model_2

    # Build data loaders
    train_loader, test_loader = set_loader(opt)
    train_loader2, test_loader2 = set_loader(opt2)

    # Build and load two models
    model1, _ = set_model(opt)
    model2, _ = set_model(opt2)

    print(opt.model_1_path)
    check1 = torch.load(opt.model_1_path, map_location=torch.device('cpu'))
    model1.load_state_dict(check1['model'])
    model1.eval()

    print(opt2.model_2_path)
    check2 = torch.load(opt2.model_2_path, map_location=torch.device('cpu'))
    model2.load_state_dict(check2['model'])
    model2.eval()

    # Set devices
    model1 = model1.to(opt.device)
    model2 = model2.to(opt2.device)

    # Evaluate individual models using `sample_evaluation`
    print("Evaluating Model 1...")
    out_list1, label_list1 = sample_evaluation(test_loader, model1, opt)
    print("Evaluating Model 2...")
    out_list2, label_list2 = sample_evaluation(test_loader2, model2, opt2)
    # Ensure labels are consistent between loaders
    assert np.array_equal(label_list1, label_list2), "Label mismatch between test loaders"
    label_array = np.array(label_list1)
    print(out_list1.all() == out_list2.all())

    # Ensemble predictions (weighted average)
    weight1 = 0.5
    weight2 = 0.5
    # print(out_list1.shape)
    ensemble_output = (weight1 * np.array(out_list1) + weight2 * np.array(out_list1))
    # print(ensemble_output == np.array(out_list1))
    ensemble_output = np.round(ensemble_output)  # Binary predictions
    # print(ensemble_output)
    print(out_list1.all() == ensemble_output.all())
    accuracy = accuracy_score(label_array, ensemble_output)
    f1 = f1_score(label_array, ensemble_output, average='macro')
    
    print(f"Ensemble Accuracy: {accuracy:.2%}")
    print(f"F1 Score (macro): {f1:.4f}")



if __name__ == '__main__':
    main()
    