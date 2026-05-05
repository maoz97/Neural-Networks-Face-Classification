import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from helpers import *
import pandas as pd

np.random.seed(42)
torch.manual_seed(42)

class EuropeDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:ש
            csv_file (string): Path to the CSV file with annotations.
        """
        #### YOUR CODE HERE ####
        # Load the data into a tensors
        # The features shape is (n,d)
        # The labels shape is (n)
        # The feature dtype is float
        # THe labels dtype is long
        df = pd.read_csv(csv_file)
        self.features = torch.tensor(df[['long', 'lat']].values, dtype=torch.float32)
        self.labels = torch.tensor(df['country'].values, dtype=torch.long)
        #### END OF YOUR CODE ####
        

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        #### YOUR CODE HERE ####
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data row
        
        Returns:
            dictionary or list corresponding to a feature tensor and it's corresponding label tensor
        """
        #### YOUR CODE HERE ####
        return self.features[idx], self.labels[idx]
    

class MLP(nn.Module):
    def __init__(self, num_hidden_layers, hidden_dim, output_dim):
        super(MLP, self).__init__()
        """
        Args:
            num_hidden_layers (int): The number of hidden layers, in total you'll have an extra layer at the end, from hidden_dim to output_dim
            hidden_dim (int): The hidden layer dimension
            output_dim (int): The output dimension, should match the number of classes in the dataset
        """
        #### YOUR CODE HERE ####
        layers = []
        input_dim = 2

        for i in range(num_hidden_layers):
            if i == 0: #for first layer
                in_d = input_dim
            else:
                in_d = hidden_dim
            layers.append(nn.Linear(in_d, hidden_dim))
            # layers.append(nn.BatchNorm1d(hidden_dim)) #for question 6.3 add this batch norm!!
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers) #create one model

    def forward(self, x):
        #### YOUR CODE HERE ####
       return self.network(x)


def train(train_dataset, val_dataset, test_dataset, model, lr=0.001, epochs=50, batch_size=256, batch_loss_buffer=None):    

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)    
    
    #### YOUR CODE HERE ####
    # initialize your criterion and optimizer here
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accs = [] #save stats per epoch
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []

    for ep in range(epochs):
        #
        model.train()
        #### YOUR CODE HERE ####
        # perform training epoch here
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in trainloader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward() #calculate gradients
            optimizer.step() #update weights

            if batch_loss_buffer is not None: #save loss and calc accuracy
                batch_loss_buffer.append(loss.item())
            running_loss += loss.item() * x.size(0)
            not_relevant, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        
        train_losses.append(running_loss / total)
        train_accs.append(correct / total)

        model.eval() #for testing
        with torch.no_grad():
            #### YOUR CODE HERE ####
            # perform validation loop and test loop here
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            for x, y in valloader:
                outputs = model(x)
                loss = criterion(outputs, y)
    
                val_loss += loss.item() * x.size(0)
                not_relevant, predicted = torch.max(outputs.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
            
            val_losses.append(val_loss / val_total)
            val_accs.append(val_correct / val_total)

            test_loss = 0.0
            test_correct = 0
            test_total = 0
            for x, y in testloader:
                outputs = model(x)
                loss = criterion(outputs, y)
                test_loss += loss.item() * x.size(0)
                not_relevant, predicted = torch.max(outputs.data, 1)
                test_total += y.size(0)
                test_correct += (predicted == y).sum().item()

            test_losses.append(test_loss / test_total)
            test_accs.append(test_correct / test_total)
                
        print('Epoch {:}, Train Acc: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(ep, train_accs[-1], val_accs[-1], test_accs[-1]))        

    return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses

#question 6.1
def run_learning_rate_experiment(train_dataset, val_dataset, test_dataset):
    learning_rates = [1, 0.01, 0.001, 0.00001] #all rates to test
    results = {}
    num_classes = len(torch.unique(train_dataset.labels))
    for lr in learning_rates:
        print(f"\n Training with LR = {lr} ")
        model = MLP(num_hidden_layers=6, hidden_dim=16, output_dim=num_classes) #new model for each lr
        train_result = train(train_dataset, val_dataset, test_dataset, model, lr=lr, epochs=50, batch_size=256)
        val_losses = train_result[5] #get validation loss
        results[lr] = val_losses

    #plotting
    plt.figure(figsize=(10, 6))
    for lr, losses in results.items():
        plt.plot(losses, label=f'LR={lr}')
    plt.title('Validation Loss per Epoch for different Learning Rates')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

#question 6.2
def run_epoch_experiment(train_dataset, val_dataset, test_dataset):
    num_classes = len(torch.unique(train_dataset.labels))
    model = MLP(num_hidden_layers=6, hidden_dim=16, output_dim=num_classes)
    #training for 100 epochs
    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train(
        train_dataset, val_dataset, test_dataset, 
        model, lr=0.001, epochs=100, batch_size=256
    )
    #plotting
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='red')
    plt.plot(val_losses, label='Validation Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='green', linestyle='--')
    plt.title('Loss over 100 Epochs (Train vs Val vs Test)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

#question 6.4 
def run_batch_size_experiment(train_dataset, val_dataset, test_dataset):
    batch_sizes = [
        (1, 1),
        (16, 10),
        (128, 50),
        (1024, 50)
    ]
    num_classes = len(torch.unique(train_dataset.labels))
    results_val_acc = {}      
    results_batch_losses = {} 
    
    for batch_size, epochs in batch_sizes:
        iters_per_epoch = len(train_dataset) / batch_size
        print(f"{batch_size:<15} | {epochs:<10} | {iters_per_epoch:.1f}") #step
        model = MLP(num_hidden_layers=6, hidden_dim=16, output_dim=num_classes)
        current_batch_losses = [] 
        
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train(
            train_dataset, val_dataset, test_dataset, 
            model, lr=0.001, epochs=epochs, batch_size=batch_size,
            batch_loss_buffer=current_batch_losses 
        )
        
        #for graphs
        results_val_acc[batch_size] = val_accs
        results_batch_losses[batch_size] = current_batch_losses

    
    #graph 1- Validation Accuracy vs Epoch
    plt.figure(figsize=(10, 6))
    for bs, accs in results_val_acc.items():
        plt.plot(accs, label=f'Batch Size {bs}', marker='o')
    plt.title('(i) Validation Accuracy vs Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    #graph 3- Training Loss vs Batch Iterations (Stability)
    plt.figure(figsize=(12, 6))
    for bs, losses in results_batch_losses.items():
        plt.plot(losses, label=f'Batch Size {bs}', alpha=0.6, linewidth=0.8)
        
    plt.title('(iii) Training Loss vs Batch Iterations (Stability)')
    plt.xlabel('Total Batch Iterations (Steps)')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

#question 6.2.1 + 6.2.2
def run_architecture_experiment(train_dataset, val_dataset, test_dataset):
    sizes = [
        (1, 16),
        (2, 16),
        (6, 16),
        (10, 16),
        (6, 8),
        (6, 32),
        (6, 64)
    ]
    num_classes = len(torch.unique(train_dataset.labels))
    best_accuracy = -1.0
    worst_accuracy = 2.0 #catch lowest
    
    best_model_data = None
    worst_model_data = None

    for depth, width in sizes:
        # num_hidden_layers = depth
        # hidden_dim = width
        model = MLP(num_hidden_layers=depth, hidden_dim=width, output_dim=num_classes)
        
        #50 epochs train
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train(
            train_dataset, val_dataset, test_dataset, 
            model, lr=0.001, epochs=50, batch_size=256
        )
        
        final_val_acc = val_accs[-1]
        print(f"{depth:<10} | {width:<10} | {final_val_acc:.3f}")
        #chcek best
        if final_val_acc > best_accuracy:
            best_accuracy = final_val_acc
            best_model_data = {
                'config': (depth, width),
                'model': model,
                'losses': (train_losses, val_losses, test_losses),
                'accs': (train_accs, val_accs, test_accs)
            }    
        #check worst
        if final_val_acc < worst_accuracy:
            worst_accuracy = final_val_acc
            worst_model_data = {
                'config': (depth, width),
                'model': model,
                'losses': (train_losses, val_losses, test_losses),
                'accs': (train_accs, val_accs, test_accs)
            }

    print(f"Best Model: Depth={best_model_data['config'][0]}, Width={best_model_data['config'][1]} (Acc: {best_accuracy:.3f})")
    print(f"Worst Model: Depth={worst_model_data['config'][0]}, Width={worst_model_data['config'][1]} (Acc: {worst_accuracy:.3f})")

    def plot_results(model_data, title_prefix):
        depth, width = model_data['config'] #unpack from dict
        train_l, val_l, test_l = model_data['losses']
        model = model_data['model']
        
        #loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_l, label='Train Loss', color='red')
        plt.plot(val_l, label='Validation Loss', color='blue')
        plt.plot(test_l, label='Test Loss', color='green', linestyle='--')
        plt.title(f'{title_prefix} Model (D={depth}, W={width}) - Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        #decision boundary plot
        X_test = test_dataset.features.numpy()
        y_test = test_dataset.labels.numpy()
        plot_decision_boundaries(model, X_test, y_test, f'{title_prefix} Model Decision Boundaries')

    plot_results(best_model_data, "Best")
    plot_results(worst_model_data, "Worst")

#question 6.2.3
def run_depth_experiment(train_dataset, val_dataset, test_dataset):
    depths = [1, 2, 6, 10]
    fixed_width = 16
    num_classes = len(torch.unique(train_dataset.labels))
    #for final results
    final_train_accs = []
    final_val_accs = []
    final_test_accs = []

    for depth in depths:
        model = MLP(num_hidden_layers=depth, hidden_dim=fixed_width, output_dim=num_classes)
        model, train_accs, val_accs, test_accs, not_relevant1, not_relevant2, not_relevant3 = train(
            train_dataset, val_dataset, test_dataset, 
            model, lr=0.001, epochs=50, batch_size=256
        )
        final_train_accs.append(train_accs[-1])
        final_val_accs.append(val_accs[-1])
        final_test_accs.append(test_accs[-1])
        print(f"{depth:<10} | {val_accs[-1]:.3f}")

    #plot
    plt.figure(figsize=(10, 6))
    plt.plot(depths, final_train_accs, label='Train Acc', marker='o', color='red')
    plt.plot(depths, final_val_accs, label='Val Acc', marker='o', color='blue')
    plt.plot(depths, final_test_accs, label='Test Acc', marker='o', color='green', linestyle='--')
    
    plt.title(f'Accuracy vs. Network Depth (Width={fixed_width})')
    plt.xlabel('Number of Hidden Layers (Depth)')
    plt.ylabel('Final Accuracy')
    plt.xticks(depths) 
    plt.legend()
    plt.grid(True)
    plt.show()

#question 6.2.4
def run_width_experiment(train_dataset, val_dataset, test_dataset):
    widths = [8, 16, 32, 64]
    fixed_depth = 6
    num_classes = len(torch.unique(train_dataset.labels))
    
    final_train_accs = []
    final_val_accs = []
    final_test_accs = []

    for width in widths:
        model = MLP(num_hidden_layers=fixed_depth, hidden_dim=width, output_dim=num_classes)
        model, train_accs, val_accs, test_accs, not_relevant1, not_relevant2, not_relevant3 = train(
            train_dataset, val_dataset, test_dataset, 
            model, lr=0.001, epochs=50, batch_size=256
        )
        
        final_train_accs.append(train_accs[-1])
        final_val_accs.append(val_accs[-1])
        final_test_accs.append(test_accs[-1])
        print(f"{width:<10} | {val_accs[-1]:.3f}")

    #plot
    plt.figure(figsize=(10, 6))
    plt.plot(widths, final_train_accs, label='Train Acc', marker='o', color='red')
    plt.plot(widths, final_val_accs, label='Val Acc', marker='o', color='blue')
    plt.plot(widths, final_test_accs, label='Test Acc', marker='o', color='green', linestyle='--')
    
    plt.title(f'Accuracy vs. Network Width (Depth={fixed_depth})')
    plt.xlabel('Number of Neurons (Width)')
    plt.ylabel('Final Accuracy')
    plt.xticks(widths) 
    plt.legend()
    plt.grid(True)
    plt.show()

#question 6.2.5
def run_gradient_experiment(train_dataset):
    #setting parameters
    num_hidden_layers = 100
    hidden_dim = 4
    num_classes = len(torch.unique(train_dataset.labels))
    epochs = 10
    batch_size = 256
    layers_to_monitor = [0, 30, 60, 90, 95, 99]
    
    #crearte model, optimizer, criterion, dataloader
    model = MLP(num_hidden_layers=num_hidden_layers, hidden_dim=hidden_dim, output_dim=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    #save the average grad per epoch for each layer
    grads_history = {layer_idx: [] for layer_idx in layers_to_monitor}
    
    for epoch in range(epochs):
        epoch_grad_sums = {layer_idx: 0.0 for layer_idx in layers_to_monitor} #per epoch sum
        num_batches = 0
        
        for x, y in trainloader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            
            #all linear layers in the model
            linear_layers = [m for m in model.network if isinstance(m, nn.Linear)]
            
            for layer_idx in layers_to_monitor:
                layer = linear_layers[layer_idx]
                
                #calc the norm squared
                if layer.weight.grad is not None:
                    grad_norm = layer.weight.grad.norm(2).item() ** 2
                    epoch_grad_sums[layer_idx] += grad_norm
            
            optimizer.step()
            num_batches += 1
            
        #average grad per layer for this epoch
        for layer_idx in layers_to_monitor:
            avg_grad = epoch_grad_sums[layer_idx] / num_batches
            grads_history[layer_idx].append(avg_grad)
            
        print(f"Epoch {epoch+1}/{epochs} done.")

    #plot
    plt.figure(figsize=(12, 7))
    for layer_idx in layers_to_monitor:
        plt.plot(grads_history[layer_idx], label=f'Layer {layer_idx}')
    
    plt.title('Mean Gradient Magnitude per Layer over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Gradient Magnitude (||grad||^2)')
    plt.yscale('log') 
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.show()

#question 6.2.6
class BnMLP(nn.Module):
    def __init__(self, num_hidden_layers, hidden_dim, output_dim):
        super(BnMLP, self).__init__()
        layers = []
        input_dim = 2
        for i in range(num_hidden_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim)) #adding BatchNorm
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def run_bonus_experiment(train_dataset):
    num_hidden_layers = 100
    hidden_dim = 4
    num_classes = len(torch.unique(train_dataset.labels))
    epochs = 10
    batch_size = 256
    layers_to_monitor = [0, 30, 60, 90, 95, 99]
    
    #with BatchNorm model
    model = BnMLP(num_hidden_layers=num_hidden_layers, hidden_dim=hidden_dim, output_dim=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    grads_history = {layer_idx: [] for layer_idx in layers_to_monitor}
    
    for epoch in range(epochs):
        epoch_grad_sums = {layer_idx: 0.0 for layer_idx in layers_to_monitor}
        num_batches = 0
        
        for x, y in trainloader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            linear_layers = [m for m in model.network if isinstance(m, nn.Linear)]
            
            for layer_idx in layers_to_monitor:
                layer = linear_layers[layer_idx]
                if layer.weight.grad is not None:
                    grad_norm = layer.weight.grad.norm(2).item() ** 2
                    epoch_grad_sums[layer_idx] += grad_norm
            
            optimizer.step()
            num_batches += 1
            
        for layer_idx in layers_to_monitor:
            avg_grad = epoch_grad_sums[layer_idx] / num_batches
            grads_history[layer_idx].append(avg_grad)
            

    #plot
    plt.figure(figsize=(12, 7))
    for layer_idx in layers_to_monitor:
        plt.plot(grads_history[layer_idx], label=f'Layer {layer_idx}')
    plt.title('Bonus: Gradient Magnitude with Batch Normalization')
    plt.xlabel('Epochs')
    plt.ylabel('Gradient Magnitude (||grad||^2)')
    plt.yscale('log') 
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.show()

#question 6.2.7
class ImplicitMLP(nn.Module):
    def __init__(self, num_hidden_layers, hidden_dim, output_dim):
        super(ImplicitMLP, self).__init__()
        self.alphas = torch.arange(0.1, 1.1, 0.1) #0.1 to 1.0 with step 0.1
        input_dim = 2 * len(self.alphas) # 2 *10 = 20
        
        layers = []
        for i in range(num_hidden_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.ReLU()) #no batch norm here
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):    
        #x is [Batch, 2]    
        features = []
        device = x.device
        self.alphas = self.alphas.to(device)
        
        for alpha in self.alphas:
            # sin(alpha * x)
            features.append(torch.sin(alpha * x))
        out = torch.cat(features, dim=1) #join pairs
        return self.network(out)

def run_implicit_bonus_experiment(train_dataset, val_dataset, test_dataset):    
    num_classes = len(torch.unique(train_dataset.labels))
    epochs = 50
    batch_size = 256
    
    #regular MLP
    std_model = MLP(num_hidden_layers=6, hidden_dim=16, output_dim=num_classes)
    std_model, _, _, _, _, _, _ = train(
        train_dataset, val_dataset, test_dataset, 
        std_model, lr=0.001, epochs=epochs, batch_size=batch_size
    )
    #implicit MLP
    imp_model = ImplicitMLP(num_hidden_layers=6, hidden_dim=16, output_dim=num_classes)
    imp_model, _, _, _, _, _, _ = train(
        train_dataset, val_dataset, test_dataset, 
        imp_model, lr=0.001, epochs=epochs, batch_size=batch_size
    )
    #compare 
    X_test = test_dataset.features.numpy()
    y_test = test_dataset.labels.numpy()
    
    #plot
    plot_decision_boundaries(std_model, X_test, y_test, 'Standard Model Decision Boundaries', implicit_repr=False)
    plot_decision_boundaries(imp_model, X_test, y_test, 'Implicit Model Decision Boundaries', implicit_repr=False)


if __name__ == '__main__':
    # seed for reproducibility
    torch.manual_seed(42)    
    np.random.seed(42)

    train_dataset = EuropeDataset('train.csv')
    val_dataset = EuropeDataset('validation.csv')
    test_dataset = EuropeDataset('test.csv')

    #### YOUR CODE HERE ##### all the experiments functions called here:
    run_learning_rate_experiment(train_dataset, val_dataset, test_dataset) 
    run_epoch_experiment(train_dataset, val_dataset, test_dataset)
    run_batch_size_experiment(train_dataset, val_dataset, test_dataset)
    run_architecture_experiment(train_dataset, val_dataset, test_dataset)
    run_depth_experiment(train_dataset, val_dataset, test_dataset)
    run_width_experiment(train_dataset, val_dataset, test_dataset)
    run_gradient_experiment(train_dataset)
    run_bonus_experiment(train_dataset)
    run_implicit_bonus_experiment(train_dataset, val_dataset, test_dataset)


    # Find the number of classes, e.g.:
    # output_dim = len(train_dataset.labels.unique())
    # model = MLP(6, 16, output_dim)

    # model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = train(train_dataset, val_dataset, test_dataset, model, lr=0.001, epochs=50, batch_size=256)

    # plt.figure()
    # plt.plot(train_losses, label='Train', color='red')
    # plt.plot(val_losses, label='Val', color='blue')
    # plt.plot(test_losses, label='Test', color='green')
    # plt.title('Losses')
    # plt.legend()
    # plt.show()

    # plt.figure()
    # plt.plot(train_accs, label='Train', color='red')
    # plt.plot(val_accs, label='Val', color='blue')
    # plt.plot(test_accs, label='Test', color='green')
    # plt.title('Accs.')
    # plt.legend()
    # plt.show()



    # train_data = pd.read_csv('train.csv')
    # val_data = pd.read_csv('validation.csv')
    # test_data = pd.read_csv('test.csv')
    # plot_decision_boundaries(model, test_data[['long', 'lat']].values, test_data['country'].values, 'Decision Boundaries', implicit_repr=False)
