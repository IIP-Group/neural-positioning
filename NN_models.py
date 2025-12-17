import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class FCNet(nn.Module):
    # Constructor
    def __init__(self,Layers,p=0):
        super(FCNet, self).__init__()
        self.hidden = nn.ModuleList()
        self.drop = nn.Dropout(p=p) # to allow dropout. For now, p=0
        
        for input_size, output_size in zip(Layers, Layers[1:]): # gets the 1st and 2nd to build the first layer, the 2nd and 3rd to build the second layer, and so on
            self.hidden.append(nn.Linear(input_size, output_size)) # building each hidden layer
            torch.nn.init.xavier_uniform_(self.hidden[-1].weight)
            torch.nn.init.kaiming_uniform_(self.hidden[-1].weight,nonlinearity='relu') # weight initialization (He Method)
            # self.hidden.append(nn.BatchNorm1d(output_size)) # for batch normalization 
    
    # Prediction
    def forward(self, activation):
        L = len(self.hidden) # numbers of layers in the neural network
        for (l, linear_transform) in zip(range(L), self.hidden):
            if 0 < l and l < L - 1:
                if l in [1,2]:
                    activation = F.relu(self.drop(linear_transform(activation)))
                else:
                    activation = F.relu(linear_transform(activation)) # apply ReLU to all layers except batch norm layer and the last one
                    # activation = F.silu(self.drop(linear_transform(activation))) # apply SiLU to all layers but the last one
            else:
                activation = linear_transform(activation) # do not apply ReLU on the last layer
        return activation
    
class ProbabilityNetwork(nn.Module):
    def __init__(self, Layers, dropout_rate=0.):
        super(ProbabilityNetwork, self).__init__()
        self.hidden = nn.ModuleList()
        self.batchnorm = None  # Store BatchNorm separately
        self.drop = nn.Dropout(p=dropout_rate)

        for input_size, output_size in zip(Layers, Layers[1:]): # gets the 1st and 2nd to build the first layer, the 2nd and 3rd to build the second layer, and so on
            self.hidden.append(nn.Linear(input_size, output_size)) # building each hidden layer
            # torch.nn.init.xavier_uniform_(self.hidden[-1].weight)
            torch.nn.init.kaiming_uniform_(self.hidden[-1].weight,nonlinearity='relu') # weight initialization (He Method)
            
            # Apply BatchNorm only after FC1
            if input_size == Layers[0]:
                self.batchnorm = nn.BatchNorm1d(output_size) # for batch normalization
        
    def forward(self, activation):
        L = len(self.hidden) # numbers of layers in the neural network
        for l, layer in enumerate(self.hidden):
            activation = layer(activation)

            # Apply BatchNorm only after the first FC layer
            if l == 0 and self.batchnorm is not None:
                activation = self.batchnorm(activation)

            # Apply ReLU & Dropout to hidden layers
            if l < len(self.hidden) - 1:  # Skip the last layer
                activation = F.relu(activation)
                if l in [1]:  # Apply Dropout after 2nd layer (1st hidden layer when counting from zero)
                    activation = self.drop(activation)

        activation = F.softmax(activation, dim=1)  # Apply Softmax only at the last layer
        return activation
    
class SmallNetwork(nn.Module):
    def __init__(self, device, per_AP_feat_size):
        super(SmallNetwork, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(per_AP_feat_size, per_AP_feat_size*8)
        self.fc2 = nn.Linear(per_AP_feat_size*8, per_AP_feat_size*4)
        self.fc3 = nn.Linear(per_AP_feat_size*4, per_AP_feat_size*2)
        self.fc4 = nn.Linear(per_AP_feat_size*2, per_AP_feat_size)
        self.fc5 = nn.Linear(per_AP_feat_size, 64)
        self.fc6 = nn.Linear(64, 4)
        # Move all layers to the specified device
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        # x = F.relu(self.fc7(x))
        x = self.fc6(x)  # No activation on final layer
        return x

class CombinedNetwork(nn.Module):
    def __init__(self,nof_APs,per_AP_feat_size,device):
        super(CombinedNetwork, self).__init__()
        self.device = device
        self.AP_networks = nof_APs*[None]
        self.nof_APs = nof_APs
        for idx in range(nof_APs):
            self.AP_networks[idx] = SmallNetwork(device,per_AP_feat_size)
        self.fc8 = nn.Linear(nof_APs*4, 16)
        self.fc9 = nn.Linear(16, 2)
        # Move all layers to the specified device
        self.to(self.device)

    def forward(self, x):
        # Process inputs through the independent AP networks
        out = self.nof_APs*[0.0]
        for idx in range(self.nof_APs):
            out[idx] = self.AP_networks[idx](x[:,idx*208:(idx+1)*208])
        
        # Concatenate outputs from the two networks
        combined = torch.cat(out, dim=1)
        
        # Pass through the subsequent layers
        x = F.relu(self.fc8(combined))
        x = self.fc9(x)  # Final output layer
        return x

class SupervisedModel():
    def __init__(self,device,total_feature_size,nof_APs,config,prob_map_size=484):
        # Parameters of the NN
        self.device = device
        self.X_size = int(total_feature_size/nof_APs)
        self.epochs = config["epochs"]
        self.batch_size = 10
        self.learning_rate = config["learning_rate"]
        self.step_size_decay = 20

        if config["architecture"] == "single_NN":
            self.layers = [self.X_size, self.X_size, 512, 256, 64, 2]
            # Architecture 1: One fully-connected network
            self.network = FCNet(self.layers,p=0.0) # change p if you want to apply dropout
            self.network.to(self.device) # convert the layers created in my NN to CUDA tensors
            self.criterion = self.mse
        elif config["architecture"] == "combined_NN":
            # NEW: A combined network with independent networks doing pre-processing per AP
            self.network = CombinedNetwork(nof_APs,self.X_size,device)
            self.criterion = self.mse
        elif config["architecture"] == "probability_NN":
            self.layers = [self.X_size, 512, 512, 512, 512, prob_map_size]
            self.network = ProbabilityNetwork(self.layers, dropout_rate=config["dropout_rate"])
            self.network.to(self.device)
            self.criterion = nn.BCELoss(reduction='mean')
        else:
            raise ValueError(f"Invalid architecture: {config['architecture']}")

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate) # define learning rate and momentum for Stochastic Gradient Descent (SGD) 
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.step_size_decay) # decays the learning rate of each parameter group by gamma every step_size epochs. Default gamma = 0.1
        
    def train(self,data_set,ap=0, noise_var=0.0):
        train_set = torch.utils.data.TensorDataset(torch.t(data_set.X_train[ap]),data_set.Y_train[ap])
        train_loader = DataLoader(dataset=train_set,batch_size = self.batch_size, shuffle=True, drop_last=True) 
        num_batches = len(train_loader)
        
        loss_per_epoch = torch.zeros(self.epochs,device=self.device)
        val_loss_per_epoch = torch.zeros(self.epochs,device=self.device)
        for epoch in range(self.epochs):
            print(f"Training for epoch {epoch}; AP {ap}")
            self.network.train()
            sum_loss = 0
            for x, y in train_loader:
                self.network.zero_grad() # necessary to calculate gradients. set all gradients to zero
                x = x.to(self.device)
                y = y.to(self.device)

                # add gaussian noise
                if noise_var > 0.0:
                    x += torch.randn_like(x) * np.sqrt(noise_var)

                yhat = self.network(x) # performing the prediction
                loss = self.criterion(yhat, y) # comparing y to yhat
                loss.backward() 
                self.optimizer.step()
                sum_loss = sum_loss + loss.item()
            loss_per_epoch[epoch] = sum_loss/num_batches
            val_loss_per_epoch[epoch] = self.validate(data_set,ap,epoch)
            # wandb.log({"training_loss": loss_per_epoch[epoch], "validation_loss": val_loss_per_epoch[epoch]})
            self.scheduler.step()
        return loss_per_epoch, val_loss_per_epoch

    # Use MSE as the loss function
    def mse(self,yhat,y):
        return torch.mean(torch.linalg.norm(yhat-y)**2)
        # return torch.mean(torch.linalg.norm(yhat-y))

    # Test function
    def test(self,data_set,ap=0):
        test_loader = DataLoader(dataset=torch.t(data_set.X_test[ap]),batch_size = self.batch_size, shuffle = False) 
        self.network.eval() # turn off everything that happens in training, like dropout, batch normalization, etc...
        Y_test_all = torch.tensor([])
        for x in test_loader:
            x = x.to(self.device)
            Y_test = self.network(x)
            Y_test = Y_test.cpu().detach()
            Y_test_all = torch.cat((Y_test_all,Y_test),dim=0)
        return Y_test_all

    def validate(self,data_set,ap,epoch):
        validation_set = torch.utils.data.TensorDataset(torch.t(data_set.X_test[ap]),data_set.Y_test[ap])
        validation_loader = DataLoader(dataset=validation_set,batch_size = self.batch_size, shuffle = False)
        num_batches = len(validation_loader)

        self.network.eval() # turn off everything that happens in training, like dropout, batch normalization, etc...
        sum_loss = 0
        print(f"Validation for epoch {epoch}")
        for x, y in validation_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            yhat = self.network(x) # performing the prediction
            loss = self.criterion(yhat, y) # comparing y to yhat
            sum_loss = sum_loss + loss.item()
        return sum_loss/num_batches