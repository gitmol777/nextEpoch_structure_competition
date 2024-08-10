from src.data import get_dataloaders
from src.model import RNA_net
from src.util import compute_f1, compute_precision, compute_recall, plot_structures
from src.submission_formatter import format_submission

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader = get_dataloaders(batch_size =64, max_length=110, split=0.8, max_data=6000)

# Init model, loss function, optimizer
embedding_dim = 64
model = RNA_net(embedding_dim).to(device)
loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([300])).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
train_losses = []
test_losses = []
train_f1 = []
test_f1 = []
train_prece = []
test_prece = []
train_recall = []
test_recall = []

for epoch in range(25):
    train_loss = 0.0
    test_loss = 0.0
    train_metric = 0.0
    test_metric = 0.0
    train_met_prece = 0.0
    test_met_prece = 0.0
    train_met_recall = 0.0
    test_met_recall = 0.0
    
    # Training loop
    for batch in train_loader:

        #set to gpu
        x = batch['sequence'].to(device)
        y = batch['structure'].to(device)
        
        y_pred = model(x)
        l = loss(y_pred, y)

        # Optimization
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        # Metrics
        train_loss += l.item()
        
        train_metric += compute_f1(y_pred, y)
        train_met_prece += compute_precision(y_pred, y)
        train_met_recall += compute_recall(y_pred, y)

# Validation loop
    for batch in val_loader:

        #set to gpu
        x = batch['sequence'].to(device)
        y = batch['structure'].to(device)
        
        with torch.no_grad(): 
            y_pred = model(x)
            l = loss(y_pred, y)

        # Metrics
        test_loss += l.item()
        test_metric += compute_f1(y_pred, y)
        test_met_prece += compute_precision(y_pred, y)
        test_met_recall += compute_recall(y_pred, y)
        
    # Log and print
    train_losses.append(train_loss/len(train_loader))
    test_losses.append(test_loss/len(val_loader))
    
    
    train_f1.append(train_metric/len(train_loader))
    test_f1.append(test_metric/len(val_loader))

    train_prece.append(train_met_prece/len(train_loader))
    test_prece.append(test_met_prece/len(val_loader))

    train_recall.append(train_met_recall/len(train_loader))
    test_recall.append(test_met_recall/len(val_loader))
    
    print(f'Epoch {epoch} TRAIN::: loss: {train_losses[-1]:.3f}  F1: {train_f1[-1]:.2f}   Precision: {train_prece[-1]:.2f}  Recall: {train_recall[-1]:.2f}')
    print(f'Epoch {epoch} VALIDATION::: loss: {test_losses[-1]:.3f}  F1: {test_f1[-1]:.2f}   Precision: {test_prece[-1]:.2f}  Recall: {test_recall[-1]:.2f}')
    print()



# Test loop
structures = []
for sequence in test_loader[1]:
    # Replace with your model prediction !
    structure = (model(sequence.to(device).unsqueeze(0)).squeeze(0)>0.5).type(torch.int) # Has to be shape (L, L) ! 
    structures.append(structure)

format_submission(test_loader[0], test_loader[1], structures, 'test_pred.csv')
