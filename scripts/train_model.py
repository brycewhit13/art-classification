# Imports
import os
import numpy as np
from load_data import load_data
import torch
from torchvision.models import resnet50, ResNet50_Weights

def process_images():
    # Load the data
    human_images, ai_images = load_data()
    
    # Resize the images to be 224x224
    human_images_resized = [image.resize((224, 224)) for image in human_images]
    ai_images_resized = [image.resize((224, 224)) for image in ai_images]
    
    # Convert the PIL images to numpy arrays
    human_images_resized = [np.moveaxis(np.array(image), -1, 0) for image in human_images_resized]
    ai_images_resized = [np.moveaxis(np.array(image), -1, 0) for image in ai_images_resized]
    #print("image shape: ", np.array(ai_images_resized[0]).shape)
    #print("image: ", np.array(ai_images_resized[0]))
    
    # Seperate into train and test sets
    human_train = human_images_resized[:int(len(human_images_resized) * 0.8)]
    human_val = human_images_resized[int(len(human_images_resized) * 0.8):]
    ai_train = ai_images_resized[:int(len(ai_images_resized) * 0.8)]
    ai_val = ai_images_resized[int(len(ai_images_resized) * 0.8):]
    
    # Create Label info
    human_train_labels = np.zeros(len(human_train))
    human_val_labels = np.zeros(len(human_val))
    ai_train_labels = np.ones(len(ai_train))
    ai_val_labels = np.ones(len(ai_val))
    
    # Combine the train data
    train_images = np.array([image for image in human_train] + [image for image in ai_train])
    train_labels = np.array([label for label in human_train_labels] + [label for label in ai_train_labels])
    val_images = np.array([image for image in human_val] + [image for image in ai_val])
    val_labels = np.array([label for label in human_val_labels] + [label for label in ai_val_labels])
    
    # Create the datasets and dataloaders
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_images), torch.tensor(train_labels))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(val_images), torch.tensor(val_labels))
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)
    
    # Return the dataloaders
    return train_dataloader, val_dataloader    

def train_model(train_dataloader, epochs=1):
    # Instantiate the model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # Freeze the layers that are already trained
    for param in model.parameters():
        param.requires_grad = False
    # Replace the output layer
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    
    # Define the criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Track the running loss and number of correct predictions
    running_loss = 0
    running_correct = 0
    
    # Train the model
    model.train()
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}-------------------------------")
        for image, label in train_dataloader:
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass
            output = model(image.float())
            _, preds = torch.max(output, 1)
            # Calculate the loss
            loss = criterion(output, label.type(torch.LongTensor))
            # Backward pass
            loss.backward()
            # Update the weights
            optimizer.step()
            
            # Update the running loss and number of correct predictions
            if epoch == epochs - 1:
                running_loss += loss.item()
                running_correct += torch.sum(preds == label)
            
    # Calculate the average loss and accuracy
    acc = running_correct / len(train_dataloader.dataset)
    print(f"Training accuracy: {acc}\n")
            
    # Return the trained model
    return model
                     
def evaluate_model(trained_model, val_dataloader):
    # Define the criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    
    # Track the running loss and number of correct predictions
    running_loss = 0
    running_correct = 0
    
    with torch.no_grad():
        # Set the model to eval mode
        trained_model.eval()
        
        for image, label in val_dataloader:
            # Forward pass
            output = trained_model(image.float())
            _, preds = torch.max(output, 1)
            # Calculate the loss
            loss = criterion(output, label.type(torch.LongTensor))
            
            # Update the running loss and number of correct predictions
            running_loss += loss.item()
            running_correct += torch.sum(preds == label)
    
    # Print the accuracy
    val_acc = running_correct / len(val_dataloader.dataset)
    print(f"Validation accuracy: {val_acc}\n")
    
    
def make_prediction(trained_model, image):
    # Pass the image through the model
    output = trained_model(image.float())
    pred = torch.argmax(output)
    
    # Convert the prediction to a string and return it
    if pred == 0:
        return "Human"
    return "AI"

def make_proba_prediction(trained_model, image):
    # Pass the image through the model
    output = trained_model(image.float())
    probas = torch.nn.functional.softmax(output, dim=1)
    
    # Return the probability of the image being AI and human
    output_dict = {"Human": probas[0][0].item(), "AI": probas[0][1].item()}
    return output_dict
    
    
if __name__ == "__main__":
    print("Loading the data...")
    train_dataloader, val_dataloader = process_images()
    
    print("Training the model...")
    trained_model = train_model(train_dataloader=train_dataloader, epochs=2)
    
    print("Evaluating the model...")
    evaluate_model(trained_model=trained_model, val_dataloader=val_dataloader)
    
    # Save the model
    torch.save(trained_model.state_dict(), os.path.join(os.getcwd(), '..', 'models', 'model.pth'))