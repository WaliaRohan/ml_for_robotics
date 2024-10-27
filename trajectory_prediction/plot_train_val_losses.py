import matplotlib.pyplot as plt
import numpy as np

file_name = "1st_train_val_losses.npz"

train_losses = np.load(file_name)['train_losses']
val_losses = np.load(file_name)['val_losses']
num_epochs=np.load(file_name)['num_epochs']

# Plotting the training and validation loss
plt.figure(figsize=(8, 6), constrained_layout = True)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')

# Add legend in the top right corner
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Save the plot
plt.savefig('training_validation_loss.png')

# Display the plot
plt.show()

