import matplotlib.pyplot as plt
import numpy as np

num_epochs=5
train_losses = np.load("train_val_losses.npz")['train_losses']
val_losses = np.load("train_val_losses.npz")['val_losses']


# Plotting the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')

# Set y-axis range and increments
plt.ylim(0, 100)
# plt.yticks(range(0, 101, 5e))  # Y-axis from 0 to 100 in steps of 5

# Set x-axis to show only integer values
plt.xticks(range(1, num_epochs + 1))

# Add legend in the top right corner
plt.legend(loc='upper right')
plt.grid()

# Display the plot
plt.show()

# Save the plot
plt.savefig('training_validation_loss.png')