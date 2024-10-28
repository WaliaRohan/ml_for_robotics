1. **Data generation:**

   - Create a `data` directory in the current folder.
   - Run `python3 generate_data.py`
   - This will create `train_data.npz`, `val_data.npz`, and `test_data.npz`. With current settings, this data will take up 10.5 GB of space.

2. **Training:**

   - Set model architecture in lines 49 and 50.
   - Set training parameters in lines 56 and 57.
   - Run `python3 train_val.py`

3. **Viewing training/validation losses for a model:**

   - Ensure the data file name is correct in line 4. This file contains the number of epochs and training/validation losses for each epoch.
   - Run `python3 plot_train_val_losses.py`

4. **Inference and comparison with ground truth:**

   - Create a `plots` directory in the current folder.
   - Ensure the correct model name is used in line 99, and model architecture coincides with what was set in `train_val.py` during training.
   - Run `python3 plot_output.py`

---

**Note:** We skipped uploading dataset files as they are huge. They are available upon request, or you can generate your own dataset files using the instructions provided above. Otherwise, we have provided models as well as training/evaluation losses for those models to conduct inference as well as view our training/validation results.
