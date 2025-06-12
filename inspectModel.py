import torch

# Load the model file
state_dict = torch.load("skinLesionModel.pth", map_location='cpu')

# If it's a checkpoint dict, extract actual weights
if isinstance(state_dict, dict) and "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]

# Print all layer names (keys) in the state_dict
print("Layer names in the state_dict:")
for key in state_dict.keys():
    print(key)
