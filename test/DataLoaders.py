from torch.utils.data import Dataloader

traing_dataloader = Dataloader(training_data, batch_size=64, shuffle=True)
test_dataloader = Dataloader(test_data, batch_size=64, shuffle=True)

traing_features, traing_labels = next(iter(traing_dataloader))
print(f"Feature batch shape: {traing_features.size()}")
print(f"Labels batch shape: {traing_features.size()}")
img = traing_features[0].squeeze()
label = traing_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")