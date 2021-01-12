from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent, Wasserstein
from art.estimators.classification import PyTorchClassifier
import numpy as np


# ART PytorchClassifier: https://adversarial-robustness-toolbox.readthedocs.io/en/stable/modules/estimators/classification.html?highlight=PyTorchClassifier#art.estimators.classification.PyTorchClassifier
class AttackWrapper:

    def __init__(self, model, train_dataset, criterion, optimizer, min_resize, max_resize):

        self.classifier = PyTorchClassifier(
            model=model,
            # clip_values=(min_pixel_value, max_pixel_value), #This is optional
            loss=criterion,
            optimizer=optimizer,
            input_shape=(3, max_resize, max_resize),
            nb_classes=1000,  # Maybe 999?
        )

        self.x_train=train_dataset

    # %% Step 4: Train the ART classifier
    def fit(self):
        self.classifier.fit(self.x_train, self.y_train, batch_size=4, nb_epochs=5)  # Is this needed?? Our net is already trained

# %% Step 5: Evaluate the ART classifier on benign test examples
    def predict(self):
        predictions = self.classifier.predict(self.x_test)  # Will this still work if we dont fit?
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("Accuracy on benign test examples: {}%".format(accuracy * 100))

    def eval_attack(self, attack):
        # Step 6: Generate adversarial test examples
        x_test_adv = attack.generate(x=self.x_train)  # TODO: Should be a testset, not a trainset!

        # Step 7: Evaluate the ART classifier on adversarial test examples
        predictions = self.classifier.predict(x_test_adv)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(self.y_test, axis=1)) / len(self.y_test)
        print("Accuracy on adversarial test examples with: {}%".format(accuracy * 100))
        return accuracy, x_test_adv


# %% FGSM


# To use the attacked images in this evaluation function we need first need to merge x_test_adv and y test in single testset and load that via the dataloader
# .......
# Note: right now we are first resizing and adding padding and then we attack the images, this doesnt really make sense, does it?


"""
    correct_total = 0
total_tested = 0
model.eval()  # Put the network in eval mode
for i, (x_batch, y_batch) in enumerate(testloader):
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # Move the data to the device that is used
    # show_sample(x_batch, y_batch)
    y_pred = model(x_batch)
    y_pred_max = torch.argmax(y_pred, dim=1)
    total_tested += len(y_batch)
    correct_total += torch.sum(torch.eq(y_pred_max, y_batch)).item()
    print(f"Correct: {correct_total}/{total_tested} ({correct_total/total_tested*100:.2f}%)")
print(f'Accuracy on the test set: {correct_total / len(test_dataset):.5f}')
"""