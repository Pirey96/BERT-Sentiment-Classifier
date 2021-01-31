import torch
from torch import nn
import numpy as np

class Training:

    def __init__(self, model, data_loader, loss_func, optimizer, scheduler, data):
        self.model = model
        self.data_loader = data_loader
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data = data

    def training_model(self):

        model = self.model.train()
        loss = []
        correct_classifications = []

        for datal in self.data_loader:
            input_ids = datal['input_ids']
            attention_mask = datal['attention_mask']
            # emotion = datal["Emotion"]  # not used because of a design change
            intensity = datal["intensity"]

            output = model(input_ids=input_ids, attention_mask=attention_mask)

            _, pred = torch.max(output, dim=1)
            loss_function = self.loss_func(output, intensity)
            loss.append(loss_function.item())

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            return correct_classifications.double() / self.data, np.mean(loss)


    def evaluate(self):
        model = self.model.eval()
        loss = []
        correct_classifications = 0

        with torch.no_grad():
            for datal in self.data_loader:
                input_ids = datal['input_ids']
                attention_mask = datal['attention_mask']
                #emotion = datal["Emotion"]  # not used because of a design change
                intensity = datal["intensity"]

                output = model(input_ids=input_ids, attention_mask=attention_mask)

                _,pred = torch.max(output, dim=1)
                loss_function = self.loss_func(output, intensity)
                loss.append(loss_function.item())
                correct_classifications = correct_classifications + torch.sum(pred == intensity)

                return correct_classifications.double() / self.data, np.mean(loss)







