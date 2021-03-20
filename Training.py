import pandas
import seaborn as seaborn
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
import numpy

class Training:


    def __init__(self, model, data_loader_train, data_loader_test, loss_func, optimizer, scheduler, data_train, data_test):
        self.model = model
        self.data_loader_train = data_loader_train
        self.data_loader_test = data_loader_test
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_train = data_train
        self.data_test = data_test

    def flatten (self, tensors):
        return [j for tensor in tensors for j in tensor]



    def confusion_matrix(self, actual, pred):
        y_true = self.flatten(actual)
        y_pred = self.flatten(pred)
        # clf = SVC(random_state=0)
        # clf.fit(y_true, y_pred)
        # plot_confusion_matrix(clf, y_true, y_pred)
        cf = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2, 3])
        cm = pandas.DataFrame(cf, index=[i for i in "0123"],
                              columns=[i for i in "0123"])
        plt.figure(figsize=(5, 5))
        seaborn.heatmap(cm, annot=True)
        print(cf)
        plt.show()

    def euclide (self, actual):
        zero = []
        three =[]
        for i in range(len(actual)):
            zero.append(0)
            three.append(3)
        return zero, three

    def alternate_f1(self, pred, actual):
        neo_actual = []
        neo_pred = []
        actual = self.flatten(actual)
        pred = self.flatten(pred)
        for i, j in zip(pred, actual):
            if(i==0 and j == 0):
                neo_actual.append(j)
                neo_pred.append(j)
            elif (i != 0 and j != 0 and abs(i - j) < 2):
                neo_actual.append(j)
                neo_pred.append(j)
            else:
                neo_actual.append(j)
                neo_pred.append(i)

        return f1_score(neo_actual, neo_pred, average='macro')






    def training_model(self, device):

        model = self.model.train()
        loss = []
        pred = []
        actual = []

        for datal in self.data_loader_train:
            input_ids = datal['input_ids'].to(device)
            attention_mask = datal['attention_mask'].to(device)
            # emotion = datal["Emotion"]  # not used because of a design change
            intensity = datal["intensity"].to(device)

            print(datal["intensity"])

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            _, prediction = torch.max(output, dim=1)
            print(prediction)

            pred.append(prediction.cpu().numpy())
            actual.append(datal["intensity"].numpy())
            loss_function = self.loss_func(output, intensity)
            loss.append(loss_function.item())

            # correct_classifications = f1_score(actual, pred, average='micro')
            loss_function.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        #self.confusion_matrix(actual, pred)
        print(f'The custom f1-score {self.alternate_f1(actual, pred)}')
        return f1_score(self.flatten(actual), self.flatten(pred), average='macro'), np.mean(loss)


    def testing_model(self, device):
        model = self.model.eval()
        loss = []
        pred = []
        actual = []
        with torch.no_grad():
            for datal in self.data_loader_test:
                input_ids = datal['input_ids'].to(device)
                attention_mask = datal['attention_mask'].to(device)
                # emotion = datal["Emotion"]
                # not used because of a design change
                intensity = datal["intensity"].to(device)

                output = model(input_ids=input_ids, attention_mask=attention_mask)

                _, prediction = torch.max(output, dim=1)
                loss_function = self.loss_func(output, intensity)
                loss.append(loss_function.item())
                pred.append(prediction.cpu().numpy())
                actual.append(datal["intensity"].numpy())
                print(datal["intensity"])
                print(prediction)
        self.confusion_matrix(actual, pred)
        print(f'The custom f1-score {self.alternate_f1(actual, pred)}')
        return f1_score(self.flatten(actual), self.flatten(pred), average='macro'), np.mean(loss)







