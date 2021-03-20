import torch
from Classifier import Classifier

class Predict:

    def __init__(self,encoded_text, fear, joy, sadness, anger):
        self.encoded_text = encoded_text
        self.fear = fear
        self.anger = anger
        self.joy = joy
        self.sadness = sadness
        self.prediction()


    def prediction(self):
        device = torch.device("cuda")
        #print("working...")
        model_fear = Classifier(4)
        model_joy = Classifier(4)
        model_sadness = Classifier(4)
        model_anger = Classifier(4)
        fear = self.fear + ".bin"
        joy = self.joy + ".bin"
        sadness = self.sadness + ".bin"
        anger = self.anger + ".bin"
        model_fear.load_state_dict(torch.load(fear))
        model_joy.load_state_dict(torch.load(joy))
        model_sadness.load_state_dict(torch.load(sadness))
        model_anger.load_state_dict(torch.load(anger))
        model_fear = model_fear.eval()
        model_joy = model_joy.eval()
        model_sadness = model_sadness.eval()
        model_anger = model_anger.eval()
        model_fear.to(device)
        model_joy.to(device)
        model_sadness.to(device)
        model_anger.to(device)
        #pred = []
        with torch.no_grad():
                input_ids = self.encoded_text['input_ids'].to(device)
                attention_mask = self.encoded_text['attention_mask'].to(device)

                output_fear = model_fear(input_ids=input_ids, attention_mask=attention_mask)
                output_joy = model_joy(input_ids=input_ids, attention_mask=attention_mask)
                output_sadness = model_sadness(input_ids=input_ids, attention_mask=attention_mask)
                output_anger = model_anger(input_ids=input_ids, attention_mask=attention_mask)

                _, prediction_fear = torch.max(output_fear, dim=1)
                _, prediction_joy = torch.max(output_joy, dim=1)
                _, prediction_sadness = torch.max(output_sadness, dim=1)
                _, prediction_anger = torch.max(output_anger, dim=1)
                #loss_function = self.loss_func(output, intensity)
                #loss.append(loss_function.item())
                #pred.append(prediction.cpu().numpy())

                #print(self.encoded_text["intensity"])


        print(f'Intensity {self.fear}: {prediction_fear}')
        print(f'Intensity {self.joy}: {prediction_joy}')
        print(f'Intensity {self.sadness}: {prediction_sadness}')
        print(f'Intensity {self.anger}: {prediction_anger}')

        return prediction_fear, prediction_joy, prediction_anger, prediction_anger