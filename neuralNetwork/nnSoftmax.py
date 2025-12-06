# see previous SoftMax File, this is the better version now
# improvements, more data, normalized data using sci-kit learn
# more layers, hidden ones of size 32 and 16: 9 -> 32 -> 16 -> 1
import torch
import torch.nn as nn
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 0) Prepare datasets
mysteries = []
for file in os.listdir("mysteries/JSON"):
    with open(f"mysteries/JSON/{file}", 'r') as f:
        mysteries.append(json.load(f))

# New structure: group suspects by mystery
X = []  # features for all suspects
y = []  # culprit index for each mystery
mysteryLengths = []  # number of suspects per mystery
trainAccuracies = [] # For visualization of accuracies

for mystery in mysteries:
    suspectFeatures = []
    culpritId = mystery['culprit_id'] # LLM switched up somewhere, now index 0
    y.append(culpritId)
    count = 0
    for suspect in mystery['suspects']:
        count += 1
        numClues = 0
        totalMisdirection = 0.0 #NEW DATA
        avgMisdirection = 0.0 #NEW DATA
        
        for clue in mystery['crime_scene']['clues']:
            if suspect['id'] in clue['related_suspects']: 
                numClues += 1
                totalMisdirection += clue.get('misdirection_probability', 0) # Getting the misdirection Prob
        
        if numClues > 0:
            avgMisdirection = totalMisdirection / numClues #also get avg, important siince numClues
        
        sharedClueCount = 0 # amount of clues shared since value may go down
        for clue in mystery['crime_scene']['clues']:
            l = len(clue['related_suspects'])
            if suspect['id'] in clue['related_suspects']:
                sharedClueCount += l - 1
        
        motiveAlibiRatio = suspect['motive_strength'] / (suspect['alibi_strength'] + 0.01 )#to make sure not neg
        # J WAY MORE INFO
        suspicionScore = suspect['motive_strength'] * (1 - suspect['alibi_strength']) 
        # print(suspicionScore) # make sure in between 0 and 1

        suspectFeatures.append([
            suspect['motive_strength'],
            suspect['alibi_strength'],
            numClues,
            avgMisdirection, # NEW
            sharedClueCount, # NEW
            motiveAlibiRatio,# NEW
            suspicionScore, # NEW
            len(suspect.get('traits', [])), # NEW, since cant feed actual traits
            len(suspect.get('habits', [])), # NEW, since cant feed actual habits
        ])
    
    X.append(suspectFeatures)
    mysteryLengths.append(count)

XNorm = []
# okay one thing i have learned is that this is useful because the order of mag
# differs in all my features, so normalization brings them all to the same level
# otherwise bias w domination
scaler = StandardScaler()

# flatten to get avgs
allFeatures = []
for mystery_features in X:
    allFeatures.extend(mystery_features) 

scaler.fit(allFeatures) # scale em

for mysteryFeatures in X:
    normalized = scaler.transform(mysteryFeatures) # norm = (val - mean) / std
    XNorm.append(normalized.tolist())

X = XNorm

l = len(mysteries)
indices = [i for i in range(l)]
trainInd, testInd = train_test_split(indices, test_size=0.2, random_state=42) # get it!
# print(trainInd, testInd)

def arrGetter(L, indL):
    return [L[i] for i in indL]

XTrain = arrGetter(X, trainInd)
XTest = arrGetter(X, testInd)
yTrain = arrGetter(y, trainInd)
yTest = arrGetter(y, testInd)
trainLengths = arrGetter(mysteryLengths, trainInd)
testLengths = arrGetter(mysteryLengths, testInd)

# 1) Model, OOP! Hella more layers 
class SoftmaxModel(nn.Module):
    def __init__(self, inputLen, layer1, layer2):
        super().__init__()
        self.fc1 = nn.Linear(inputLen, layer1) # 9 -> 32
        self.bn1 = nn.BatchNorm1d(layer1) # normalizing!
        self.fc2 = nn.Linear(layer1, layer2) # 32 -> 16, chosen because of hourglass shape
        self.bn2 = nn.BatchNorm1d(layer2) # normalizing!
        self.fc3 = nn.Linear(layer2, 1)# 16 -> 1
        self.dropout = nn.Dropout(0.3) # explained in .eval(), js turns of 30% of neurons
        
    def forward(self, x):
        sample = x.dim() == 1 # reshaping
        if sample:
            x = x.unsqueeze(0)
            
        x = torch.relu(self.bn1(self.fc1(x))) # LT, norm, and activation
        x = self.dropout(x) 
        x = torch.relu(self.bn2(self.fc2(x))) # LT, norm, and activation
        scores = self.fc3(x)
        
        scores = scores.squeeze(0)
            
        return scores.squeeze(-1) # j removing the dim 1 part of the vector

model = SoftmaxModel(9, 32, 16)

# 2) Loss and optimizer
learningRate = 0.001 # since way larger num neurons
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

if __name__ == '__main__': # needed because flask was never running the server
    numEpochs = 200

    # 3) Training loop
    for epoch in range(numEpochs):
        totalLoss = 0.0
        correctEpoch = 0 # number solved correctly
        
        l = len(XTrain)
        for ind in range(l):
            
            xBatch = torch.tensor(XTrain[ind], dtype=torch.float32)
            yBatch = torch.tensor(yTrain[ind], dtype=torch.long)
            
            # Forward pass
            if yBatch < 0:
                continue
            # check added after lecture thursday
            
            scores = model(xBatch) # raw scores
            loss = criterion(scores.unsqueeze(0), yBatch.unsqueeze(0)) # finding loss between expected vs actual
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            totalLoss += loss.item() # across all stories
            
            # measure accuracy during training, if person guess is correct
            prob = torch.softmax(scores, dim=0)
            probNP = prob.detach().numpy()
            # print(f'Train: {probNP}')
            
            maxInd = 0
            maxVal = probNP[0]
            m = len(probNP)
            
            for j in range(1, m):
                if probNP[j] > maxVal:
                    maxVal = probNP[j]
                    maxInd = j
            
            if maxInd == yBatch.item():
                correctEpoch += 1
            
        acc = correctEpoch / l
        trainAccuracies.append(acc) # to graph, j for visual
        
        if (epoch + 1) % 10 == 0:
            avgLoss = totalLoss / l
            # print(f'Epoch {epoch+1}/{numEpochs}, Loss: {avgLoss:.4f}')

    # Evaluation
    model.eval() 
    # another thing i found during research, this makes it so it goes from some neurons not used
    # "Drop Out" to all usage
    total = len(XTest)
    correct = 0

    with torch.no_grad():
        for ind in range(total):
            features = XTest[ind]
            target = yTest[ind]

            xBatch = torch.tensor(features, dtype=torch.float32)

            scores = model(xBatch)
            probs = torch.softmax(scores, dim=0)
            probNP = probs.detach().numpy()
            # print(f'Test: {probNP}')
            
            maxInd = 0
            maxVal = probNP[0]
            m = len(probNP)
            
            for j in range(1, m):
                if probNP[j] > maxVal:
                    maxVal = probNP[j]
                    maxInd = j
            
            if maxInd == target:
                correct += 1

    testAccuracy = correct / total
    print(f"\nTest Accuracy: {testAccuracy * 100:.2f}% ({correct}/{total})")
    # Needed help displaying, used AI here for this
    plt.plot(range(1, numEpochs+1), trainAccuracies, marker='x')
    plt.xlabel("Number")
    plt.ylabel("Training Accuracy")
    plt.title("Training Accuracy Over Time")
    plt.grid(True)
    plt.show()

def predictOnStep(mysteryData, model, scaler): 
    # get prob of each suspect after process
    # same as code before
    suspectFeatures = []
    
    # Loop through each suspect
    for suspect in mysteryData['suspects']:
        numClues = 0
        totalMisdirection = 0  # Initialize this!
        avgMisdirection = 0
        avgRelevance = 0
        avgSubtlety = 0
        
        for clue in mysteryData['crime_scene']['clues']:
            if suspect['id'] in clue['related_suspects']: 
                numClues += 1
                totalMisdirection += clue.get('misdirection_probability', 0) # Getting the misdirection Prob
        
        if numClues > 0:
            avgMisdirection = totalMisdirection / numClues #also get avg, important siince numClues
        
        sharedClueCount = 0 # amount of clues shared since value may go down
        for clue in mysteryData['crime_scene']['clues']:
            l = len(clue['related_suspects'])
            if suspect['id'] in clue['related_suspects']:
                sharedClueCount += l - 1
        
        motiveAlibiRatio = suspect['motive_strength'] / (suspect['alibi_strength'] + 0.01 )#to make sure not neg
        # J WAY MORE INFO
        suspicionScore = suspect['motive_strength'] * (1 - suspect['alibi_strength']) 
        # print(suspicionScore) # make sure in between 0 and 1

        suspectFeatures.append([
            suspect['motive_strength'],
            suspect['alibi_strength'],
            numClues,
            avgMisdirection, # NEW
            sharedClueCount, # NEW
            motiveAlibiRatio,# NEW
            suspicionScore, # NEW
            len(suspect.get('traits', [])), # NEW, since cant feed actual traits
            len(suspect.get('habits', [])), # NEW, since cant feed actual habits
        ])
    
    # normalize
    normalized = scaler.transform(suspectFeatures)
    
    # prediction
    model.eval()
    with torch.no_grad():
        xBatch = torch.tensor(normalized, dtype=torch.float32)
        scores = model(xBatch)
        probs = torch.softmax(scores, dim=0)
        
    # now, we are returning as a list to be processed
    return probs.numpy().tolist()
