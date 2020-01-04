# Chatbot
A chatbot that is able to talk about my work experience and concepts mentioned in the "Additional" part of my resume.
It uses techniques like stemmer and bag of words.
It's powered by a simple multi-layer neural network. The 2 hidden layers have 8 neurons and use relu as activation.
The output layer produces the outputs using softmax.
Gradients are calculated using crossentropy loss.
The bot understands context, provided in the greeting tag of this dataset.
## Installation
git clone/download this project,
pip3 install -r requirements.txt in your virtualenv
## Usage
This project is divided in two parts.
The first part(train.py) is responsible for training, the second one(test.py) can be used to chat with. The training data is stored in intents.json.
