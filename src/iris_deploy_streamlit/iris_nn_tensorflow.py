from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Model
from keras.layers import Input, Dense
import pandas as pd
from keras.utils import to_categorical
import numpy as np

df = pd.read_csv('datasets/iris.csv')

db = df.copy()
db.drop(columns=['Id'], axis=1, inplace=True)
db.rename(columns={'Species': 'target'}, inplace=True)

# Instancia um ENCODER, que irá transformar as classes categóricas de espécies em números
encoder = preprocessing.LabelEncoder()

# aplica o encoder à coluna target e salva o resultado em nova coluna 'target_encoded'
db['target_encoded'] = encoder.fit_transform(db['target'])

X = np.array(db.drop(
    columns=['target', 'target_encoded', 'SepalLengthCm', 'SepalWidthCm'], axis=1))
y = db['target_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=30)

y_onehot = to_categorical(y_train)  # one-hot encoder
y_train = np.array(y_onehot)
# X_train = np.array(X_train)
# print(X_train)
# functional
input = Input(shape=(2,))  # instancia um tensor Keras
hidden_layers = Dense(32, activation='relu')(input)  # 32 neuronios
output = Dense(3, activation='softmax')(hidden_layers)

model = Model(inputs=input, outputs=output)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50)
model.save('model.keras')
