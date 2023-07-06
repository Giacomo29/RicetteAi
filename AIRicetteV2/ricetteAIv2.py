import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Lettura del file JSON degli ingredienti
with open("train.json", "r") as file:
    data = json.load(file)

# Estrazione degli ingredienti
ingredienti = []
for ricetta in data:
    ingredienti.append(ricetta["ingredients"])

# Creazione del vocabolario degli ingredienti
vocabolario = set()
for ingr_list in ingredienti:
    for ingr in ingr_list:
        vocabolario.add(ingr)

# Creazione del tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(vocabolario)

# Creazione degli indici degli ingredienti
indici_ingredienti = {ingr: i + 1 for i, ingr in enumerate(tokenizer.word_index.keys())}

# Creazione dei vettori di input per le ricette
X = []
for ingr_list in ingredienti:
    ingr_indices = [indici_ingredienti.get(i, 0) for i in ingr_list]
    X.append(ingr_indices)

# Padding dei vettori di input
X_padded = pad_sequences(X)

# Verifica se il modello è stato addestrato in precedenza
try:
    # Carica il modello salvato
    model = load_model("modello_autoencoder.keras")
    print("Modello caricato.")
except (OSError, IOError):
    # Definizione del modello autoencoder
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=X_padded.shape[1]))
    model.add(LSTM(128))
    model.add(Dense(X_padded.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    # Allenamento del modello autoencoder
    model.fit(X_padded, X_padded, epochs=10, batch_size=16)

    # Salva il modello addestrato su disco
    model.save("modello_autoencoder.keras")
    print("Modello salvato.")

while True:
    # Richiedi nuovi ingredienti all'utente
    nuovi_ingredienti = input("Inserisci nuovi ingredienti (separati da virgola): ")
    nuovi_ingredienti = [ingrediente.strip() for ingrediente in nuovi_ingredienti.split(",")]

    # Preprocessa i nuovi ingredienti
    nuovi_ingredienti_seq = tokenizer.texts_to_sequences([nuovi_ingredienti])
    nuovi_ingredienti_padded = pad_sequences(nuovi_ingredienti_seq, maxlen=X_padded.shape[1])
    prediction = model.predict(nuovi_ingredienti_padded)

    # Calcola la media delle predizioni per ogni ingrediente
    mean_prediction = np.mean(prediction, axis=0)

    # Trova l'ingrediente con il valore massimo nella media delle predizioni
    max_index = np.argmax(mean_prediction)

    # Trova la ricetta corrispondente all'ingrediente con il valore massimo
    ricetta_consigliata = data[max_sum_index]["id"]

    print("Ricetta consigliata:", ricetta_consigliata)
