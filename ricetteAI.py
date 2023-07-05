import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Dati di addestramento
# TODO aggiungere nuovi dati per vedere se il modello è in grado di prevedere nuove ricette
#vanno agginti senguendo il solito schema
ingredienti = [
    'farina, burro, zucchero, uova, latte',
    'pomodoro, mozzarella, basilico',
    'pollo, peperoni, cipolle',
    'riso, pollo, carote, piselli',
    'uova, pancetta, cipolle',
    # Aggiungi più dati di addestramento qui
]


#dati per le ricette e la durata devono essere messi in ordine corrispondente

ricette = [
    'Torta al cioccolato',
    'Pizza Margherita',
    'Pollo al forno',
    'Risotto al pollo',
    'Frittata di cipolle',
    # Aggiungi più ricette corrispondenti qui
]

# Durata delle ricette
durata = [
    60,
    30,
    45,
    40,
    25,
    # Aggiungi più durate delle ricette qui
]


                    ## Creazione del modello
# Creazione del tokenizer(token del modello di tensonflow)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(ingredienti)

# Sequenze di input degli ingredienti
#(vanno tokenizzate tutte le ricette (da aggiunge nuove ricette))
ingredienti_seq = tokenizer.texts_to_sequences(ingredienti)

# Padding delle sequenze degli ingredienti per uniformare la lunghezza(?)
ingredienti_padded = pad_sequences(ingredienti_seq)


#TODO
# Conversione delle etichette in numeri interi
label_encoder = LabelEncoder()
etichette_numeriche = label_encoder.fit_transform(ricette).astype(np.int32)  # Converti in np.int32

# Divisione del set di dati in set di addestramento e test
X_train, X_test, y_train, y_test, durata_train, durata_test = train_test_split(
    ingredienti_padded, etichette_numeriche, durata, test_size=0.2, random_state=42 # -> modificare per vedere se il modello migliora 
)

# Creazione dei layer di input per ingredienti e durata
ingredienti_input = Input(shape=(ingredienti_padded.shape[1],))
durata_input = Input(shape=(1,))

# Creazione del modello
embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=ingredienti_padded.shape[1])(ingredienti_input)
flatten_layer = Flatten()(embedding_layer)
concat_layer = concatenate([flatten_layer, durata_input])
dense_layer1 = Dense(64, activation='relu')(concat_layer)
output_layer = Dense(len(set(etichette_numeriche)), activation='softmax')(dense_layer1)

model = Model(inputs=[ingredienti_input, durata_input], outputs=output_layer)

# Compilazione del modello
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#ALLENAMENOT DEL MODELLO
# Addestramento del modello
model.fit([np.array(X_train), np.array(durata_train)], np.array(y_train), epochs=10, batch_size=16, validation_data=([np.array(X_test), np.array(durata_test)], np.array(y_test)))

                # Esempio di previsione su una nuova ricetta
                #TODO questo sembra non funzionare
#nuova_ricetta_ingredienti = ['pollo, patate, aglio']
nuova_ricetta_ingredienti = ['pasta, pomodoro, basilico, aglio',]
nuova_ricetta_ingredienti_seq = tokenizer.texts_to_sequences(nuova_ricetta_ingredienti)
nuova_ricetta_ingredienti_padded = pad_sequences(nuova_ricetta_ingredienti_seq, maxlen=ingredienti_padded.shape[1])
#durata_nuova_ricetta = [30]
durata_nuova_ricetta = [20]

prediction = model.predict([np.array(nuova_ricetta_ingredienti_padded), np.array(durata_nuova_ricetta)])
predicted_label = label_encoder.inverse_transform(np.argmax(prediction, axis=1))

#STAMPA RISULTATI

print('Ricetta consigliata:', predicted_label[0])
print('Durata prevista:', durata_nuova_ricetta[0], 'minuti')
