import tkinter as tk, tkinter.messagebox as messagebox
import  tensorflow as  tf,pickle
import pandas as pd
from PIL import ImageTk
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout


model = tf.keras.models.load_model('saved-model/lstm-model')

root = tk.Tk()

root.title("Fake News Prediction")
root.resizable(width=False, height=False)
canvas1 = tk.Canvas(root, width=1045, height=680, bg='Orange')
canvas1.pack(expand=tk.YES, fill=tk.BOTH)
image = ImageTk.PhotoImage(
    file="background.png")
canvas1.create_image(12, 12, image=image, anchor=tk.NW)
entry1 = tk.Entry(root, font=("Times", 15))
canvas1.create_window(520, 545, window=entry1, width=800)


def get_input():
    global data,data1
    if(len(entry1.get()) > 0):
      text = entry1.get()
      # initialize list of lists
      data = [[text]]
      df = pd.DataFrame(data, columns = ['text'])
        
      # remove special characters and punctuations
      df['clean_news'] = df['text'].str.lower()
      df['clean_news'] = df['clean_news'].str.replace('[^A-Za-z0-9\s]', '')
      df['clean_news'] = df['clean_news'].str.replace('\n', '')
      df['clean_news'] = df['clean_news'].str.replace('\s+', ' ')

      from nltk.corpus import stopwords
      nltk.download('stopwords')
      stop = stopwords.words('english')
      df['clean_news'] = df['clean_news'].apply(lambda x: " ".join([word for word in x.split() if word not in stop]))


      voc_size=5000
      onehot_repr=[one_hot(words,voc_size)for words in  df['clean_news']] 

      # Embedding representation

      sent_length=500
      embedded_docs=pad_sequences(onehot_repr,maxlen=sent_length, padding='post', truncating='post')
      predictions = model.predict(embedded_docs)
      output = "Real" if predictions[0][0] < 0.5 else "Fake";
      outputLabel = tk.Label(root, text=output, font=("Times", 12),)
      canvas1.create_window(550, 620, window=outputLabel, width=200)
      # clear the entry
      entry1.delete(0, tk.END)
      clearButton = tk.Button(text='Clear labels', width=20, height=1, bg='White', fg='Red', command=lambda: [
                              outputLabel.destroy(), clearButton.destroy()])
      canvas1.create_window(740, 620, window=clearButton, width=100)

    else:
        messagebox.showerror("Error", "Enter Values")

# Create a Button 
button1 = tk.Button(text='Predict',width=20,height=1,bg='Black',fg='white',command=get_input)
# Show it on Canvas 
canvas1.create_window(520, 580, window=button1)
label2 = tk.Label(root, text="Output Label",font=("Times", 12))
canvas1.create_window(350, 620, window=label2,width=130)
root.mainloop()