from textblob import TextBlob
from tkinter import *
import tkinter as tk
from textblob import TextBlob
import nltk
from csv import *
from tkinter import messagebox
import pandas as pd
import gensim
import seaborn as sns
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.stem import PorterStemmer
from spellchecker import SpellChecker
from textblob import Word
from PIL import ImageTk,Image 
from gensim.models import KeyedVectors #Load the Stanford GloVe model
from gensim.scripts.glove2word2vec import glove2word2vec
import matplotlib.pyplot as plt
import chart_studio.plotly as py
import plotly.tools as tls
import matplotlib.pyplot as plt 
import numpy as np 
import cv2

#Setting Window
win = Tk()
win.title("Sentiment Analysis")
win.geometry("800x600+50+50")
win.config(bg='white')

#Window First Label
label1=Label(win, text="Welcome to Sentiment Analysis", font=("Calibri",18,"bold"), bg='white')
label1.pack(pady=15)

#Adding Images
pos = PhotoImage(file='pos.png')
neg = PhotoImage(file='neg.png')

#Window 1 Text Analysis
def FirstWindow():
	win1 = Tk()
	win1.title("Text Analysis")
	win1.geometry("800x600+50+50")
	win1.focus_set()
	win1.config(bg='white')
	
	#First Label
	label1=Label(win1, text="Text Analysis", font=("Calibri",18,"bold"), bg='white', fg="black")
	label1.pack(pady=15)
	
	#Second Label
	label2=Label(win1,text="Enter your text: ",font=("Calibri",16,"bold"), bg='white')
	label2.pack(pady=30)
	
	#Text Field
	entry1=Entry(win1,font=("Calibri",16))
	entry1.pack(pady=2)
	
	#Button Action
	def disp():
		global n
		n=entry1.get()
		#Sentiment Analysis
		sa=TextBlob(n)
		result=sa.sentiment.polarity
		if result > 0:
			result=result*100
			rs="Entered Text is "+str(result)+"% Positive."
			label3=Label(win1,text=rs,font=("Calibri",16), bg='white')
			label3.pack()
		elif result == 0:
			rs="Entered Text is Neutral."
			label3=Label(win1,text=rs,font=("Calibri",16), bg='white')
			label3.pack()
		else:
			result=result*(-1)
			result=result*100
			rs="Entered Text is "+str(result)+"% Negative."
			label3=Label(win1,text=rs,font=("Calibri",16), bg='white')
			label3.pack()
	
	#Button
	button1=Button(win1,text="Analyze",width=10,height=2, font=("Calibri",16,"bold"),bg="black",fg="white", command=disp)
	button1.pack(pady=5)
	
	win1.mainloop()
	
#Window Button1
button1=Button(win,text="Text Analysis",width=15,height=2, font=("Calibri",16,"bold"),bg="black",fg="white", command=FirstWindow)
button1.pack(pady=50)

#Window 2 File Analysis
def SecondWindow():
	win2 = Tk()
	win2.title("File Analysis")
	win2.geometry("800x600+50+50")
	win2.focus_set()
	win2.config(bg='white')
	
	#First Label
	label1=Label(win2, text="File Analysis", font=("Calibri",18,"bold"), bg='white', fg="black")
	label1.pack(pady=15)
	
	#Frame1
	frame1=Frame(win2,bg='white')
	frame1.pack()
	
	#Second Label
	label2=Label(frame1,text="Insert the File: ",font=("Calibri",16,"bold"),bg='white')
	label2.pack(pady=10)
	
	#Text Field
	dt = StringVar(win2, value='movie.csv')
	entry1=Entry(frame1, textvariable=dt, font=("Calibri",16))
	entry1.pack(side=LEFT)
	
	#Button1
	button1=Button(frame1,text="Insert",width=8,height=1, font=("Calibri",16,"bold"),bg="black",fg="white", command=None)
	button1.pack(padx=2, side=LEFT)
	
	#Button2 Action Sentiment Analysis
	def filesentimentaction():
		global file
		file=pd.read_csv('movie.csv')	
				
		#Lower Case
		file['text'] = file['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
		lowercase=file['text'].head()
		
		#Remove Punctuation
		lowercase = lowercase.str.replace('[^\w\s]','')
		removepunctuation=lowercase.head()
				
		#Removal of Stop Words
		#nltk.download('wordnet')
		stop = stopwords.words('english')
		removepunctuation = removepunctuation.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
		removalofstopwords=removepunctuation.head()
				
		#Spelling correction
		spellingcorrection=removalofstopwords[:14].apply(lambda x: str(TextBlob(x).correct()))
				
		#Lematization
		spellingcorrection = spellingcorrection.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
		lematization=spellingcorrection.head()
					
		#Sentiment Analysis		
		lematization = lematization.apply(lambda x: TextBlob(x).sentiment[0])
		label3=Label(win2,text=lematization.head(),font=("Calibri",16), bg='white')
		label3.pack()
		
		#Calculation
		pos=0
		neg=0
		for val in lematization.head():
			if val > 0:
				pos=pos+1
			else:
				neg=neg+1
		rpos=(pos*100)/5
		rneg=(neg*100)/5
		frpos="File's Sentiment is "+str(rpos)+"% Positive."
		frneg="File's Sentiment is "+str(rneg)+"% Negative."
		
		#Third Label
		label3=Label(win2,text=frpos,font=("Calibri",15,"bold"),bg='white')
		label3.pack()
		
		#Fourth Label
		label4=Label(win2,text=frneg,font=("Calibri",15,"bold"),bg='white')
		label4.pack()
		
		#Frame2
		frame2=Frame(win2, bg='white')
		frame2.pack(side='bottom')
	
		#Graph Actions
		class graphactions:
			def __init__(self, ans):
				self.ans=ans
			#Simple Graph Action
			def simplegraph(self):
				plt.title("Simple Graph of Sentiment Analysis")
				plt.plot(self.ans)
				plt.xlabel('X-Axis')
				plt.ylabel('Y-Axis')
				plt.show()
			#Scatter Graph Action
			def scattergraph(self):
				plt.title("Scatter Graph of Sentiment Analysis")
				#X-Axis Values 
				x = [1,2,3,4,5] 
				#Y-Axis Values 
				y = [self.ans]
				#Plotting points as a Scatter Plot 
				plt.scatter(x, y, label= "stars", color= "red",  marker= "*", s=30) 
				plt.xlabel('X-Axis')
				plt.ylabel('Y-Axis')
				plt.show()
					
		a=lematization.head()	
		obj=graphactions(a)
		
		#Button3
		button3=Button(frame2,text="Simple Graph",width=15,height=1, font=("Calibri",16,"bold"),bg="black",fg="white", command=obj.simplegraph)
		button3.pack(side='left')
		
		#Button4
		button4=Button(frame2,text="Scatter Graph",width=15,height=1, font=("Calibri",16,"bold"),bg="black",fg="white", command=obj.scattergraph)
		button4.pack(padx=10, side='left')
			
	#Button2
	button2=Button(win2,text="Analyze",width=10,height=2, font=("Calibri",16,"bold"),bg="black",fg="white", command=filesentimentaction)
	button2.pack(pady=15)
	
	win2.mainloop()

#Window Button2
button2=Button(win,text="File Analysis",width=15,height=2, font=("Calibri",16,"bold"),bg="black",fg="white", command=SecondWindow)
button2.pack(pady=5)

#Window 3 Deep Analysis
def ThirdWindow():
	win3 = Tk()
	win3.title("File Analysis")
	win3.geometry("800x600+50+50")
	win3.focus_set()
	win3.config(bg='white')
	
	#First Label
	label1=Label(win3, text="Deep Analysis", font=("Calibri",18,"bold"), bg='white', fg="black")
	label1.pack(pady=15)
	
	#Frame1
	frame1=Frame(win3,bg='white')
	frame1.pack()
	
	#Second Label
	label2=Label(frame1,text="Insert the File: ",font=("Calibri",16,"bold"),bg='white')
	label2.pack(pady=10)
	
	#Text Field
	dt = StringVar(win3, value='movie.csv')
	entry1=Entry(frame1, textvariable=dt, font=("Calibri",16))
	entry1.pack(side=LEFT)
	
	#Button1
	button1=Button(frame1,text="Insert",width=8,height=1, font=("Calibri",16,"bold"),bg="black",fg="white", command=None)
	button1.pack(padx=2, side=LEFT)
	
	#Button2 Deep Learning Sentiment Analysis
	def deepfilesentimentaction():
		global file
		file=pd.read_csv('movie.csv')	
				
		#Lower Case
		file['text'] = file['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
		lowercase=file['text'].head()
		
		#Remove Punctuation
		lowercase = lowercase.str.replace('[^\w\s]','')
		removepunctuation=lowercase.head()
				
		#Removal of Stop Words
		#nltk.download('wordnet')
		stop = stopwords.words('english')
		removepunctuation = removepunctuation.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
		removalofstopwords=removepunctuation.head()
				
		#Spelling correction
		spellingcorrection=removalofstopwords[:14].apply(lambda x: str(TextBlob(x).correct()))
				
		#Lematization
		spellingcorrection = spellingcorrection.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
		lematization=spellingcorrection.head()
					
		#Sentiment Analysis		
		lematization = lematization.apply(lambda x: TextBlob(x).sentiment[0])
		label3=Label(win3,text=lematization.head(),font=("Calibri",16), bg='white')
		label3.pack()
		
		#Calculation
		pos=0
		neg=0
		for val in lematization.head():
			if val > 0:
				pos=pos+1
			else:
				neg=neg+1
		rpos=(pos*100)/5
		rneg=(neg*100)/5
		frpos="File's Sentiment is "+str(rpos)+"% Positive."
		frneg="File's Sentiment is "+str(rneg)+"% Negative."
		
		#Third Label
		label3=Label(win3,text=frpos,font=("Calibri",15,"bold"),bg='white')
		label3.pack()
		
		#Fourth Label
		label4=Label(win3,text=frneg,font=("Calibri",15,"bold"),bg='white')
		label4.pack()
		
		#Frame2
		frame2=Frame(win3, bg='white')
		frame2.pack(side='bottom')
	
		#Graph Actions
		class graphactions:
			def __init__(self, ans):
				self.ans=ans
			#Simple Graph Action
			def simplegraph(self):
				plt.title("Simple Graph of Sentiment Analysis")
				plt.plot(self.ans)
				plt.xlabel('X-Axis')
				plt.ylabel('Y-Axis')
				plt.show()
			#Scatter Graph Action
			def scattergraph(self):
				plt.title("Scatter Graph of Sentiment Analysis")
				#X-Axis Values 
				x = [1,2,3,4,5] 
				#Y-Axis Values 
				y = [self.ans]
				#Plotting points as a Scatter Plot 
				plt.scatter(x, y, label= "stars", color= "red",  marker= "*", s=30) 
				plt.xlabel('X-Axis')
				plt.ylabel('Y-Axis')
				plt.show()
					
		a=lematization.head()	
		obj=graphactions(a)
		
		#Button3
		button3=Button(frame2,text="Simple Graph",width=15,height=1, font=("Calibri",16,"bold"),bg="black",fg="white", command=obj.simplegraph)
		button3.pack(side='left')
		
		#Button4
		button4=Button(frame2,text="Scatter Graph",width=15,height=1, font=("Calibri",16,"bold"),bg="black",fg="white", command=obj.scattergraph)
		button4.pack(padx=10, side='left')
	
	#Button2
	button2=Button(win3,text="Deep Analyze",width=13,height=2, font=("Calibri",16,"bold"),bg="black",fg="white", command=deepfilesentimentaction)
	button2.pack(pady=15)
	win3.mainloop()
	
#Window Button3
button3=Button(win,text="Deep Analysis",width=15,height=2, font=("Calibri",16,"bold"),bg="black",fg="white", command=ThirdWindow)
button3.pack(pady=50)

win.mainloop()