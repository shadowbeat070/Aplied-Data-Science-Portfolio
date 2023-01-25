# ADS Portfolio - Mohamed Amajoud (20198752)

**Naam:** Mohamed Amajoud\
**Studentnummer:** 20198752\
**Course:** Applied Data Science

## Inhoud

[1. Introductie](#1-Introductie)\
[2. Datacamp courses](#2-Datacamp-courses)\
[3. Research project](#3-Research-project)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[3.1 Task definition](#41-Task-definition)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[3.2 Evaluation](#42-Evaluation)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[3.3 Conclusions](#43-Conclusions)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[3.4 Planning](#44-Planning)\
[4. Predictive Analytics](#4-Predictive-Analytics)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[4.1 Selecting a Model](#51-Selecting-a-Model)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[4.2 Configuring a Model](#52-Configuring-a-Model)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[4.3 Training a model](#53-Training-a-model)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[4.4 Evaluating a model](#54-Evaluating-a-model)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[4.5 Visualizing the outcome of a model (explanatory)](#55-Visualizing-the-outcome-of-a-model-explanatory)\
[5. Domain knowledge](#5-Domain-knowledge)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[5.1 Introduction of the subject field](#61-Introduction-of-the-subject-field)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[5.2 Literature research](#62-Literature-research)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[5.3 Explanation of Terminology, jargon and definitions](#63-Explanation-of-Terminology-jargon-and-definitions)\
[6. Data preprocessing](#6-Data-preprocessing)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[6.1 Data exploration](#71-Data-exploration)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[6.2 Data cleansing](#72-Data-cleaning)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[6.3 Data preparation](#73-Data-preparation)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[6.4 Data explanation](#74-Data-explanation)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[6.5 Data visualization (exploratory)](#75-Data-visualization-exploratory)\
[7. Communication](#7-Communication)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[7.1 Presentations](#81-Presentations)\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[7.2 Writing paper](#82-Writing-paper)

# 1. Introductie

In dit portfolio lees je over mijn deelnamen aan de minor Applied Data Science als student ICT Software Engineering.

# 2. Datacamp courses

Ik heb alle nodige Datacamp courses voltooit. Ik had al een beetje Python voorkennis, maar ik heb besloten om alle courses te maken. Dit heb ik gedaan, omdat ik alleen nog maar basis kennis had over Python en ik wist niets over alle library's die gebruikt worden in data science. Ik heb geen moeilijkheden ondervonden bij het maken van de courses

<details>
<summary>Open om alle <i>Statements of Accomplishments</i> te zien</summary>

![cert](/Data/datacamp1.png)
![cert](/Data/datacamp2.png)


# 3. Research project

## 3.1 Task definition

Zoals word benoemd in [5.1 Introduction of the subject field](#51-Introduction-of-the-subject-field) zijn wij onderdeel van het Smart Teddy project. Ons doel is het classifiseren van emoties uit audio.

Onze hoofdvraag is als volgt:

> Which machine learning models achieve the highest precision classifying emotions, using datasets containing audio with labeled vocal emotional expressions recorded in a professional recording studio, in order to recognize emotions within household environments?
Zoals je ziet specificeren wij niet in de hoofdvraag dat ons project voor dementerende senioren is. Dit kunnen wij namelijk aan het einde van ons onderzoek helemaal niet bewijzen, omdat wij niet beschikken over audio fragmenten van dementerende senioren. Wel hebben wij besloten om huishoudelijke achtergrond geluiden toe te voegen aan onze audio dataset. Dit hebben wij gedaan om onze data zo goed mogelijk te laten lijken op de data hoe deze er uiteindelijk zou gaan uitzien.

## 3.2 Evaluation

Voor future work zou ik een gelabelde dataset opzetten die bestaat uit emoties opgenomen door Smart Teddy die bij een senior in het huis staat. Ik zou dan testen hoe ons model scoort op de echte data van senioren en niet op audio ingesproken door acteurs. Veder kan je ook nog ons model testen op langere samples. De datasets die wij hebben gebruikt spreken de acteurs allemaal 1 zin. Volgens de Harvard hersenwetenschapper Dr. Jill Bolte Taylor [90 Seconds to Emotional Resilience](https://www.alysonmstone.com/90-seconds-to-emotional-resilience/) duurt een emotie vaak 90 seconden. Misschien zou ons model beter reageren op langere sampels, omdat hierin dan duidelijker blijkt dat de persoon een bepaalde emotie toont.

## 3.3 Conclusions

Uit onze conclusie blijkt dat wij met een precision van 84% emoties kunnen classifiseren op onze gecombineerde dataset. Voor ons project hebben wij ook onze ook huishoudelijke achtergrond geluiden toegevoegd. Dit hebben wij gedaan om onze samples meer te laten lijken op de huishoudelijke omgeving van de senioren. Met het toevoegen van de achtergrond geluiden resulteerde dit op een precision van 80%. Dit resulteerd dus in een iets lagere precision. Hieruit kunnen wij concluderen dat ons model goed gebruikt zo kunnen worden in de zorg.

## 3.4 Planning

Als scum omgeving hebben wij GitHub. Wij hebben de GitHub issues gebruikt als stories en de GitHub milestones als sprints. Op deze manier konden wij stories aan sprints toevoegen en deze vervolgens filteren op de sprint.

![github scrum boaard](assets/images/gh-scrum-board.png)

Hier zien je ons scrum board voor sprint 8 (onze laatste sprint). Wij hebben het verdeeld in 4 colommen To do, In progress, waiting for feedback en Done. Op deze manier konden wij tijdens de daily standup gemakkelijk zien waar wij precies stonden in de sprint en welke stories stil stonden, omdat wij bijvoorbeeld op feedback van een docent moesten wachten.

Wij hadden elke dag een daily standup waarin iedereen zijn voortgang deelde met de rest van het team. Ook gebruikte wij dit moment voor als iemand vast zat met een story en hulp nodig had om veder te gaan. Dan werd er met die persoon een moment gepland om er gezamelijk naar te kijken.

Eerst hanteerde wij sprints van 1 week. Na een aantal sprints kwamen we tijdens een retrospective tot de conclusie dat dit te kort is. Wij hebben daarna sprints van 2 weken aangehouden. Op de helft van de sprint hielden wij een refinement waarin we bespraken of alles nog haalbaar was of dat er misschien een paar nieuwe stories bij zouden kunnen komen.

Na elke spring hielden wij een retrospective. Hierin bespraken wij wat er goed ging, wat er minder goed ging en wat er verbeterd moet worden. Wij gebruikte een de tool [Fraankly](https://www.fraankly.com) voor onze retrospectives. Bij deze tool hadden we eerst de tijd om annoniem kaartjes aan te maken voor wat er allemaal goed en niet goed ging. De kaartjes waaren annoniem zodat jouw kaartjes niet beinvloed zouden worden door andere. Daarna moest iedereen zijn kaartjes bespreken en moest er gestemd worden op de kaartjes waar je het meest mee eens was. Op het einde van de retroscpective maakte wij een lijst met _actie punten_. Dit waren punten die wij volgende sprint aandacht aan moeten besteden. Hier maakte wij vervolgens ook stories van zodat deze volgende sprint opgepakt zouden worden.

![retro](assets/images/retro.png)

Wij hebben als groep ook een roadmap gemaakt. Dit vonden wij allemaal erg handig om tijdens de sprint planning naar te kijken. Op deze manier wisten wij of we op schema lagen volgens onze planning. Natuurlijk haalden we dit niet altijd of hadden we dingen verkeerd ingeschat op de roadmap. Dit hebben we dan direct aangepast.

![roadmap](assets/images/roadmap.jpg)

# 4. Predictive Analytics

## 4.1 Selecting a model

Voor het project heb ik een SVM machine learning model gemaakt. Ik heb dit model gekozen, omdat ik tijdens het literatuur onderzoek [dit onderzoek](https://www.sciencedirect.com/science/article/pii/S1746809420300501) heb gevonden. In dit onderzoek word een SVM gebruikt voor het classifiseren van emotie op de RAVDESS dataset.

## 4.2 Configuring a Model

Zoals eerder benoemd heeft ieder project lid een machine learning model gekozen om te implementeren. Daarna wilde wij alle resultaten met elkaar vergelijken. Om alles makkelijk te runnen hebben we een `BaseModel` class gemaakt. Elk model overerft deze class.

```py
class BaseModel():
    def train(cls):
        raise NotImplementedError
    def grid_search():
        raise NotImplementedError
    def model_accuracy(model, X_train, X_test, y_train, y_test):
        # Train accuracy
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        print(f"Train accuracy is: {accuracy}")
        # Test accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test accuracy is: {accuracy}")
        # Precision score
        print(f"Precision: {precision_score(y_test, y_pred, average=None)}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        # Confusion matrix
        plot_confusion_matrix(model, X_test, y_test, normalize='true')
        fig1 = plt.gcf()
        plt.show()
        plt.draw
        fig1.savefig(path + dt)
```

Samen met Dr. Hani al-Ers zijn wij tot het besluit gekomen om ons te richten op precision. Om tot de beste hyper parameters te komen hebben wij een grid search uitgevoerd. Na het runnen van de grid search waren dit de hyper parameters die eruit kwamen.

```json
{
  "C": 10,
  "gamma": 0.001,
  "kernel": "rbf"
}
```

[Zie hier het notebook voor de SVM](assets/notebooks/svm.ipynb)

## 4.3 Training a model

Zoals hierboven vermeld heb ik voor het trainen een gridsearch uitgevoerd om tot de optimale hyperparameters te komen. Wij trainde onze modellen op een fysiek gesplite dataset. Dit hebben wij gedaan zodat we de modellen eerlijker kunnen vergelijken met elkaar, omdat ze allemaal op dezelfde data trainen en testen. Na het trainen word in de `BaseModel` de functie `model_accuracy()` aangeroepen. Deze print vervolgens resultaten waarmee wij later de modellen met elkaar kunnen vergelijken.

## 4.4 Evaluating a model

Zoals eerder benoemd hebben we allemaal een model gemaakt die de `BaseModel` oveerft. Daarna heb ik een `moddel_runner` ontwikkeld waarin wij heel simpel alle modellen kunnen runnen op verschillende datasets.

```py
# Dit zijn de modellen die gerunt moeten worden
models = [KNN, SVM, MLP, LogisticalRegression]
# Loaders zijn helper classen voor het laden van de datasets
loaders = [RavdessPosNegLoader, CremadPosNegLoader]
# Variants zijn de varianten op de dataset door middel van augmentatie
variants = ["OriginalPN", "TrimmedPN", "augmented_PitchDownPN", ...]
for model in models:
    for variant in variants:
        for loader in loaders:
            print(f"training {model.instance} with {variant} data loaded with {loader.identifier}")
            data = loader.load_dataset(variant)
            model.train(data)
```

Op deze manier konden wij veel testen achter elkaar runnen en de outputs loggen naar een file om deze vervolgens te evalueren ([evaluatie bestand](assets/other/evaluation.xlsx)). Met deze informatie konden wij makkelijk bekijken hoe de verschillende modelen reageren op bijvoorbeeld het cleanen of augementeren van de audio. Ook konden wij de resultaten van alle modelen gemakkelijk met elkaar vergelijken voor de evaluatie. Uiteindelijk is er voor het eindproduct geen een van deze modellen gekozen en hebben we ons de laatste tijd gefocust op de CNN die op het moment het beste presteerde.

## 4.5 Visualizing the outcome of a model (explanatory)

Voor de SVM die ik heb gemaakt voor het Smart Teddy project heb ik de uitkomst niet gevisualiseerd. Ons groepje heeft voor de learning lab de SVM besproken. Ik was verantwoordelijk voor het maken van het example notebook. Ik dit notebook heb ik wel de uitkomsten van het model gevisualiseerd. Op deze manier kon ik het effect van hyperparameters laten zien op de decision boundary van het model. [bekijk hier het notebook voor de learning lab](assets/notebooks/learning_lab.ipynb)

# 5. Domain knowledge

## 5.1 Introduction of the subject field

Ik heb tijdens dit project gewerkt aan het project Smart Teddy. Dit is een project van Dr. Hani al-Ers. De Smart Teddy moet de leef omstandigheden van dementerende senioren verbeteren. Dit kan omdat de Smart Teddy helemaal vol zit met sensoren. Wij waren verantwoordeijk voor het detecteren en classifiseren van emotie. Emotie speelt een belangrijke rol als indicatie voor het welzijn van de senioren. Waneer de senioren bijvoorbeeld vaak boos zijn of juist helemaal geen emotie tonen kan dit betekenen dat zij aan het verslechteren zijn. Helaas was er voor dit project geen echte data beschrikbaar. Wij moesten dus bestaande datasets van acteurs gebruiken. Deze acteurs spraken dan zinnen in verschillende emoties.

## 5.2 Literature research

Emoties classificeren is geen nieuw onderwerp in de wereld van data science. Er zijn al veel mensen die dit al eens gedaan hebben en daar een paper over hebben geschreven. Yuri, Jaap, Zahir en Brenno hadden een story om nuttige onderzoeken te vinden. Daarna hebben zij de meest nuttige door gestuurd naar Julian en ik die op dat moment met een andere stories bezig waren.

Ik heb vervolgens de volgende onderzoeken gelezen:

- [Speech Emotion Detection using IoT based Deep Learning for Health Care](https://ieeexplore.ieee.org/abstract/document/9005638/authors#authors)
- [A Speech Emotion Recognition Solution-based on Support Vector Machine for Children with Autism Spectrum Disorder to Help Identify Human Emotions](https://ieeexplore.ieee.org/document/9249147)

Deze twee onderzoeken gebruikte beide de RAVDESS dataset. Het verschil bij de onderzoeken zit hem vooral in de manier hoe de data word voorbereid en welk model er word gebruikt.

|                     | Speech Emotion Detection using IoT based Deep Learning for Health Care | A Speech Emotion Recognition Solution-based on Support Vector Machine for Children with Autism Spectrum Disorder to Help Identify Human Emotions |
| ------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Dataset             | RAVDESS                                                                | RAVDESS                                                                                                                                          |
| Emoties             | happines, sadness, suprised, anger, fearful, disgust                   | happines, sadness, suprised, anger, fearful, disgust                                                                                             |
| Model               | CNN                                                                    | SVM                                                                                                                                              |
| Audio preprocessing | Spectrogram                                                            | MFCC's                                                                                                                                           |
| Accuracy            | 63,9% (met achtergrond geluiden)                                       | zonder augmentation<br>- 78% vrouw <br>- 71% man <br><br>data augmentation<br>- 93% man<br>- 95% vrouw                                           |

## 5.3 Explanation of Terminology, jargon and definitions

- **Emotie**: Een emotie is een innerlijke beleving of gemoedsbeweging zoals vreugde, angst, boosheid, verdriet en kan door een situatie of gebeurtenis worden opgeroepen [bron](https://nl.wikipedia.org/wiki/Emotie)
- **Support vector machine (SVM)**: Een set van supervised learning methodes voor classificatie, regressie en outliner detection. [bron](https://scikit-learn.org/stable/modules/svm.html)
- **Convolutional Neural Network (CNN)**: Een deep learning algoritme wat gebruikt word voor beeldherkenning. [bron](https://www.techtarget.com/searchenterpriseai/definition/convolutional-neural-network)
- **Augmentatie**: Je maakt meer data door bijvoorbeeld bij elke sample de pitch of de snelheid van je audio te veranderen. [bron](https://en.wikipedia.org/wiki/Data_augmentation)
- **RAVDESS, CREMA-D, SAVEE, TESS**: Dit zijn de twee datasets die het meeste worden gebruikt bij classifiseren van emotie. Bij deze datasets worden zinnen ingesproken door acteurs in vershillende emoties. [CREMA-D](https://www.kaggle.com/ejlok1/cremad), [RAVDESS](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio), [SAVEE](https://www.kaggle.com/barelydedicated/savee-database) & [TESS](https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess)
- **Audio Features**: Het halen van zinvolle informatie uit audiosignalen. [bron](https://www.sciencedirect.com/topics/engineering/audio-feature)
- **Scrum**: Scrum is een agile framework voor het devolopen van producten in groepen/teams.
- **Inherentice (OOP)**: Inherentice is het mechanisme van het baseren van een object of klasse op een ander object of klasse, met behoud van een vergelijkbare implementatie. [bron](<https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)>)
- **Spectrogram**: Een spectrogram is een visuele weergave van het spectrum van frequenties van een audio signaal. [bron](https://en.wikipedia.org/wiki/Spectrogram)
- **MFCC**: De MFCC zijn de coëfficiënten van het Mel-cepstrum. Het Mel-cepstrum is het cepstrum berekend op de Mel-banden (geschaald naar het menselijk oor) in plaats van het Fourier-spectrum. [bron](https://www.igi-global.com/dictionary/gtm-user-modeling-aiga-weight/18257)
- **Hyper parameters**: Een waarde die effect heeft op het leer proces van het machine learning model
- **Grid search**: Grid search is een techniek die probeert de optimale waarden van hyperparameters te berekenen [bron](https://medium.com/fintechexplained/what-is-grid-search-c01fe886ef0a)
- **Precision**: Precision is een indicator van de prestaties van een machine learning-model, de kwaliteit van een positieve voorspelling van het model [bron](https://c3.ai/glossary/machine-learning/precision/)

# 6. Data preprocessing

## 6.1 Data exploration

Jaap heeft zich bezig gehouden met de data exploration. Ik heb daar veder geen bijdragen aan geleverd.

## 7.2 Data cleaning

De audio bestanden uit onze dataset beginnen en eidigen vaak met een stuk silte. Wij hebben librosa gebruikt om deze stilte eraf te trimmen. Dit leide namelijk tot een betere presatie van de models.

```py
def _process_audio(value):
    x, sample_rate = librosa.load(value['file_path'], sr=44100)
    x, index = librosa.effects.trim(x, top_db=20)
    return x, sample_rate
```

## 6.3 Data preparation

Voor data preparation/preprocessing heb ik meerde dingen gedaan.

Helemaal in het begin had ik een pipeline voor het processen van audio files naar MFCC. Zie hier het [notebook](assets/notebooks/mfcc.ipynb). Deze pipeline hebben wij in het vervolg van het project niet meer gebruikt.

Julian heeft in dit project de pipeline opgezet. Deze bestaat uit `Generators`, `DatasetLoaders` en `Processables`. Later in het project heb ik augmentation toe gevoegd aan de pipeline voor de data van de CNN. Zie hier een implementatie van de [processable met augmentation](assets/notebooks/processable.ipynb). Na het testen met de augmentation had dit alleen maar negatief effect op de CNN en hebben wij dit niet gebruikt. Veder heb ik ook nog een aantal imlementaties gemaakt op basis van de `DatasetLoader`. [zie hier het notebook](assets/notebooks/dataset_loader.ipynb). Ik heb niet alle implementaties gemaakt in dit notebook.

## 6.4 Data explanation

Voor dit project hebben wij meerdere datasets gecombineerd tot een grote dataset. Wij hebben de volgende datasets gebruikt:

1. [CREMA-D](https://www.kaggle.com/ejlok1/cremad)
2. [RAVDESS](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio)
3. [SAVEE](https://www.kaggle.com/barelydedicated/savee-database)
4. [TESS](https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess)

|                      | RAVDESS                                                                | CREMA-D                                             | TESS                                                           | SAVEE                                                          |
| -------------------- | ---------------------------------------------------------------------- | --------------------------------------------------- | -------------------------------------------------------------- | -------------------------------------------------------------- |
| Aantal samples       | 7356                                                                   | 7442                                                | 2800                                                           | 480                                                            |
| Aantal audio samples | 1440                                                                   | 7442                                                | 2800                                                           | 480                                                            |
| Aantal acteurs       | 24 (12 man, 12 vrouw)                                                  | 91 (48 man, 43 vrouw)                               | 2 (0 man, 2 vrouw)                                             | 4 (4 man, 0 vrouw)                                             |
| Emoties              | Anger<br>Disgust<br>Fear<br>Happy<br>Neutral<br>Sad<br>Calm<br>Suprise | Anger<br>Disgust<br>Fear<br>Happy<br>Neutral<br>Sad | Anger<br>Disgust<br>Fear<br>Happy<br>Neutral<br>Sad<br>Suprise | Anger<br>Disgust<br>Fear<br>Happy<br>Neutral<br>Sad<br>Suprise |

Zoals je ziet heeft niet elke dataset dezelfde emoties. Wij hebben dan ook samen met Dr. Hani al-Ers besproken om op de volgende emoties te focussen:

- Anger
- Happy
- Neutral
- Sad

## 6.5 Data visualization (exploratory)

Wij hebben dit project voornaamlijk gewerkt met audio data. Audio kan je op veel manieren weergeven. Een van de meest gebruikte manieren is een spectrogram. Hieronder zie je twee verschillende samples. De eerste is van een neutrale sample en de tweede is een angry sample. Je kunt wel verschil zien tussen de twee spectrogramen, maar audio blijft wel een van de moeilijkere dingen om af te lezen. [Dit artiekel](https://www.izotope.com/en/learn/understanding-spectrograms.html) heeft mij erg geholpen met het beter kunnen lezen van spectrogrammen.

![neutral spectrogram](assets/images/neu-spec.png)
![angry spectrogram](assets/images/ang-spec.png)

# 7. Communication

## 7.1 Presentations

Tijdens de internen presentaties nam Jaap altijd het iniatief om te presenteren. Voor de externe presentaties spraken we van te voren af wie er welk gedeelte zou presenteren. Ik heb een deel gepresenteerd bij de volgende presentaties:

- [External Presentation 1](assets/other/ex-pres1.pdf)
- [Learning Lab](assets/other/ll-pres.pdf)

## 8.2 Writing paper

Zahir, Brenno, Jaap en Yuri waren begonnen aan de paper terwijl ik nog bezig was met het prototypen. Nadat ik klaar was met het prototypen heb ik aangehaakt bij het helpen bij de paper. Wij hebben met zijn alle de paper zin voor zin grondig doorgenomen om te zorgen dat alles goed zou zijn.
