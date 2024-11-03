- aruco_library: definisce delle funzioni per rilevare e definire posa e angolazione di ArUco Markers, attraverso l'uso della libreria OpenCV
- maglev_rotationCV: riceve un video in input ed usa le funzioni definite in aruco_library, per: identificare gli ID's degli ArUco Markers, rilevare posa ed orientamento, e generare un video in output con le informazioni contrassegnate (questo script si chiama maglev_rotationCV, perchè è stato applicato ad un video raffigurante un magnete levitante, ma può essere addattato a qualsiasi video contenente ArUco markers)
-ideal_harmonic_oscillator: genera la risposta rotazionale di un'oscillatore armonico in funzione di parametri fisici forniti dall'utente
-model_estimation_naive: ricava i parametri ottimali di un'oscillatore armonico a partire da un dataset continuo contenente coppie: (istante di tempo,angolo di rotazione) (è adeguato solo per dataset ideali)
-model_estimation_naive_4real_data: script model_estimation_naive applicato ad una dataset reale (mostra l'inadeguatezza della stima)
-model_estimation_simple: versione migliorata di model_estimation_naive , funziona anche per dataset reali , fintantochè i disturbi sono minimi e non sono presenti effetti non lineari nel sistema
-model_estimation_advanced: non fa veramente una stima avanzata, ma prende in causa i parametri ottimali di modelli di oscillatori a parametri variabili nel tempo, il che generalmente migliora la stima dei parametri per sistemi con disturbi e/o effetti non lineari
-Kalman:  questo script riceve delle misurazioni reali ed i parametri stimati del corrispondente modello di oscillatore armonico, dopo a partire da queste informazioni, fa il tuning del rumore di misurazione e di processo e fa il Kalman filtering, fornendo dei plot e delle informazioni numeriche per valutare la bontà del filtraggio

 
