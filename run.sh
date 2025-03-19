#!/bin/bash
# Verifica che Python3 sia installato
if ! command -v python3 &> /dev/null; then
    echo "Errore: Python3 non è installato. Installalo prima di eseguire lo script."
    exit 1
fi
# Definizione delle variabili
DATASETS=("CrossSeason" "DayNight" "DepthOptical" "Infrared_Optical" "Map_Optical" "Optical_Optical" "SAR_Optical")
BASE_FOLDER="DATASET/RemoteSensing"
OUTPUT_BASE="output"
METHODS="SIFT,SURF,ORB,BRISK,AKAZE,RIFT,LGHD,MINIMA"
MODEL="sp_lg,loftr"
SCRIPT="main.py"
# Itera su ciascun dataset ed esegui il comando
for DATASET in "${DATASETS[@]}"; do
    DATA_FOLDER="$BASE_FOLDER/$DATASET"
    OUTPUT_DIR="$OUTPUT_BASE/$DATASET"  # Nome dell'output uguale al dataset

    # Esegui GT_Fixer.py prima di elaborare il dataset
    echo "Eseguendo GT_Fixer.py per $DATASET..."
    python3 GT_Fixer.py "$DATA_FOLDER"

    # Controlla se GT_Fixer è andato a buon fine
    if [ $? -ne 0 ]; then
        echo "Errore durante l'esecuzione di GT_Fixer.py per $DATASET!"
        exit 1
    fi

    echo "Eseguendo: python3 $SCRIPT --data_folder $DATA_FOLDER --visualize --methods=$METHODS --output_dir $OUTPUT_DIR"
    python3 "$SCRIPT" --data_folder "$DATA_FOLDER" --visualize --methods="$METHODS" --model "$MODEL" --output_dir "$OUTPUT_DIR"
    # Controlla se il comando è andato a buon fine
    if [ $? -eq 0 ]; then
        echo "Esecuzione completata con successo per $DATASET!"
    else
        echo "Errore durante l'esecuzione per $DATASET!"
        exit 1
    fi
done
echo "Tutti i dataset sono stati elaborati con successo!"
