import threading
import time

def funzione_periodica(stop_event):
    while not stop_event.is_set():
        print("Funzione chiamata")
        time.sleep(0.1)  # pausa di 100ms

def sub_function():
    # Creiamo un Event per poter fermare il thread
    stop_event = threading.Event()

    # Avviamo il thread e gli passiamo lo stop_event
    thread = threading.Thread(target=funzione_periodica, args=(stop_event,))
    thread.start()

    # Esegui altre operazioni all'interno della funzione
    for i in range(5):
        print("Sotto funzione in esecuzione")
        time.sleep(1)

    # Al termine, segnaliamo al thread di fermarsi e attendiamo la sua terminazione
    stop_event.set()
    thread.join()


def main():
    while True:
        print("Inizio del programma principale")
        time.sleep(2)
    # Chiamata della funzione che include il thread periodico
        sub_function()
        print("Fine del programma principale")


if __name__ == "__main__":
    main()