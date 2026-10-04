#!/bin/bash
# Monitor headswap_final ogni 15 minuti
OUT_DIR="/Users/francescosaf/Downloads/FaceSwapAlternative"
LOG_FILE="$OUT_DIR/headswap_monitor.log"

echo "=== Monitor avviato $(date) ===" >> "$LOG_FILE"
while true; do
    sleep 900
    echo "--- $(date) ---" >> "$LOG_FILE"
    for i in 01 02 03; do
        pid_file="/tmp/final_$i.pid"
        # Verifica file output
        if [ -f "$OUT_DIR/headswap_final_$i.mp4" ]; then
            size=$(ls -lh "$OUT_DIR/headswap_final_$i.mp4" | awk '{print $5}')
            echo "[$i] COMPLETATO: $OUT_DIR/headswap_final_$i.mp4 ($size)" >> "$LOG_FILE"
            # Controlla anche log processo
            tail -2 /tmp/final_$i.log >> "$LOG_FILE" 2>/dev/null
        else
            # Verifica se processo è ancora attivo (usiamo pgrep con nome python)
            count=$(pgrep -f "run.py.*foto.$([ $i -eq 1 ] && echo 1 || [ $i -eq 2 ] && echo 2 || echo 3).*mp4" | wc -l)
            # Se non troviamo, controlliamo il PID storico se disponibile
            echo "[$i] IN CORSO: file non ancora creato (processo attivo? pgrep count=$count)" >> "$LOG_FILE"
            tail -2 /tmp/final_$i.log >> "$LOG_FILE" 2>/dev/null
        fi
        echo "" >> "$LOG_FILE"
    done
    # Aggiungi breve stato progressi (se log esiste)
    for i in 01 02 03; do
        if [ -f "/tmp/final_$i.log" ]; then
            last_line=$(tail -1 /tmp/final_$i.log)
            echo "[$i] Ultimo stato: $last_line" >> "$LOG_FILE"
        fi
    done
done
