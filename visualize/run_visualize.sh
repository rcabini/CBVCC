#!/bin/bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/raffaella/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/raffaella/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/raffaella/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/raffaella/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate tfrc
#python /home/debian/action_rec/visualize/visualize.py
# Verifica se è stato passato un argomento
if [ $# -eq 0 ]; then
    # Se non è stato passato alcun argomento, esegui il programma Python senza argomenti
    python /home/raffaella/Documenti/USI/action_rec/visualize/visualize.py
else
    # Se è stato passato un argomento, esegui il programma Python con l'argomento passato
    python /home/raffaella/Documenti/USI/action_rec/visualize/visualize.py "$1"
fi
