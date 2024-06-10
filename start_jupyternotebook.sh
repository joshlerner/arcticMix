###########################################################
#
# Start a jupyter notebook and tell users where to connect
#
# Adapted from Wojtek Fedorko's script by J. Lerner
#
###########################################################

thishost=localhost
jupyter-lab --no-browser --ip=$thishost --notebook-dir=$PWD >& jupyter_logbook.txt &
sleep 5
echo ""
echo ""
echo "___________________________________________________________________"
echo "**    FOLLOW THE INSTRUCTIONS BELOW TO SET UP AN SSH TUNNEL      **"
echo "___________________________________________________________________"
echo ""

python3 print_instructions.py
