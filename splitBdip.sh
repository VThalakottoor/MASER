awk '{if (NR%7==1) {print > "Bdip_t.txt"}}' Bdip.txt
awk '{if (NR%7==2) {print > "Bdip_Bdx.txt"}}' Bdip.txt
awk '{if (NR%7==3) {print > "Bdip_Bdy.txt"}}' Bdip.txt
awk '{if (NR%7==4) {print > "Bdip_Bdz.txt"}}' Bdip.txt
awk '{if (NR%7==5) {print > "Bdip_Mx.txt"}}' Bdip.txt
awk '{if (NR%7==6) {print > "Bdip_My.txt"}}' Bdip.txt
awk '{if (NR%7==0) {print > "Bdip_Mz.txt"}}' Bdip.txt
