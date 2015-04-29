#!/bin/bash

for DEPTH in 4 5
do
for R_D in 1 4 16 32 64
do
for N_PTS in 5 10 15 20
do
cat <<EOS | sbatch
#!/bin/bash

#SBATCH -J conv-$DEPTH-$R_D-$N_PTS
#SBATCH -o conv-$DEPTH-$R_D-$N_PTS.out
#SBATCH -n 4
#SBATCH -N 4
#SBATCH -p normal
#SBATCH -t 00:45:00
##SBATCH --mail-user=keith@ices.utexas.edu
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
#SBATCH -A PADAS
cd ~/projects/invmed_rewrite/build/
R_DN=$(($R_D<$N_PTS*$N_PTS?$R_D:$N_PTS*$N_PTS))
ibrun ./faims -fmm_m 8 -fmm_q 10 -min_depth $DEPTH -max_depth $DEPTH -k $R_DN -R_d $RD_N -N_pts $N_PTS

exit 0
EOS
done
done
done
