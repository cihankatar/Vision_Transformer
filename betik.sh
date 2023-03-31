#!/bin/bash
#SBATCH -p barbun           # Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A ckatar           # Kullanici adi
#SBATCH -J vision           # Gonderilen isin ismi
#SBATCH -o sonuc.out        # Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:1        # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 1                # Gorev kac node'da calisacak?
#SBATCH -n 1                # Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 32  # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=0:15:00      # Sure siniri koyun.

eval "$(/truba/home/$USER/miniconda3/bin/conda shell.bash hook)"
conda activate segmentation 
cd Desktop/Github_Repo/Vision_Transformer
python ViT_Paper_Replica.py
exit
