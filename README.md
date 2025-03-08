All code (besides the spectogram pipeline) was implemented in Google Colab, in order to run the code in Colab, one must convert each file to .ipynb file.
After running the pipeline on converted .mp3 files, one needs to run the main.ipynb file with the desired ViT model: the code supports ViT/S-14, ViT/B-14 and ViT/L-14 models.
In main.ipynb file, simply change the variable **model** between 'vits14', 'vitb14', 'vitl14'.


In order to see attention heads of a spectogram, you need to upload a .jpeg file of the spectogram and change the **img** variable filename. 
All attention heads will appear in the folder.
