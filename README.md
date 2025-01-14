## The code for my master thesis on cuneiform sign recognition

### File Explaination:

#### Most elaborate setup: 
code/yunus_data/final_finetune_yunus_language.py
code/yunus_data/final_finetune_yunus_purevision.py 

Both files use the artificial_images that were created using the file code/yunus_data/images_from_signs.py and first train SwinBERT on these artificial images, then on the rotated and original resized training images. In the final step, the finetuned models are tested on the test dataset from the original paper. 

final_finetune_yunus_language.py includes the path to the pretrained BERT model
final_finetune_yunus_purevision.py uses the standard pretrained BERT, without any connection to the cuneiform vocabulary.

All images in the most elaborate setup were herefore preprocessed by resizing and adaptive thresholding (-> see code/yunus_data/preprocessing_photos.py) 

#### "Standard" Setups: 
finetuning_yunus_data.py was mostly used for finetuning on image datasets (whether the datasets contained the original data from big_dataset, the data that was extracted from big_dataset to fit yunus_data, and yunus_data itself). 

The corresponding "standard" pretraining file is swin_bert_pretrain_finetune.py. However, depending on the specific setup, I created some other files that do more or less the same thing, with small variations (pretrain_bertbase_20241215.py, language_data_pretrain.py, 2024-09-09.py, pretraining_yunus_data). As the database changed often during this project and I never knew if the new dataset would lead to better or worse results, I kept all the files.

#### Other files:
