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
excluding_unsure_tokens/.. -> used to generate the code of big_dataset without unsure tokens
code/yunus_data/create_big_augment -> original resized images are rotated and randomly preprocessed
code/yunus_data/images_from_signs -> create synthetic data from small images
code/yunus_data/preprocessing_data -> coded this to check if the actual ground truth transliterations for yunus_data could be deduced from big_dataset
code/yunus_data/preprocessing_images -> all image preprocessing related stuff: Cutting images, sorting bounding boxes, resizing, applying preprocessing like erosion or adaptive thresholding 
code/exploring_data -> exploring data downloaded from Cobanoglu paper & downloading from database & checking if more observations from big_dataset can be used for yunus_dataset. 

