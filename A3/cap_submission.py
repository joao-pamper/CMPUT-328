from functools import partial

import torch
from PIL import Image
from torch.utils.data import Dataset
from nltk.translate.bleu_score import corpus_bleu
from transformers import Seq2SeqTrainer
from transformers import default_data_collator
# ##########
# TODO: Add more imports

# ##########

class Args:
    """Configuration.
    """
    # Encoder-Decoder for captioning
    encoder = None
    decoder = None

    # Dataset path
    root_dir = "./flickr8k"

    # Save your model as "cap-vlm-{YOUR_CCID}"
    YOUR_CCID = ""
    name = f"cap-vlm-{YOUR_CCID}"

    # Hyperparameters
    batch_size = 64
    lr = 5e-5
    epochs = 5

    # Generation cfgs
    # TODO: Add more as you see fit
    num_beams = 5
    max_length = 45     # TODO: Can play around


    # Train ops
    # TODO: Add more as you see fit
    logging_steps = 50

class FlickrDataset(Dataset):
    def __init__(
        self, 
        args, 
        processor, 
        tokenizer,
        mode: str = "train",
        ):
        assert mode in ["train", "val", "test"]
        self.args = args
        # ####################
        # TODO: Load Flickr8k dataset
        # TODO: Initialize vision encoder's processor
        # TODO: Initialize langauge decoder's tokenizer
        self.processor = processor
        self.tokenizer = tokenizer

        self.img_paths, self.captions = None, None
        # ####################

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # ####################
        # TODO: Load and process image-caption data
        encoding = {
            "pixel_values": None,       # Return processed image as a tensor
            "labels": None,             # Return tokenized caption as a padded tensor
            "path": self.img_paths[idx],
            "captions": self.captions[idx],
        }
        # ####################

        return encoding

    
def train_cap_model(args):
    # Define your vision processor and language tokenizer
    tokenizer = None
    processor = None

    # Define your Image Captioning model using Vision-Encoder-Decoder model
    # Reference: https://huggingface.co/docs/transformers/en/model_doc/vision-encoder-decoder
    model = None    # NOTE: Send your model to GPU

    # Modify the embedding lookup table in decoder model and the tokenizer
    # to include bos_token "<|beginoftext|>" and pad_token "<|pad|>"
    # NOTE: The format of GPT2 inputs:
    # <|endoftext|> + article + " TL;DR: " + summary + <|endoftext|>
    # For captoning, we want:
    # <|beginoftext|> + caption + <|endoftext|>
    # followed by a number of paddings "<|pad|>"



    # Load train/val dataset
    train_dataset = FlickrDataset(args, "train", tokenizer=tokenizer, processor=processor)
    val_dataset = FlickrDataset(args, "val", tokenizer=tokenizer, processor=processor)

    # Model configuration. 
    # Reference: https://huggingface.co/docs/transformers/en/model_doc/vision-encoder-decoder
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # TODO: Play around with some generation config parameters
    # e.g. For beam search, you can potentially have a larger beam size of 5
    # Add more as you see fit
    model.generation_config.max_length = args.max_length #None
    model.generation_config.num_beams = args.num_beams #None

    # TODO: Define training arguments for Seq2Seq model (Seq2SeqTrainingArguments)
    # Reference: https://huggingface.co/docs/transformers/en/main_classes/trainer
    training_args = None

    # Instantiate seq2seq model trainer
    compute_metrics = partial(compute_bleu_score, tokenizer=tokenizer)
    trainer = Seq2SeqTrainer(
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
    )

    # Start training
    # TODO: A good performing model should easily reach a BLEU score above 0.07
    trainer.train()
    trainer.save_model(args.name)
    

def load_trained_model(
    ckpt_dir: str,
    ):
    """TODO: Load your best trained model, processor and tokenizer.
    """
    # TODO: Load your model configuration
    config = None

    # TODO: Load encoder processor
    processor = None

    # TODO: Load decoder tokenizer
    tokenizer = None
    
    # TODO: Load your best trained model
    model = None
    if torch.cuda.is_available():
        model = model.cuda()

    return model, processor, tokenizer

def inference(
    img_path,
    model, 
    processor,
    tokenizer,
    ):
    """TODO: Example inference function to predict a caption for an image.
    """
    # TODO: Load and process the image
    image = Image.open(img_path).convert("RGB")
    img_tensor = None   # TODO: Preproces the image

    # Ensure your img_tensor is on GPU
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    # TODO: Generate the caption with VisionEncoderDecoderModel's generate API
    generated_ids = None

    # Tokens -> Str
    generated_caption = None

    return generated_caption

def compute_bleu_score(pred, tokenizer):
    """NOTE: DO NOT CHANGE.
    Compute BLEU score.
    NOTE: if you are interested in learning about the BLEU score, here are some interesting resources:
    https://www.geeksforgeeks.org/nlp-bleu-score-for-evaluating-neural-machine-translation-python/
    https://cloud.google.com/translate/automl/docs/evaluate#interpretation
    https://www.nltk.org/api/nltk.translate.bleu_score.html
    """

    pred_ids = pred.predictions
    labels_ids = pred.label_ids#.squeeze(1)

    # Decode predictions and labels while handling special tokens and padding
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == tokenizer.pad_token_id] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # Prepare data for BLEU score calculation
    pred_bleu = [line.split() for line in pred_str]
    label_bleu = [[line.split()] for line in label_str]

    # Calculate BLEU score
    bleu_output = corpus_bleu(label_bleu, pred_bleu)
    bleu_score = round(bleu_output, 4)
    print("BLEU:", bleu_score)

    return {
        "bleu_score": bleu_score
    }
