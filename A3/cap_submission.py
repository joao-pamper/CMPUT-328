from functools import partial

import torch
from PIL import Image
from torch.utils.data import Dataset
from nltk.translate.bleu_score import corpus_bleu
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import default_data_collator, AutoProcessor, AutoTokenizer, VisionEncoderDecoderModel


class Args:
    """Configuration.
    """
    # Encoder-Decoder for captioning
    encoder = None
    decoder = None

    # Dataset path
    root_dir = "./flickr8k"

    # Save your model as "cap-vlm-{YOUR_CCID}"
    YOUR_CCID = "amaralpe"
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
        
        # Load the data into lines
        with open(f"{args.root_dir}/{mode}.txt", "r") as f:
            lines = f.readlines()

        # Parse image paths and captions
        self.img_paths = []
        self.captions = []
        for line in lines:
            img_path, caption = line.strip().split(";")  # Assuming tab-separated file
            self.img_paths.append(f"{args.root_dir}/images/{img_path}")
            self.captions.append(caption)
        
        # Initialize vision encoder's processor
        # Initialize langauge decoder's tokenizer
        self.processor = processor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load and process image-caption data
        # Load image and process it
        image = Image.open(self.img_paths[idx]).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values

        # Tokenize the caption
        # Add the padding token to the GPT-2 tokenizer
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # labels = self.tokenizer(
        #     self.captions[idx],
        #     return_tensors="pt",
        #     padding="max_length",
        #     truncation=True,
        #     max_length=self.args.max_length
        # ).input_ids

        labels = self.tokenizer(
            f"<|beginoftext|> {self.captions} <|endoftext|>",
            padding="max_length",
            truncation=True,
            max_length=self.args.max_length,
            return_tensors="pt"
        ).input_ids
        
        encoding = {
            "pixel_values": pixel_values.squeeze(0),       # Return processed image as a tensor
            "labels": labels.squeeze(0),             # Return tokenized caption as a padded tensor
            "path": self.img_paths[idx],
            "captions": self.captions[idx],
        }

        return encoding

    
def train_cap_model(args):
    # Define your vision processor and language tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    processor = AutoProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    # Define your Image Captioning model using Vision-Encoder-Decoder model
    # Reference: https://huggingface.co/docs/transformers/en/model_doc/vision-encoder-decoder
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        "google/vit-base-patch16-224-in21k", "gpt2"
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    # Modify the embedding lookup table in decoder model and the tokenizer
    # to include bos_token "<|beginoftext|>" and pad_token "<|pad|>"
    # NOTE: The format of GPT2 inputs:
    # <|endoftext|> + article + " TL;DR: " + summary + <|endoftext|>
    # For captoning, we want:
    # <|beginoftext|> + caption + <|endoftext|>
    # followed by a number of paddings "<|pad|>"
    # Modifying GPT-2 tokenizer to add special tokens
    special_tokens_dict = {
    'bos_token': '<|beginoftext|>',
    'eos_token': '<|endoftext|>',
    'pad_token': '<|pad|>'
    }

    # Add the special tokens to the tokenizer's vocabulary
    tokenizer.add_special_tokens(special_tokens_dict)
    
    # Resize the decoder’s embeddings to match the updated tokenizer size
    model.decoder.resize_token_embeddings(len(tokenizer))



    # Load train/val dataset
    train_dataset = FlickrDataset(args, mode="train", tokenizer=tokenizer, processor=processor)
    val_dataset = FlickrDataset(args, mode="val", tokenizer=tokenizer, processor=processor)

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

    # Define training arguments for Seq2Seq model (Seq2SeqTrainingArguments)
    # Reference: https://huggingface.co/docs/transformers/en/main_classes/trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        eval_strategy="steps",
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_steps=500,
        eval_steps=500,
        learning_rate=args.lr,
        load_best_model_at_end=True,
        save_total_limit=1,
        report_to="none"  # Disables WandB reporting
    )

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
    # Load your model configuration
    # Load the model configuration
    config = VisionEncoderDecoderModel.from_pretrained(ckpt_dir).config

    # Load encoder processor
    processor = AutoProcessor.from_pretrained(ckpt_dir)

    # Load decoder tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    
    # Load your best trained model
    model = VisionEncoderDecoderModel.from_pretrained(ckpt_dir, config=config)

    
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
    img_tensor = processor(images=image, return_tensors="pt").pixel_values()#.to("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure your img_tensor is on GPU
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    # Generate the caption with VisionEncoderDecoderModel's generate API
    generated_ids = model.generate(img_tensor, max_length=45, num_beams=5)



    # Tokens -> Str
    generated_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

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
