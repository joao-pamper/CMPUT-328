from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from cap_submission import *

def inspect_data(
    args,
    n_imgs: int = 2,
    ):

    # TODO:
    # Define your vision processor and language tokenizer
    # You do not need to submit this part, or the ``cap_main.py'' file. 
    # This is simply here for debugging purpose.
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    processor = AutoProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    dataset = FlickrDataset(args, tokenizer=tokenizer, processor=processor)
    indices = np.random.randint(0, len(dataset), size=(n_imgs, ))
    # Visualize with matplotlib
    for i, idx in enumerate(indices):
        encoding = dataset[idx]
        print(encoding["labels"])
        print(encoding["captions"])
        img = np.array(Image.open(encoding["path"]))

        plt.subplot(1, n_imgs, i + 1)
        plt.imshow(img, cmap="gray")
        print()

    plt.show()
    del dataset


def main():
    args = Args()
    try:
        import paramparse
        paramparse.process(args)
    except ImportError:
        print("WARNING: You have not installed paramparse. Please manually edit the arguments.")

    # -----
    # NOTE: Always inspect your data
    inspect_data(args)
    print(' after inspect; before train cap')
    # -----
    # Train your image captioning model
    train_cap_model(args)
    print("after train_cap an before load")
    # -----
    # A few quick test
    model, processor, tokenizer = load_trained_model("./" + args.name)
    print(inference('./flickr8k/images/2631625732_75b714e685.jpg', model, processor, tokenizer))
    print(inference('./flickr8k/images/1215334959_b1970965f7.jpg', model, processor, tokenizer))
    print(inference('./flickr8k/images/3695064885_a6922f06b2.jpg', model, processor, tokenizer))


if __name__ == "__main__":
    main()