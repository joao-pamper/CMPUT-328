from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from cap_submission import *

def inspect_data(
    args,
    n_imgs: int = 2,
    ):
    dataset = FlickrDataset(args)
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

    # -----
    # Train your image captioning model
    train_cap_model(args)

    # -----
    # A few quick test
    model, processor, tokenizer = load_trained_model("./" + args.name)
    print(inference('./flickr8k/images/2631625732_75b714e685.jpg', model, processor, tokenizer))
    print(inference('./flickr8k/images/1215334959_b1970965f7.jpg', model, processor, tokenizer))
    print(inference('./flickr8k/images/3695064885_a6922f06b2.jpg', model, processor, tokenizer))


if __name__ == "__main__":
    main()