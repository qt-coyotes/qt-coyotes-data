import cv2
import glob
import os

os.environ["PYTHONHASHSEED"] = "0"


def hashes(glob_pattern):
    dictionary = {}
    for filename in glob.glob(glob_pattern, recursive=True):
        image = cv2.imread(filename)
        h = hash(image.data.tobytes())
        if h in dictionary:
            print(
                f"Warning: {filename} has same hash as {dictionary[h]}"
            )
        dictionary[h] = filename
    return set(dictionary.keys())


def main():
    chil = hashes("data/raw/CHIL/**/*.jpg")
    print(len(chil))
    chil_earlier = hashes("data/raw/CHIL-earlier/**/*.JPG")
    print(len(chil_earlier))
    assert len(chil.intersection(chil_earlier)) == 0


if __name__ == "__main__":
    main()
