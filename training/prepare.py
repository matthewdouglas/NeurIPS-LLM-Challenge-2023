from datasets import load_dataset
dataset = load_dataset("mosaicml/instruct-v3")
dataset = dataset.filter(lambda x: x["source"] != "dolly_hhrlhf")
dataset.save_to_disk("./instruct-data")
