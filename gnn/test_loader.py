from json_graph_dataset import JsonGraphDataset

def main():
    dataset = JsonGraphDataset("sample_dataset.json")  # or .jsonl
    print(dataset)
    print(dataset[0].x.shape)  # should match SentenceTransformer embedding dim

if __name__ == "__main__":
    main()
