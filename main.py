from utils.data_prep.data_prep import create_dataloader
import csv
import tiktoken
from utils.gpt.gpt import AaronGPTModel

def read_format_csv(file_path):
    jokes = ""
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i != 0:
                jokes += row[1] + " <|EOS|> "
    return jokes


def __main__():
    max_length = 4
    # text = read_format_csv("./utils/data/short_jokes_100.csv")

    text = "Hello my name is Aaron, next week I will be free. Today, I went to the mall."
    dataloader = create_dataloader(text, batch_size=2, max_length=max_length, stride=max_length, shuffle=False)
    data_iter = iter(dataloader)

    inputs, targets = next(data_iter)
    
    model = AaronGPTModel()

    out = model(inputs)

    print(inputs)
    print(out.shape)
    print(out)


__main__()

