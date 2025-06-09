from utils.data_prep.data_prep import create_dataloader, embed_tokens
import csv
import tiktoken

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
    text = read_format_csv("./utils/data/short_jokes_100.csv")
    dataloader = create_dataloader(text, batch_size=1, max_length=max_length, stride=max_length, shuffle=False)
    data_iter = iter(dataloader)

    inputs, targets = next(data_iter)
    print(embed_tokens(inputs, max_length))


__main__()

