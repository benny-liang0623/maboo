from visualization import createHTML
from torch.utils.data.dataset import Dataset
from transformers import BertTokenizer


def print_text_samples(dataset: Dataset, encoder: BertTokenizer, indices, export_file, att_heads=None, weights=None,
                       title=''):
    """Print text samples of dataset specified by indices to export_file text file."""

    export_txt = export_file + '.txt'
    txt_file = open(export_txt, 'a')

    if title:
        txt_file.write(f'{title}\n\n')

    texts = []
    texts_weights = []
    i = 1
    for idx in indices:
        tokens = dataset[idx]['ids']
        text = encoder.decode(tokens, True)

        if att_heads is not None:
            att_head = att_heads[i-1]
            txt_file.write(f'{i:02}. (h{att_head:02})\n {text}\n\n')
        else:
            txt_file.write(f'{i:02}.\n {text}\n\n')

        if weights is not None:
            texts_weights.append(weights[i-1][:len(tokens)])
        texts.append(text)

        i += 1

    txt_file.close()

    if weights is not None:
        export_html = export_file + '.html'
        createHTML(texts, att_heads, texts_weights, export_html)

    return