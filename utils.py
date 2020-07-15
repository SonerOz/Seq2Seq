import sys
import spacy
import torch
from torchtext.data.metrics import bleu_score


################################################################################
def translate_sentence(model, sentence, german, english, device, max_length=50):
    ########## Load german tokenizer ##########
    spacy_ger = spacy.load('de_core_news_sm')

    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    ########## Begin <SOS> & End <EOS> ##########
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    text_to_indices = [german.vocab.stoi[token] for token in tokens]

    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    with torch.no_grad():
         hiddens, cells = model.encoder(sentence_tensor)

    outputs = [english.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hiddens, cells = model.decoder(
                previous_word, hiddens, cells
            )
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        
        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]

    return translated_sentence[1:]

################################################################################

def bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)

################################################################################
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

################################################################################
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])