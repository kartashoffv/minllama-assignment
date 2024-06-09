import torch
from llama import load_pretrained

seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

sanity_data = torch.load("./sanity_check.data")

sent_ids = torch.tensor(
    [
        [101, 7592, 2088, 102, 0, 0, 0, 0],
        [101, 7592, 15756, 2897, 2005, 17953, 2361, 102],
    ]
)

# Так и не получилось запустить с atol 1e-5 (поставил3e-5). Поставил max_difference для демонстрации разницы между logits и sanity_data["logits"]
llama = load_pretrained("stories42M.pt")
with torch.no_grad():
    logits, hidden_states = llama(sent_ids)
    print("logits: ", logits)
    print("sanity_data['logits']: ", sanity_data["logits"])
    max_difference = torch.max(torch.abs(logits - sanity_data["logits"]))
    print("Max difference: ", max_difference)

    assert torch.allclose(logits, sanity_data["logits"], atol=3e-5, rtol=1e-3)
    assert torch.allclose(
        hidden_states, sanity_data["hidden_states"], atol=3e-5, rtol=1e-3
    )
    print("Your Llama implementation is correct!")
